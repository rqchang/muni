"""
inst_trade.py
Compute institutionalization of municipal bond trades from MSRB clean files.

Definitions
-----------
- Odd lot:        par_traded <= 100,000
- Customer trade: trade_type_indicator in {"P", "S"}
- Dealer trade:   trade_type_indicator == "D"

Institutional signals
---------------------
  Flag-based (lower bound, customer odd-lots only):
    NOTE: ATS and BB are inter-dealer (D) only fields — always null on customer
    trades (P/S), so inst_channel for customer trades reduces to ntbc | wap.
    ATS:   ats_indicator == "Y"                   [D trades only]
    BB:    brokers_broker_indicator in {"P","S"}  [D trades only]
    NTBC:  ntbc_indicator == "Y"                  [customer trades; no markup charged → institutional]
    WAP:   weighted_price_indicator == "Y"        [averaged/systematic pricing → institutional]
    inst_channel = any of the above

  SMA clustering (Bagley & Vieira, MSRB 2025):
    SMAs rebalance across many client accounts simultaneously, producing clusters
    of same-direction customer trades on the same CUSIP-date that retail investors
    almost never generate.
    cluster_ct  = count of same (cusip, trade_date, trade_type_indicator, trade_hour) customer trades
                  trade_hour = floor(time_of_trade_seconds / 3600)
    inst_sma    = cluster_ct >= SMA_MIN_TRADES  (default: 3)

  Markup-based (Harris & Piwowar 2006; Green, Hollifield & Schürhoff 2007):
    ref_price   = median dollar_price of same-CUSIP same-day inter-dealer (D) trades
    markup      = |customer_price - ref_price| / ref_price
    inst_markup = markup < MARKUP_THRESH  (default: 10 bps)
    Only defined for customer trades with at least one same-day D trade.

  Combined:
    inst_combined = inst_channel | inst_sma | inst_markup
    (markup component requires a valid ref_price)

Data availability notes
-------------------------------------------------
- 2005–2010: All indicator fields (ATS, NTBC, BB) are entirely null — MSRB did
  not collect them in early years. inst_channel = 0 for these years by design.
- BB (brokers_broker_indicator) first appears in ~2012.
- ATS and NTBC first appear in ~2016 when MSRB mandated their reporting.
- 2005–2011 data contains only customer (P/S) trades — no inter-dealer (D) trades
  are present in the WRDS export. This means ref_price is undefined and
  inst_markup = 0 for those years.
- inst_sma (SMA clustering) is the only signal available across the full 2005–2025
  window and is the primary vehicle for tracking institutionalization trends over time.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm
from utils.set_paths import PROC_DIR, OUT_DIR

MARKUP_THRESH  = 0.0010   # 10 basis points
SMA_MIN_TRADES = 3        # min same-direction customer trades on same CUSIP-date → SMA cluster

logfile = os.path.join(PROC_DIR, "MSRB", "institutionalization_log.txt")
with open(logfile, "w") as f:
    pass

def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    with open(logfile, "a") as f:
        f.write(line + "\n")


""" Compute institutionalization """
USECOLS = [
    "trade_date", "cusip", "trade_type_indicator",
    "par_traded", "dollar_price", "time_of_trade",
    "ats_indicator", "brokers_broker_indicator",
    "ntbc_indicator", "weighted_price_indicator",
    "offer_price_takedown_indicator",
]

years = range(2005, 2026)
results = []

log(f"START institutionalization loop. Years: {min(years)}-{max(years)}")

for y in tqdm(years):
    f = os.path.join(PROC_DIR, "MSRB", f"msrb_clean_{y}.csv")
    log("-----")
    log(f"Year {y} - loading: {f}")

    if not os.path.exists(f):
        log(f"WARNING: file not found, skipping year {y}")
        results.append({"year": y})
        continue

    dt = pd.read_csv(f, usecols=USECOLS, low_memory=False)
    n_before = len(dt)

    # Drop primary takedowns
    if "offer_price_takedown_indicator" in dt.columns:
        dt = dt[dt["offer_price_takedown_indicator"] != "Y"]
        log("Dropped primary takedown trades using offer_price_takedown_indicator.")
    else:
        log("NOTE: offer_price_takedown_indicator not found; no primary-market filter applied.")

    n_after  = len(dt)
    n_drop   = n_before - n_after
    pct_drop = 100 * n_drop / n_before if n_before > 0 else float("nan")
    log(f"Rows before: {n_before} | after: {n_after} | dropped: {n_drop} | pct_dropped: {pct_drop:.2f}%")

    dt["dollar_price"] = pd.to_numeric(dt["dollar_price"], errors="coerce")
    dt["par_traded"]   = pd.to_numeric(dt["par_traded"],   errors="coerce")

    " Trade-type flags "
    dt["odd_lot"]        = dt["par_traded"] <= 100_000
    dt["customer_trade"] = dt["trade_type_indicator"].isin(["P", "S"])
    dt["dealer_trade"]   = dt["trade_type_indicator"] == "D"

    " Dealer flag signal "
    # NOTE: ats_indicator and brokers_broker_indicator are inter-dealer (D)
    # only fields — they are always null on customer trades (P/S) and
    # therefore contribute nothing to inst_channel for customer odd-lots.
    dt["inst_ats"]  = dt["ats_indicator"].isin(["Y"])
    dt["inst_bb"]   = dt["brokers_broker_indicator"].isin(["P", "S"])
    dt["inst_ntbc"] = dt["ntbc_indicator"].isin(["Y"])
    dt["inst_wap"]  = dt["weighted_price_indicator"].isin(["Y"])

    # For customer trades, the effective flag-based channel is ntbc | wap only.
    dt["inst_channel"] = dt[["inst_ats", "inst_bb", "inst_ntbc", "inst_wap"]].any(axis=1)

    " Clustered trading signal "
    # SMAs rebalance across many client accounts → multiple small same-direction
    # customer trades on the same CUSIP within a 1-hour window. The time window
    # tightens the signal: retail investors almost never generate ≥ SMA_MIN_TRADES
    # same-direction trades on the same bond within the same hour.
    # time_of_trade is in seconds since midnight; floor to hour bucket.
    dt["trade_hour"] = pd.to_numeric(dt["time_of_trade"], errors="coerce") // 3600
    cust_mask = dt["customer_trade"]
    dt["inst_sma"] = False
    if cust_mask.sum() > 0:
        cluster_ct = (
            dt.loc[cust_mask]
            .groupby(["cusip", "trade_date", "trade_type_indicator", "trade_hour"])["cusip"]
            .transform("count")
        )
        dt.loc[cust_mask, "inst_sma"] = cluster_ct >= SMA_MIN_TRADES

    " Markup-based institutional signal "
    # Reference price: median dealer-to-dealer price per CUSIP-date
    ref = (
        dt[dt["dealer_trade"] & dt["dollar_price"].notna()]
        .groupby(["cusip", "trade_date"], as_index=False)["dollar_price"]
        .median()
        .rename(columns={"dollar_price": "ref_price"})
    )
    n_ref_cusip_days = len(ref)

    dt = pd.merge(dt, ref, on=["cusip", "trade_date"], how="left")

    # Markup only defined for customer trades with a valid reference price
    has_ref = dt["customer_trade"] & dt["ref_price"].notna() & dt["dollar_price"].notna()
    dt["markup"] = np.where(
        has_ref,
        (dt["dollar_price"] - dt["ref_price"]).abs() / dt["ref_price"],
        np.nan
    )
    dt["inst_markup"] = dt["markup"] < MARKUP_THRESH   # NaN < threshold → False

    # Combined: flag-based OR SMA-cluster OR markup-based (customer trades)
    dt["inst_combined"] = dt["inst_channel"] | dt["inst_sma"] | dt["inst_markup"]

    pct_with_ref = 100 * has_ref.sum() / dt["customer_trade"].sum() if dt["customer_trade"].sum() > 0 else float("nan")
    log(f"Customer trades with a ref price: {has_ref.sum():,} / {dt['customer_trade'].sum():,} ({pct_with_ref:.1f}%)")

    # Year-level summary
    n_trades  = len(dt)
    par_total = dt["par_traded"].sum()

    odd      = dt[dt["odd_lot"]]
    cust_odd = dt[dt["odd_lot"] & dt["customer_trade"]]

    n_oddlot        = len(odd)
    par_oddlot      = odd["par_traded"].sum()
    n_cust_oddlot   = len(cust_odd)
    par_cust_oddlot = cust_odd["par_traded"].sum()

    def _count_share(sub, col):
        return sub[col].mean() if len(sub) > 0 else float("nan")

    def _vol_share(sub, col):
        denom = sub["par_traded"].sum()
        return (sub["par_traded"] * sub[col]).sum() / denom if denom > 0 else float("nan")

    # flag-based (all odd-lots)
    inst_all_count  = _count_share(odd,      "inst_channel")
    inst_all_vol    = _vol_share(odd,         "inst_channel")

    # flag-based (customer odd-lots)
    inst_cust_count = _count_share(cust_odd, "inst_channel")
    inst_cust_vol   = _vol_share(cust_odd,   "inst_channel")

    # SMA-cluster (customer odd-lots)
    inst_sma_count = _count_share(cust_odd,  "inst_sma")
    inst_sma_vol   = _vol_share(cust_odd,    "inst_sma")

    # markup-based (customer odd-lots with ref price)
    cust_odd_ref = cust_odd[cust_odd["ref_price"].notna()]
    inst_markup_count = _count_share(cust_odd_ref, "inst_markup")
    inst_markup_vol   = _vol_share(cust_odd_ref,   "inst_markup")

    # combined (all customer odd-lots; inst_markup is already False where ref_price is missing,
    # so restricting denominator to cust_odd_ref would bias toward liquid CUSIPs)
    inst_combined_count = _count_share(cust_odd,   "inst_combined")
    inst_combined_vol   = _vol_share(cust_odd,     "inst_combined")

    # Ref-price coverage (customer odd-lots)
    ref_cov = len(cust_odd_ref) / len(cust_odd) if len(cust_odd) > 0 else float("nan")

    # Signal overlap decomposition (customer odd-lots)
    f  = cust_odd["inst_channel"]
    s  = cust_odd["inst_sma"]
    m  = cust_odd["inst_markup"]   # False where ref_price missing
    n_co = len(cust_odd)
    overlap = {
        "n_only_flag":    int((f  & ~s & ~m).sum()),
        "n_only_sma":     int((~f &  s & ~m).sum()),
        "n_only_markup":  int((~f & ~s &  m).sum()),
        "n_flag_sma":     int((f  &  s & ~m).sum()),
        "n_flag_markup":  int((f  & ~s &  m).sum()),
        "n_sma_markup":   int((~f &  s &  m).sum()),
        "n_all_three":    int((f  &  s &  m).sum()),
        "n_combined":     int((f  |  s |  m).sum()),
        "n_none":         int((~f & ~s & ~m).sum()),
    }
    log(
        f"  Ref-price coverage: {ref_cov*100:.1f}% | "
        f"Combined inst: {overlap['n_combined']/n_co*100:.1f}% | "
        f"Only-flag: {overlap['n_only_flag']/n_co*100:.1f}% | "
        f"Only-SMA: {overlap['n_only_sma']/n_co*100:.1f}% | "
        f"Only-markup: {overlap['n_only_markup']/n_co*100:.1f}% | "
        f"None: {overlap['n_none']/n_co*100:.1f}%"
    )

    # P vs S split (customer odd-lots)
    cust_odd_P = dt[dt["odd_lot"] & (dt["trade_type_indicator"] == "P")]
    cust_odd_S = dt[dt["odd_lot"] & (dt["trade_type_indicator"] == "S")]
    inst_sma_P_count = _count_share(cust_odd_P, "inst_sma")
    inst_sma_S_count = _count_share(cust_odd_S, "inst_sma")

    # Size-bucket SMA (customer odd-lots)
    SIZE_BUCKETS = [
        ("0_15k",   0,       15_000),
        ("15_25k",  15_000,  25_000),
        ("25_50k",  25_000,  50_000),
        ("50_100k", 50_000, 100_000),
    ]
    stats_by_bucket = {}
    for label, lo, hi in SIZE_BUCKETS:
        bkt     = cust_odd[(cust_odd["par_traded"] > lo) & (cust_odd["par_traded"] <= hi)]
        bkt_ref = bkt[bkt["ref_price"].notna()]
        stats_by_bucket[f"inst_sma_{label}"]      = _count_share(bkt,     "inst_sma")
        stats_by_bucket[f"inst_channel_{label}"]  = _count_share(bkt,     "inst_channel")
        stats_by_bucket[f"inst_markup_{label}"]   = _count_share(bkt_ref, "inst_markup")
        stats_by_bucket[f"inst_combined_{label}"] = _count_share(bkt,     "inst_combined")

    # Block-trade combined signal calibration: ≥$1M customer trades should be
    # nearly all institutional. Combined should outperform SMA alone here since
    # large CUSIPs are more liquid (higher ref_price coverage) and may carry NTBC/WAP flags.
    block = dt[dt["customer_trade"] & (dt["par_traded"] >= 1_000_000)]
    block_ref = block[block["ref_price"].notna()]
    inst_sma_block_count      = _count_share(block,     "inst_sma")
    inst_channel_block_count  = _count_share(block,     "inst_channel")
    inst_markup_block_count   = _count_share(block_ref, "inst_markup")
    inst_combined_block_count = _count_share(block,     "inst_combined")
    ref_cov_block = len(block_ref) / len(block) if len(block) > 0 else float("nan")
    n_block = len(block)

    results.append({
        "year":                  y,
        "n_trades":              n_trades,
        "n_oddlot":              n_oddlot,
        "n_cust_oddlot":         n_cust_oddlot,
        "par_total_bn":          par_total       / 1e9,
        "par_oddlot_bn":         par_oddlot      / 1e9,
        "par_cust_oddlot_bn":    par_cust_oddlot / 1e9,
        # flag-based
        "inst_all_count":        inst_all_count,
        "inst_all_vol":          inst_all_vol,
        "inst_cust_count":       inst_cust_count,
        "inst_cust_vol":         inst_cust_vol,
        # SMA clustering
        "inst_sma_count":        inst_sma_count,
        "inst_sma_vol":          inst_sma_vol,
        # markup-based
        "inst_markup_count":     inst_markup_count,
        "inst_markup_vol":       inst_markup_vol,
        # combined: flag | sma | markup
        "inst_combined_count":   inst_combined_count,
        "inst_combined_vol":     inst_combined_vol,

        # ref-price coverage and signal overlap
        "ref_price_cov":         ref_cov,
        **{k: v for k, v in overlap.items()},
        # P vs S SMA split
        "inst_sma_P_count":      inst_sma_P_count,
        "inst_sma_S_count":      inst_sma_S_count,
        # size-bucket signals (sma, channel, markup, combined)
        **stats_by_bucket,
        # block-trade calibration (≥$1M customer trades)
        "inst_sma_block_count":      inst_sma_block_count,
        "inst_channel_block_count":  inst_channel_block_count,
        "inst_markup_block_count":   inst_markup_block_count,
        "inst_combined_block_count": inst_combined_block_count,
        "ref_cov_block":             ref_cov_block,
        "n_block":                   n_block,
    })

    del dt
    log(f"Year {y} done.")

summary_year = pd.DataFrame(results)
summary_year.to_csv(os.path.join(PROC_DIR, "MSRB/oddlot_sum_year.csv"), index=False)
log("Saved summary_year.")


" Sanity checks "
valid = summary_year[summary_year["n_cust_oddlot"] > 0].copy()
diag_cols = [
    "year", "n_cust_oddlot",
    "ref_price_cov",
    "inst_cust_count", "inst_sma_count", "inst_markup_count", "inst_combined_count",
    "n_only_flag", "n_only_sma", "n_only_markup",
    "n_flag_sma", "n_flag_markup", "n_sma_markup", "n_all_three",
    "n_combined", "n_none",
]
diag = valid[[c for c in diag_cols if c in valid.columns]].copy()

# Format share columns as percentages for readability
for col in ["ref_price_cov", "inst_cust_count", "inst_sma_count",
            "inst_markup_count", "inst_combined_count"]:
    if col in diag.columns:
        diag[col] = (diag[col] * 100).round(1)

print("\n=== Sanity Check: Combined Institutional Signal ===")
print(diag.to_string(index=False))

# Highlight most recent year with all signals active
recent = valid[valid["year"] == valid["year"].max()].iloc[0]
n = recent["n_cust_oddlot"]
print(f"\n--- Most recent year: {int(recent['year'])} ---")
print(f"  Customer odd-lots:       {int(n):,}")
print(f"  Ref-price coverage:      {recent['ref_price_cov']*100:.1f}%")
print(f"  Flag-based rate:         {recent['inst_cust_count']*100:.1f}%")
print(f"  SMA cluster rate:        {recent['inst_sma_count']*100:.1f}%")
print(f"  Markup-based rate:       {recent['inst_markup_count']*100:.1f}%  (denominator: ref-price trades only)")
print(f"  Combined rate:           {recent['inst_combined_count']*100:.1f}%")
print(f"  MSRB 2025 benchmark:     53–67%  (2024)")
print(f"  Signal overlap:")
for k in ["n_only_flag","n_only_sma","n_only_markup",
          "n_flag_sma","n_flag_markup","n_sma_markup","n_all_three","n_none"]:
    if k in recent:
        print(f"    {k:20s}: {int(recent[k]):>8,}  ({int(recent[k])/n*100:.1f}%)")


""" Plots """
summary_year = pd.read_csv(os.path.join(PROC_DIR, "MSRB/oddlot_sum_year.csv"))
summary_year["year"] = summary_year["year"].astype(int)
summary_year["share_par_oddlot"]   = summary_year["par_oddlot_bn"]  / summary_year["par_total_bn"]
summary_year["share_count_oddlot"] = summary_year["n_oddlot"]       / summary_year["n_trades"]

# MSRB (2025) benchmark for 2024: institutional share of odd-lot customer trades
# ranges from 53% (assuming 15% of mixed-dealer trades are institutional) to
# 67% (assuming 40%), with a midpoint of ~60%.  Shown as a shaded interval at x=2024.
MSRB_LO, MSRB_HI = 53, 67

def add_msrb_benchmark(ax):
    """Shade the MSRB (2025) estimated institutional odd-lot range at x=2024."""
    ax.fill_betweenx(
        y=[MSRB_LO, MSRB_HI], x1=2023.6, x2=2024.4,
        color="grey", alpha=0.30,
        label=f"MSRB 2024 benchmark"
    )


" Plot 1: Odd-lot share (dual axis) "
k = summary_year["share_count_oddlot"].max() / summary_year["share_par_oddlot"].max()

fig, ax1 = plt.subplots(figsize=(10, 4), facecolor="white")
ax1.set_facecolor("white")
ax2 = ax1.twinx()

ax1.plot(summary_year["year"], summary_year["share_par_oddlot"] * 100,
         color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Par (volume)")
ax2.plot(summary_year["year"], summary_year["share_count_oddlot"] * 100,
         color="#d62728", linewidth=1.5, marker="o", markersize=4, label="Count (trades)")

ax1.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax1.set_xlabel("Year")
ax1.set_ylabel("Par (volume) share (%)", fontsize=11)
ax2.set_ylabel("Count (trades) share (%)", fontsize=11)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.grid(True, color="lightgrey")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
#plt.suptitle("Odd-Lot Share of Municipal Bond Trades")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/oddlot_share_ts.pdf"), bbox_inches="tight")
plt.show()


" Plot 2: Institutionalization — count share, four methods" 
fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(summary_year["year"], summary_year["inst_cust_count"] * 100,
        color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Dealer flag")
ax.plot(summary_year["year"], summary_year["inst_sma_count"] * 100,
        color="#9467bd", linewidth=1.5, marker="o", markersize=4, label=f"Clustered trading")
ax.plot(summary_year["year"], summary_year["inst_markup_count"] * 100,
        color="#2ca02c", linewidth=1.5, marker="o", markersize=4, label=f"Markup-based")
ax.plot(summary_year["year"], summary_year["inst_combined_count"] * 100,
        color="#d62728", linewidth=1.5, marker="o", markersize=4, label="Combined")
add_msrb_benchmark(ax)
ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax.set_xlabel("Year")
ax.set_ylabel("Institutional share (%)", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
#plt.suptitle("Institutionalization of Odd-Lot Customer Trades (Count)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/inst_trade_count_ts.pdf"), bbox_inches="tight")
plt.show()


" Plot 3: Institutionalization — volume share, four methods "
fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(summary_year["year"], summary_year["inst_cust_vol"] * 100,
        color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Dearler flag")
ax.plot(summary_year["year"], summary_year["inst_sma_vol"] * 100,
        color="#9467bd", linewidth=1.5, marker="o", markersize=4, label=f"Clustered trading")
ax.plot(summary_year["year"], summary_year["inst_markup_vol"] * 100,
        color="#2ca02c", linewidth=1.5, marker="o", markersize=4, label=f"Markup-based")
ax.plot(summary_year["year"], summary_year["inst_combined_vol"] * 100,
        color="#d62728", linewidth=1.5, marker="o", markersize=4, label="Combined")
add_msrb_benchmark(ax)
ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax.set_xlabel("Year")
ax.set_ylabel("Institutional share (%)", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
#plt.suptitle("Institutionalization of Odd-Lot Customer Trades (Par Volume)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/inst_trade_vol_ts.pdf"), bbox_inches="tight")
plt.show()


""" Robustness checks """
sy = summary_year.copy()
n  = sy["n_cust_oddlot"]

# Per-trade shares for each exclusive component
sy["sh_only_flag"]   = sy["n_only_flag"]   / n
sy["sh_only_sma"]    = sy["n_only_sma"]    / n
sy["sh_only_markup"] = sy["n_only_markup"] / n
sy["sh_overlap"]     = (sy["n_flag_sma"] + sy["n_flag_markup"] +
                        sy["n_sma_markup"] + sy["n_all_three"]) / n
sy["sh_none"]        = sy["n_none"] / n

" Robustness 1: Stacked area — exclusive contribution of each signal over time "
fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.stackplot(
    sy["year"],
    sy["sh_only_flag"]   * 100,
    sy["sh_only_sma"]    * 100,
    sy["sh_only_markup"] * 100,
    sy["sh_overlap"]     * 100,
    labels=["Flag only (ntbc|wap)", "SMA only", "Markup only", "2+ signals overlap"],
    colors=["#1f77b4", "#9467bd", "#2ca02c", "#bcbd22"],
    alpha=0.80,
)
ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax.set_xlabel("Year")
ax.set_ylabel("Share of customer odd-lots (%)", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey", zorder=0)
ax.legend(fontsize=10, loc="upper left")
plt.suptitle("Signal Decomposition: Exclusive Contribution to Combined Institutional Measure")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_decomp.pdf"), bbox_inches="tight")
plt.show()


" Robustness 2: Ref-price coverage vs markup institutional rate "
# If markup rate tracks ref-price coverage, the markup signal is driven by data
# availability (which CUSIPs have D-trades) rather than economics.
fig, ax1 = plt.subplots(figsize=(10, 4), facecolor="white")
ax1.set_facecolor("white")
ax2 = ax1.twinx()
ax1.plot(sy["year"], sy["ref_price_cov"] * 100,
         color="#7f7f7f", linewidth=1.5, marker="o", markersize=4, label="Ref-price coverage (left)")
ax2.plot(sy["year"], sy["inst_markup_count"] * 100,
         color="#2ca02c", linewidth=1.5, marker="s", markersize=4, label="Markup inst rate (right)")
ax1.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax1.set_xlabel("Year")
ax1.set_ylabel("Customer odd-lots with D-trade ref price (%)", fontsize=11)
ax2.set_ylabel("Markup institutional rate (%)", fontsize=11)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.grid(True, color="lightgrey")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
plt.suptitle("Robustness: Markup Signal vs Ref-Price Availability")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_markup_cov.pdf"), bbox_inches="tight")
plt.show()


" Robustness 3: Flag-based structural break check (2016 mandate) "
# MSRB mandated NTBC/ATS reporting ~2016. The flag-based signal should show a
# structural upward shift at 2016 if it is real institutional activity, not just
# indicator adoption. If it jumps discontinuously, the level pre-2016 is not comparable.
fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(sy["year"], sy["inst_cust_count"] * 100,
        color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Flag-based (ntbc|wap)")
ax.axvline(x=2016, color="#d62728", linewidth=1.2, linestyle="--", label="NTBC/ATS mandate (~2016)")
ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax.set_xlabel("Year")
ax.set_ylabel("Flag-based institutional share (%)", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
plt.suptitle("Robustness: Flag-Based Signal Around NTBC/ATS Mandate (2016)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_flag_break.pdf"), bbox_inches="tight")
plt.show()


" Robustness 4: Combined minus best single signal (marginal gain from combining) "
# If combined is barely above max(flag, sma, markup), combining adds little.
# Large gap means the signals are complementary and union meaningfully expands coverage.
sy["best_single"] = sy[["inst_cust_count", "inst_sma_count", "inst_markup_count"]].max(axis=1)
sy["marginal_gain"] = sy["inst_combined_count"] - sy["best_single"]

fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(sy["year"], sy["inst_combined_count"] * 100,
        color="#d62728", linewidth=1.5, marker="o", markersize=4, label="Combined")
ax.plot(sy["year"], sy["best_single"] * 100,
        color="#7f7f7f", linewidth=1.5, marker="o", markersize=4,
        linestyle="--", label="Best single signal")
ax.fill_between(sy["year"],
                sy["best_single"] * 100, sy["inst_combined_count"] * 100,
                color="#d62728", alpha=0.15, label="Marginal gain from combining")
ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax.set_xlabel("Year")
ax.set_ylabel("Institutional share (%)", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
plt.suptitle("Robustness: Marginal Gain of Combined Signal Over Best Single Signal")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_marginal_gain.pdf"), bbox_inches="tight")
plt.show()


" Robustness 5: Combined signal applied to block trades (≥$1M) "
# Block customer trades are almost certainly institutional. If the combined signal
# is working, it should flag a large fraction of block trades — much more than
# the ~2-3% it flags for odd-lots. The SMA component alone won't fire here (blocks
# rarely cluster), so this checks whether inst_channel + inst_markup carry the load.
# A high block rate (e.g. >50%) would validate the combined signal's institutional reach.
bl = sy[sy["n_block"] > 0].copy()

fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(bl["year"], bl["inst_combined_block_count"] * 100,
        color="#d62728", linewidth=1.5, marker="o", markersize=4, label="Combined (block ≥$1M)")
ax.plot(bl["year"], bl["inst_channel_block_count"] * 100,
        color="#1f77b4", linewidth=1.5, marker="o", markersize=4,
        linestyle="--", label="Flag only (block ≥$1M)")
ax.plot(bl["year"], bl["inst_markup_block_count"] * 100,
        color="#2ca02c", linewidth=1.5, marker="o", markersize=4,
        linestyle=":", label="Markup only (block ≥$1M, conditional on ref price)")
ax.plot(sy["year"], sy["inst_combined_count"] * 100,
        color="#7f7f7f", linewidth=1.2, marker="s", markersize=3,
        linestyle="--", alpha=0.6, label="Combined (odd-lots ≤$100K) — reference")
ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax.set_xlabel("Year")
ax.set_ylabel("Institutional share (%)", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
plt.suptitle("Robustness: Combined Signal on Block Trades (≥$1M) vs Odd-Lots")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_block_combined.pdf"), bbox_inches="tight")
plt.show()


" Robustness 6: Combined signal by par-size bucket over time "
# Each size bucket should show combined >> sma-only (larger buckets have better
# ref-price coverage and may carry ntbc/wap flags). If combined is flat across
# buckets, the markup/flag signals don't discriminate by size either.
BUCKET_META = [
    ("0_15k",   "$0–15K",   "#1f77b4"),
    ("15_25k",  "$15–25K",  "#ff7f0e"),
    ("25_50k",  "$25–50K",  "#2ca02c"),
    ("50_100k", "$50–100K", "#d62728"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="white")
for ax in axes:
    ax.set_facecolor("white")

# Left: combined rate by bucket over time
for label, display, color in BUCKET_META:
    col = f"inst_combined_{label}"
    if col in sy.columns:
        axes[0].plot(sy["year"], sy[col] * 100,
                     color=color, linewidth=1.5, marker="o", markersize=4, label=display)
axes[0].xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Combined institutional share (%)", fontsize=11)
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0].grid(True, color="lightgrey")
axes[0].legend(fontsize=10, title="Par-size bucket")
axes[0].set_title("Combined rate by size bucket (time series)")

# Right: SMA vs combined bar chart for most recent year
recent_yr = sy["year"].max()
row = sy[sy["year"] == recent_yr].iloc[0]
bucket_labels  = [d for _, d, _ in BUCKET_META]
bucket_colors  = [c for _, _, c in BUCKET_META]
sma_rates      = [row.get(f"inst_sma_{l}",      float("nan")) * 100 for l, _, _ in BUCKET_META]
combined_rates = [row.get(f"inst_combined_{l}",  float("nan")) * 100 for l, _, _ in BUCKET_META]

x = np.arange(len(BUCKET_META))
w = 0.35
axes[1].bar(x - w/2, sma_rates,      w, label="SMA only",  color="#9467bd", alpha=0.80)
axes[1].bar(x + w/2, combined_rates, w, label="Combined",  color="#d62728", alpha=0.80)
axes[1].set_xticks(x)
axes[1].set_xticklabels(bucket_labels)
axes[1].set_xlabel("Par-size bucket")
axes[1].set_ylabel("Institutional share (%)", fontsize=11)
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].grid(True, axis="y", color="lightgrey")
axes[1].legend(fontsize=10)
axes[1].set_title(f"SMA vs Combined by size bucket ({recent_yr})")

plt.suptitle("Robustness: Combined Signal by Par-Size Bucket")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_bucket_combined.pdf"), bbox_inches="tight")
plt.show()
