"""
inst_trade.py
Compute institutionalization of municipal bond trades from MSRB clean files.

- Odd lot:        par_traded <= 100,000
- Customer trade: trade_type_indicator in {"P", "S"}

Institutional signals
---------------------
  Flag-based (lower bound):
    ATS:   ats_indicator == "Y"
    BB:    brokers_broker_indicator in {"P","S"}
    NTBC:  ntbc_indicator == "Y"
    WAP:   weighted_price_indicator == "Y"
    inst_channel = any of the above

  Markup-based (Harris & Piwowar 2006; Green, Hollifield & Schürhoff 2007):
    ref_price  = median dollar_price of same-CUSIP same-day inter-dealer (D) trades
    markup     = |customer_price - ref_price| / ref_price
    inst_markup = markup < MARKUP_THRESH  (25 bps)
    Only defined for customer trades with at least one same-day D trade.

  Combined:
    inst_combined = inst_channel | inst_markup  (for customer trades with a ref price)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm
from utils.set_paths import PROC_DIR, OUT_DIR

MARKUP_THRESH = 0.0025   # 25 basis points

logfile = os.path.join(PROC_DIR, "MSRB", "institutionalization_log.txt")
with open(logfile, "w") as f:
    pass

def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    with open(logfile, "a") as f:
        f.write(line + "\n")


""" Compute institutionalization """
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

    dt = pd.read_csv(f)
    n_before = len(dt)

    # ── Drop primary takedowns ────────────────────────────────────────────────
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

    # ── Trade-type flags ──────────────────────────────────────────────────────
    dt["odd_lot"]        = dt["par_traded"] <= 100_000
    dt["customer_trade"] = dt["trade_type_indicator"].isin(["P", "S"])
    dt["dealer_trade"]   = dt["trade_type_indicator"] == "D"

    # ── Flag-based institutional signals ─────────────────────────────────────
    dt["inst_ats"]     = dt["ats_indicator"].isin(["Y"])
    dt["inst_bb"]      = dt["brokers_broker_indicator"].isin(["P", "S"])
    dt["inst_ntbc"]    = dt["ntbc_indicator"].isin(["Y"])
    dt["inst_wap"]     = dt["weighted_price_indicator"].isin(["Y"])
    dt["inst_channel"] = dt[["inst_ats", "inst_bb", "inst_ntbc", "inst_wap"]].any(axis=1)

    # ── Markup-based institutional signal ────────────────────────────────────
    # Reference price: median dealer-to-dealer price per CUSIP-date
    ref = (
        dt[dt["dealer_trade"] & dt["dollar_price"].notna()]
        .groupby(["cusip", "trade_date"], as_index=False)["dollar_price"]
        .median()
        .rename(columns={"dollar_price": "ref_price"})
    )
    n_ref_cusip_days = len(ref)

    dt = dt.merge(ref, on=["cusip", "trade_date"], how="left")

    # Markup only defined for customer trades with a valid reference price
    has_ref = dt["customer_trade"] & dt["ref_price"].notna() & dt["dollar_price"].notna()
    dt["markup"] = np.where(
        has_ref,
        (dt["dollar_price"] - dt["ref_price"]).abs() / dt["ref_price"],
        np.nan
    )
    dt["inst_markup"] = dt["markup"] < MARKUP_THRESH   # NaN < threshold → False

    # Combined: flag-based OR markup-based (for customer trades only)
    dt["inst_combined"] = dt["inst_channel"] | dt["inst_markup"]

    pct_with_ref = 100 * has_ref.sum() / dt["customer_trade"].sum() if dt["customer_trade"].sum() > 0 else float("nan")
    log(f"Customer trades with a ref price: {has_ref.sum():,} / {dt['customer_trade'].sum():,} ({pct_with_ref:.1f}%)")

    # ── Year-level summary ────────────────────────────────────────────────────
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

    # markup-based (customer odd-lots with ref price)
    cust_odd_ref = cust_odd[cust_odd["ref_price"].notna()]
    inst_markup_count = _count_share(cust_odd_ref, "inst_markup")
    inst_markup_vol   = _vol_share(cust_odd_ref,   "inst_markup")

    # combined (customer odd-lots with ref price)
    inst_combined_count = _count_share(cust_odd_ref, "inst_combined")
    inst_combined_vol   = _vol_share(cust_odd_ref,   "inst_combined")

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
        # markup-based (customer odd-lots with ref price)
        "inst_markup_count":     inst_markup_count,
        "inst_markup_vol":       inst_markup_vol,
        # combined
        "inst_combined_count":   inst_combined_count,
        "inst_combined_vol":     inst_combined_vol,
    })

    del dt
    log(f"Year {y} done.")

summary_year = pd.DataFrame(results)
summary_year.to_csv(os.path.join(PROC_DIR, "MSRB/oddlot_sum_year.csv"), index=False)
log("Saved summary_year.")


# ================================================================= #
# Plots                                                              #
# ================================================================= #
summary_year = pd.read_csv(os.path.join(PROC_DIR, "MSRB/oddlot_sum_year.csv"))
summary_year["share_par_oddlot"]   = summary_year["par_oddlot_bn"]  / summary_year["par_total_bn"]
summary_year["share_count_oddlot"] = summary_year["n_oddlot"]       / summary_year["n_trades"]

# ── Plot 1: Odd-lot share (dual axis) ────────────────────────────────────────
k = summary_year["share_count_oddlot"].max() / summary_year["share_par_oddlot"].max()

fig, ax1 = plt.subplots(figsize=(10, 4), facecolor="white")
ax1.set_facecolor("white")
ax2 = ax1.twinx()

ax1.plot(summary_year["year"], summary_year["share_par_oddlot"] * 100,
         color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Par (volume)")
ax2.plot(summary_year["year"], summary_year["share_count_oddlot"] * 100,
         color="#ff7f0e", linewidth=1.5, marker="o", markersize=4, label="Count (trades)")

ax1.set_xlabel("Year")
ax1.set_ylabel("Par (volume) share (%)", fontsize=11)
ax2.set_ylabel("Count (trades) share (%)", fontsize=11)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.grid(True, color="lightgrey")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
plt.suptitle("Odd-Lot Share of Municipal Bond Trades")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "institutionalization/oddlot_share_ts.pdf"), bbox_inches="tight")
plt.show()

# ── Plot 2: Institutionalization — count share, three methods ─────────────────
fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(summary_year["year"], summary_year["inst_cust_count"] * 100,
        color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Flag-based (lower bound)")
ax.plot(summary_year["year"], summary_year["inst_markup_count"] * 100,
        color="#ff7f0e", linewidth=1.5, marker="o", markersize=4, label=f"Markup-based (<{int(MARKUP_THRESH*10000)} bps)")
ax.plot(summary_year["year"], summary_year["inst_combined_count"] * 100,
        color="#2ca02c", linewidth=1.5, marker="o", markersize=4, label="Combined (flag OR markup)")
ax.set_xlabel("Year")
ax.set_ylabel("Institutionalization share (%)", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
plt.suptitle("Institutionalization of Odd-Lot Customer Trades (Count)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "institutionalization/inst_trade_count_ts.pdf"), bbox_inches="tight")
plt.show()

# ── Plot 3: Institutionalization — volume share, three methods ────────────────
fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(summary_year["year"], summary_year["inst_cust_vol"] * 100,
        color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Flag-based (lower bound)")
ax.plot(summary_year["year"], summary_year["inst_markup_vol"] * 100,
        color="#ff7f0e", linewidth=1.5, marker="o", markersize=4, label=f"Markup-based (<{int(MARKUP_THRESH*10000)} bps)")
ax.plot(summary_year["year"], summary_year["inst_combined_vol"] * 100,
        color="#2ca02c", linewidth=1.5, marker="o", markersize=4, label="Combined (flag OR markup)")
ax.set_xlabel("Year")
ax.set_ylabel("Institutionalization share (%)", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
plt.suptitle("Institutionalization of Odd-Lot Customer Trades (Par Volume)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "institutionalization/inst_trade_vol_ts.pdf"), bbox_inches="tight")
plt.show()
