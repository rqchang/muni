"""
inst_odd_lot.py
Compute institutionalization of municipal bond odd-lot customer trades from MSRB
clean files (WRDS). Produces year-level, state-level, and issuer-level output files.

Definitions
-----------
- Odd lot:        par_traded <= 100,000
- Customer trade: trade_type_indicator in {"P", "S"}
- Dealer trade:   trade_type_indicator == "D"
- Issuer ID:      muni_issuer_id = state abbrev for state-level issuers;
                  state_countyFIPS for county/local issuers (matched via cusip6)

Institutional signals
---------------------
  Flag-based (inst_flag):
    Binary indicators reported by MSRB. ATS and BB are inter-dealer (D) fields
    only — always null on customer trades — so inst_flag for customer odd-lots
    reduces to NTBC | WAP.
    ATS:   ats_indicator == "Y"                   [D trades only]
    BB:    brokers_broker_indicator in {"P","S"}  [D trades only]
    NTBC:  ntbc_indicator == "Y"                  [no markup charged → institutional]
    WAP:   weighted_price_indicator == "Y"        [averaged/systematic pricing → institutional]
    inst_flag = any of the above

  SMA clustering (inst_cluster; Bagley & Vieira, MSRB 2025):
    SMAs rebalance across many client accounts simultaneously, generating clusters
    of same-direction customer trades on the same CUSIP within a short window that
    retail investors almost never produce.
    trade_hour  = floor(time_of_trade_seconds / TRADE_HOUR_GAP)  (3-hour buckets)
    cluster_ct  = count of customer trades sharing (cusip, trade_date,
                  trade_type_indicator, trade_hour)
    inst_cluster = cluster_ct >= SMA_MIN_TRADES  (default: 3)

  Markup-based (inst_markup; Harris & Piwowar 2006; Green, Hollifield & Schürhoff 2007):
    ref_price   = median dollar_price of D trades for the same CUSIP within 1
                  calendar day (date−1, date, date+1 pooled)
    markup      = |customer_price − ref_price| / ref_price
    inst_markup = markup < MARKUP_THRESH  (default: 30 bps)
    Only defined for customer trades with at least one D-trade in the ±1-day window.

  Combined (inst_combined):
    inst_combined = inst_flag | inst_cluster | inst_markup

Output files
------------
  inst_oddlot_year.csv  — year-level aggregate statistics
  inst_oddlot_sy.csv    — year × state institutional share
  inst_oddlot_iy.csv    — year × muni_issuer_id institutional share
  inst_oddlot_sm.csv    — month × state institutional share
  inst_oddlot_im.csv    — month × muni_issuer_id institutional share

Data availability notes
-----------------------
- 2005–2010: ATS, NTBC, BB are entirely null; inst_flag = 0 by design.
- BB first appears ~2012; ATS and NTBC first appear ~2016 (MSRB mandate).
- 2005–2011: no inter-dealer (D) trades in WRDS export → ref_price undefined,
  inst_markup = 0 for those years.
- inst_cluster is the only signal spanning the full 2005–2025 window.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm
from utils.set_paths import PROC_DIR, OUT_DIR, RAW_DIR

MARKUP_THRESH  = 0.0030   # 30 basis points
SMA_MIN_TRADES = 3        # min same-direction customer trades on same CUSIP-date → SMA cluster
TRADE_HOUR_GAP = 3600 * 3 # time bucket width in seconds (3 hours)

logfile = os.path.join(PROC_DIR, "MSRB", "institutionalization_log.txt")
with open(logfile, "w") as f:
    pass

def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    with open(logfile, "a") as f:
        f.write(line + "\n")

# Location: cusip_county_info on cusip6 
cty = pd.read_csv(f"{RAW_DIR}/cusip_county_info.csv", dtype={"cusip6": str})
print("Read muni issuer location:", cty.shape)


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
geo_state_results        = []
geo_issuer_results       = []
geo_state_month_results  = []
geo_issuer_month_results = []

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

    " Drop primary takedowns "
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
    # therefore contribute nothing to inst_flag for customer odd-lots.
    dt["inst_ats"]  = dt["ats_indicator"].isin(["Y"])
    dt["inst_bb"]   = dt["brokers_broker_indicator"].isin(["P", "S"])
    dt["inst_ntbc"] = dt["ntbc_indicator"].isin(["Y"])
    dt["inst_wap"]  = dt["weighted_price_indicator"].isin(["Y"])

    # For customer trades, the effective flag-based channel is ntbc | wap only.
    dt["inst_flag"] = dt[["inst_ats", "inst_bb", "inst_ntbc", "inst_wap"]].any(axis=1)

    " Clustered trading signal "
    # SMAs rebalance across many client accounts → multiple small same-direction
    # customer trades on the same CUSIP within a 1-hour window. The time window
    # tightens the signal: retail investors almost never generate ≥ SMA_MIN_TRADES
    # same-direction trades on the same bond within the same hour.
    # time_of_trade is in seconds since midnight; floor to hour bucket.
    dt["trade_hour"] = pd.to_numeric(dt["time_of_trade"], errors="coerce") // TRADE_HOUR_GAP
    cust_mask = dt["customer_trade"]
    dt["inst_cluster"] = False
    if cust_mask.sum() > 0:
        cluster_ct = (
            dt.loc[cust_mask]
            .groupby(["cusip", "trade_date", "trade_type_indicator", "trade_hour"])["cusip"]
            .transform("count")
        )
        dt.loc[cust_mask, "inst_cluster"] = cluster_ct >= SMA_MIN_TRADES

    " Markup-based institutional signal "
    # Reference price: median dollar_price of D trades for the same CUSIP
    # within 1 calendar day (date-1, date, date+1 pooled).
    # Each D trade is expanded to cover the three customer-trade dates it
    # can serve as a reference for, then median is taken per (cusip, date).
    dt["trade_date"] = pd.to_datetime(dt["trade_date"])
    d_trades = dt.loc[dt["dealer_trade"] & dt["dollar_price"].notna(),
                      ["cusip", "trade_date", "dollar_price"]]
    ref = pd.concat([
        d_trades.assign(trade_date=d_trades["trade_date"] - pd.Timedelta(days=1)),
        d_trades,
        d_trades.assign(trade_date=d_trades["trade_date"] + pd.Timedelta(days=1)),
    ]).groupby(["cusip", "trade_date"], as_index=False)["dollar_price"].median() \
      .rename(columns={"dollar_price": "ref_price"})
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
    dt["inst_combined"] = dt["inst_flag"] | dt["inst_cluster"] | dt["inst_markup"]

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
    inst_all_count  = _count_share(odd, "inst_flag")
    inst_all_vol    = _vol_share(odd, "inst_flag")

    # flag-based (customer odd-lots)
    inst_cust_count = _count_share(cust_odd, "inst_flag")
    inst_cust_vol   = _vol_share(cust_odd, "inst_flag")

    # cluster-based (customer odd-lots)
    inst_cluster_count = _count_share(cust_odd, "inst_cluster")
    inst_cluster_vol   = _vol_share(cust_odd, "inst_cluster")

    # markup-based (all customer odd-lots; inst_markup is False where ref_price is missing,
    # so the rate reflects both signal strength and ref-price coverage)
    cust_odd_ref = cust_odd[cust_odd["ref_price"].notna()]   # kept for ref_cov and diagnostics
    inst_markup_count = _count_share(cust_odd, "inst_markup")
    inst_markup_vol   = _vol_share(cust_odd, "inst_markup")

    # combined: same denominator as all other signals
    inst_combined_count = _count_share(cust_odd, "inst_combined")
    inst_combined_vol   = _vol_share(cust_odd, "inst_combined")

    # Ref-price coverage (customer odd-lots)
    ref_cov = len(cust_odd_ref) / len(cust_odd) if len(cust_odd) > 0 else float("nan")

    # Signal overlap decomposition (customer odd-lots)
    f  = cust_odd["inst_flag"]
    s  = cust_odd["inst_cluster"]
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
        f"Ref-price coverage: {ref_cov*100:.1f}% | "
        f"Combined inst: {overlap['n_combined']/n_co*100:.1f}% | "
        f"Only-flag: {overlap['n_only_flag']/n_co*100:.1f}% | "
        f"Only-SMA: {overlap['n_only_sma']/n_co*100:.1f}% | "
        f"Only-markup: {overlap['n_only_markup']/n_co*100:.1f}% | "
        f"None: {overlap['n_none']/n_co*100:.1f}%"
    )

    # P vs S split (customer odd-lots)
    cust_odd_P = dt[dt["odd_lot"] & (dt["trade_type_indicator"] == "P")]
    cust_odd_S = dt[dt["odd_lot"] & (dt["trade_type_indicator"] == "S")]
    ps_stats = {}
    for sig in ["inst_flag", "inst_cluster", "inst_markup", "inst_combined"]:
        ps_stats[f"{sig}_P_count"] = _count_share(cust_odd_P, sig)
        ps_stats[f"{sig}_S_count"] = _count_share(cust_odd_S, sig)

    # Size-bucket SMA (customer odd-lots)
    SIZE_BUCKETS = [
        ("0_15k",   0,       15_000),
        ("15_25k",  15_000,  25_000),
        ("25_50k",  25_000,  50_000),
        ("50_100k", 50_000, 100_000),
    ]
    stats_by_bucket = {}
    for label, lo, hi in SIZE_BUCKETS:
        bkt = cust_odd[(cust_odd["par_traded"] > lo) & (cust_odd["par_traded"] <= hi)]
        stats_by_bucket[f"inst_cluster_{label}"] = _count_share(bkt, "inst_cluster")
        stats_by_bucket[f"inst_flag_{label}"] = _count_share(bkt, "inst_flag")
        stats_by_bucket[f"inst_markup_{label}"] = _count_share(bkt, "inst_markup")
        stats_by_bucket[f"inst_combined_{label}"] = _count_share(bkt, "inst_combined")

    # Block-trade calibration: ≥$1M trades — customer, dealer, and all
    # Goal: among all large trades, the combined institutional signal should
    # cover a much higher fraction than in odd-lots, validating the measure.
    # Dealer (D) trades are almost entirely institutional by definition;
    # inst_flag (ATS | BB) fires on D trades, so they are a natural upper-bound check.
    BLOCK_SEGMENTS = [
        ("block_cust",   dt["customer_trade"] & (dt["par_traded"] >= 1_000_000)),
        ("block_dealer", dt["dealer_trade"]   & (dt["par_traded"] >= 1_000_000)),
        ("block_all",                            dt["par_traded"] >= 1_000_000),
    ]
    block_stats = {}
    for seg_name, mask in BLOCK_SEGMENTS:
        seg = dt[mask]
        n   = len(seg)
        block_stats[f"n_{seg_name}"] = n
        block_stats[f"ref_cov_{seg_name}"] = (
            seg["ref_price"].notna().sum() / n if n > 0 else float("nan")
        )
        for sig in ["inst_flag", "inst_cluster", "inst_markup", "inst_combined"]:
            block_stats[f"{sig}_{seg_name}_count"] = _count_share(seg, sig)

    # Dealer odd-lot signal rates (robustness: inst_cluster and inst_markup are
    # customer-only by construction → should be ~0 for D-trades; inst_flag
    # should be non-zero since ATS/BB indicators fire on dealer (D) trades)
    dealer_odd = dt[dt["dealer_trade"] & dt["odd_lot"]]
    n_do = len(dealer_odd)
    dealer_odd_stats = {
        "n_dealer_oddlot":      n_do,
        "par_dealer_oddlot_bn": dealer_odd["par_traded"].sum() / 1e9,
    }
    for sig in ["inst_flag", "inst_cluster", "inst_markup", "inst_combined"]:
        dealer_odd_stats[f"{sig}_dealer_odd_count"] = _count_share(dealer_odd, sig)
        dealer_odd_stats[f"{sig}_dealer_odd_vol"]   = _vol_share(dealer_odd, sig)

    results.append({
        "year":               y,
        "n_trades":           n_trades,
        "n_oddlot":           n_oddlot,
        "n_cust_oddlot":      n_cust_oddlot,
        "par_total_bn":       par_total       / 1e9,
        "par_oddlot_bn":      par_oddlot      / 1e9,
        "par_cust_oddlot_bn": par_cust_oddlot / 1e9,
        # all-odd-lots flag baseline
        "inst_all_count":     inst_all_count,
        "inst_all_vol":       inst_all_vol,

        # institutional share of customer odd-lots: 3 signals and combined
        "inst_cust_count":    inst_cust_count,
        "inst_cust_vol":      inst_cust_vol,
        "inst_cluster_count": inst_cluster_count,
        "inst_cluster_vol":   inst_cluster_vol,
        "inst_markup_count":  inst_markup_count,
        "inst_markup_vol":    inst_markup_vol,
        "inst_combined_count":inst_combined_count,
        "inst_combined_vol":  inst_combined_vol,

        # ref-price coverage and signal overlap
        "ref_price_cov":      ref_cov,
        **overlap,
        # P vs S split
        **ps_stats,
        # size-bucket
        **stats_by_bucket,
        # block-trade calibration
        **block_stats,
        # dealer odd-lot signal rates
        **dealer_odd_stats,
    })

    " Geography-level measures "
    # attach issuer state / municipality to MSRB trades
    dt["cusip6"] = dt["cusip"].str[:6]
    dt_geo = pd.merge(dt, cty[["cusip6", "state", "fips_state", "county_fips"]],
                      on="cusip6", how="left")

    # Keep only trades with at least a state match; build muni_issuer_id
    dt_geo = dt_geo.loc[
        dt_geo["county_fips"].notnull() | dt_geo["fips_state"].notnull()
    ].copy()
    is_state = dt_geo["county_fips"].isnull()
    dt_geo.loc[is_state,  "muni_issuer_id"] = dt_geo.loc[is_state, "state"]
    dt_geo.loc[~is_state, "muni_issuer_id"] = (
        dt_geo.loc[~is_state, "state"] + "_" +
        dt_geo.loc[~is_state, "county_fips"].astype(float).astype(int).astype(str)
    )

    # Customer odd-lots with valid geography — base for geo-level inst share
    geo_odd = dt_geo[
        dt_geo["odd_lot"] & dt_geo["customer_trade"] & dt_geo["muni_issuer_id"].notna()
    ].copy()
    geo_odd["par_inst"]    = geo_odd["par_traded"] * geo_odd["inst_combined"]
    geo_odd["n_inst"]      = geo_odd["inst_combined"].astype(int)
    geo_odd["trade_month"] = geo_odd["trade_date"].dt.to_period("M").astype(str)

    def _geo_agg(df, group_cols):
        g = (df.groupby(group_cols)
               .agg(n_cust_oddlot    =("par_traded", "count"),
                    n_inst_combined  =("n_inst",     "sum"),
                    par_cust_oddlot  =("par_traded", "sum"),
                    par_inst_combined=("par_inst",   "sum"))
               .reset_index())
        g["inst_share_vol"]   = g["par_inst_combined"] / g["par_cust_oddlot"]
        g["inst_share_count"] = g["n_inst_combined"]   / g["n_cust_oddlot"]
        return g

    if len(geo_odd) > 0:
        # Annual — state
        st = _geo_agg(geo_odd, ["state", "fips_state"])
        st["year"] = y
        geo_state_results.append(st)

        # Annual — issuer
        iss = _geo_agg(geo_odd, ["muni_issuer_id", "state", "fips_state", "county_fips"])
        iss["year"] = y
        geo_issuer_results.append(iss)

        # Monthly — state
        stm = _geo_agg(geo_odd, ["state", "fips_state", "trade_month"])
        geo_state_month_results.append(stm)

        # Monthly — issuer
        ism = _geo_agg(geo_odd, ["muni_issuer_id", "state", "fips_state", "county_fips", "trade_month"])
        geo_issuer_month_results.append(ism)

    del dt, dt_geo, geo_odd
    log(f"Year {y} done.")

summary_year = pd.DataFrame(results)
summary_year.to_csv(os.path.join(PROC_DIR, "MSRB/inst_oddlot_year.csv"), index=False)
log("Saved summary_year.")

# Geography-level institutional share (combined measure, customer odd-lots, vol-weighted)
if geo_state_results:
    geo_state = pd.concat(geo_state_results, ignore_index=True)
    geo_state = geo_state[["year", "state", "fips_state",
                            "n_cust_oddlot", "n_inst_combined",
                            "par_cust_oddlot", "par_inst_combined",
                            "inst_share_vol", "inst_share_count"]]
    geo_state.to_csv(os.path.join(PROC_DIR, "MSRB/inst_oddlot_sy.csv"), index=False)
    log(f"Saved inst_combined_state: {geo_state.shape}")

if geo_issuer_results:
    geo_issuer = pd.concat(geo_issuer_results, ignore_index=True)
    geo_issuer = geo_issuer[["year", "muni_issuer_id", "state", "fips_state", "county_fips",
                              "n_cust_oddlot", "n_inst_combined",
                              "par_cust_oddlot", "par_inst_combined",
                              "inst_share_vol", "inst_share_count"]]
    geo_issuer.to_csv(os.path.join(PROC_DIR, "MSRB/inst_oddlot_iy.csv"), index=False)
    log(f"Saved inst_combined_issuer: {geo_issuer.shape}")

if geo_state_month_results:
    geo_state_m = pd.concat(geo_state_month_results, ignore_index=True)
    geo_state_m = geo_state_m[["trade_month", "state", "fips_state",
                                "n_cust_oddlot", "n_inst_combined",
                                "par_cust_oddlot", "par_inst_combined",
                                "inst_share_vol", "inst_share_count"]]
    geo_state_m.to_csv(os.path.join(PROC_DIR, "MSRB/inst_oddlot_sm.csv"), index=False)
    log(f"Saved inst_oddlot_sm (monthly state): {geo_state_m.shape}")

if geo_issuer_month_results:
    geo_issuer_m = pd.concat(geo_issuer_month_results, ignore_index=True)
    geo_issuer_m = geo_issuer_m[["trade_month", "muni_issuer_id", "state", "fips_state", "county_fips",
                                  "n_cust_oddlot", "n_inst_combined",
                                  "par_cust_oddlot", "par_inst_combined",
                                  "inst_share_vol", "inst_share_count"]]
    geo_issuer_m.to_csv(os.path.join(PROC_DIR, "MSRB/inst_oddlot_im.csv"), index=False)
    log(f"Saved inst_oddlot_im (monthly issuer): {geo_issuer_m.shape}")


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
        color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Flag-based")
ax.plot(summary_year["year"], summary_year["inst_cluster_count"] * 100,
        color="#9467bd", linewidth=1.5, marker="o", markersize=4, label="Clustering-based")
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
        color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Flag-based")
ax.plot(summary_year["year"], summary_year["inst_cluster_vol"] * 100,
        color="#9467bd", linewidth=1.5, marker="o", markersize=4, label="Clustering-based")
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


" Geography coverage check: state-month geo vs full MSRB sample "
sm = pd.read_csv(os.path.join(PROC_DIR, "MSRB/inst_oddlot_sm.csv"))
yr = pd.read_csv(os.path.join(PROC_DIR, "MSRB/inst_oddlot_year.csv"))

# Aggregate state-month geo data to year
sm["year"] = pd.to_datetime(sm["trade_month"]).dt.year
geo_yr = (sm.groupby("year")
            .agg(geo_n_cust_oddlot =("n_cust_oddlot",  "sum"),
                 geo_par_cust_oddlot=("par_cust_oddlot", "sum"))
            .reset_index())

# Merge with full-sample year totals
cov = pd.merge(yr[["year", "n_cust_oddlot", "par_cust_oddlot_bn"]], geo_yr, on="year", how="left")
cov["par_cust_oddlot_full"] = cov["par_cust_oddlot_bn"] * 1e9
cov["cov_count"] = cov["geo_n_cust_oddlot"]    / cov["n_cust_oddlot"]
cov["cov_vol"]   = cov["geo_par_cust_oddlot"]   / cov["par_cust_oddlot_full"]

print("\nGeography coverage vs full MSRB customer odd-lot sample:")
print(cov[["year", "n_cust_oddlot", "geo_n_cust_oddlot", "cov_count",
           "par_cust_oddlot_bn", "geo_par_cust_oddlot", "cov_vol"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Plot coverage over time
fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor="white")
for ax in axes:
    ax.set_facecolor("white")

axes[0].plot(cov["year"], cov["cov_count"] * 100,
             color="#1f77b4", linewidth=1.5, marker="o", markersize=4)
axes[0].set_ylabel("Coverage (%)", fontsize=11)
axes[0].set_xlabel("Year")
axes[0].set_title("Trade count matched to geography")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0].grid(True, color="lightgrey")

axes[1].plot(cov["year"], cov["cov_vol"] * 100,
             color="#2ca02c", linewidth=1.5, marker="o", markersize=4)
axes[1].set_ylabel("Coverage (%)", fontsize=11)
axes[1].set_xlabel("Year")
axes[1].set_title("Par volume matched to geography")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].grid(True, color="lightgrey")

plt.suptitle("Geography Match Rate: State-Level vs Full MSRB Customer Odd-Lot Sample")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


" State-quarter institutional share time series (2 × 1: vol share, count share) "
# Aggregate monthly file to quarterly by re-summing numerators and denominators
sm["date_q"] = pd.to_datetime(sm["trade_month"]).dt.to_period("Q")
sq = (sm.groupby(["state", "date_q"])
        .agg(par_inst_combined=("par_inst_combined", "sum"),
             par_cust_oddlot  =("par_cust_oddlot",   "sum"),
             n_inst_combined  =("n_inst_combined",    "sum"),
             n_cust_oddlot    =("n_cust_oddlot",      "sum"))
        .reset_index())
sq["inst_share_vol"]   = sq["par_inst_combined"] / sq["par_cust_oddlot"]
sq["inst_share_count"] = sq["n_inst_combined"]   / sq["n_cust_oddlot"]
sq["date"] = sq["date_q"].dt.to_timestamp()
sq = sq[sq["date_q"] > "2007Q4"]

# Top-5 states by total customer odd-lot par volume (post-2007Q4 only)
top5_par   = sq.groupby("state")["par_cust_oddlot"].sum().nlargest(5)
top5_st    = set(top5_par.index)
TOP5_COLORS = ["#fc8d59", "#feb24c", "#78c679", "#c994c7", "#bf812d"]
top5_color  = {s: c for s, c in zip(top5_par.index, TOP5_COLORS)}

# National average per quarter: par-weighted for vol, count-weighted for count
nat = (sq.groupby("date")
         .agg(par_inst=("par_inst_combined", "sum"),
              par_tot  =("par_cust_oddlot",  "sum"),
              n_inst   =("n_inst_combined",   "sum"),
              n_tot    =("n_cust_oddlot",     "sum"))
         .reset_index())
nat["nat_vol"]   = nat["par_inst"] / nat["par_tot"]
nat["nat_count"] = nat["n_inst"]   / nat["n_tot"]

PANELS = [
    ("inst_share_vol",   "nat_vol",   "Odd-Lots Institutional share: volume (%)"),
    ("inst_share_count", "nat_count", "Odd-Lots Institutional share: count (%)"),
]

fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor="white", sharex=True)
for ax in axes:
    ax.set_facecolor("white")

for ax, (col, nat_col, ylabel) in zip(axes, PANELS):
    for state, grp in sq.groupby("state"):
        grp_s = grp.sort_values("date")
        if state in top5_st:
            ax.plot(grp_s["date"], grp_s[col] * 100,
                    color=top5_color[state], linewidth=1.2, alpha=0.7, label=state)
        else:
            ax.plot(grp_s["date"], grp_s[col] * 100,
                    color="grey", linewidth=0.4, alpha=0.20, label="_nolegend_")
    ax.plot(nat["date"], nat[nat_col] * 100,
            color="#d62728", linewidth=2.2, marker="o", markersize=2.5, label="National avg")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, color="lightgrey")
    ax.legend(ncol=2, fontsize=9)

#axes[0].set_title("Top-5 states by par volume highlighted; all other states in grey")
axes[1].set_xlabel("Quarter")
#plt.suptitle("Institutionalization of Odd-Lot Customer Trades by State (Quarterly)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/state_oddlot_ts.pdf"), bbox_inches="tight")
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
    labels=["Flag-based only", "Clustering-based only", "Markup-based only", "2+ signals overlap"],
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
plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/signal_decomp.pdf"), bbox_inches="tight")
plt.show()


" Robustness 2: Flag-based structural break check (2016 mandate) "
# MSRB mandated NTBC/ATS reporting ~2016. The flag-based signal should show a
# structural upward shift at 2016 if it is real institutional activity, not just
# indicator adoption. If it jumps discontinuously, the level pre-2016 is not comparable.
fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(sy["year"], sy["inst_cust_count"] * 100,
        color="#1f77b4", linewidth=1.5, marker="o", markersize=4, label="Flag-based signal")
ax.axvline(x=2016, color="#d62728", linewidth=1.2, linestyle="--", label="MSRB disclosure mandate")
ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax.set_xlabel("Year")
ax.set_ylabel("Flag-based institutional share (%)", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
#plt.suptitle("Robustness: Flag-Based Signal Around NTBC/ATS Mandate (2016)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_flag_break.pdf"), bbox_inches="tight")
plt.show()


" Robustness 3: Combined signal on block trades (≥$1M) — customer, dealer, all "
# Three panels side by side, one per segment: customer (P/S), dealer (D), all.
# Dealer blocks are nearly 100% institutional by definition (inst_flag fires on ATS/BB),
# providing an upper-bound anchor. Customer blocks should be mostly institutional.
# All-trades combined gives the headline robustness figure.
BLOCK_PLOT = [
    ("block_cust",   "Customer (P/S) ≥$1M",  "n_block_cust"),
    ("block_dealer", "Dealer (D) ≥$1M",      "n_block_dealer"),
    ("block_all",    "All trades ≥$1M",       "n_block_all"),
]
BLOCK_SIGNALS = [
    ("inst_flag",     "Flag-based",        "#1f77b4"),
    ("inst_cluster",  "Clustering-based",  "#9467bd"),
    ("inst_markup",   "Markup-based",      "#2ca02c"),
    ("inst_combined", "Combined",          "#d62728"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 4), facecolor="white", sharey=True)
for ax in axes:
    ax.set_facecolor("white")

for ax, (seg, title, n_col) in zip(axes, BLOCK_PLOT):
    bl = sy[sy[n_col] > 0].copy()
    for sig, lbl, color in BLOCK_SIGNALS:
        col = f"{sig}_{seg}_count"
        if col in bl.columns:
            ax.plot(bl["year"], bl[col] * 100,
                    color=color, linewidth=1.5, marker="o", markersize=4, label=lbl)
    # reference: combined odd-lots
    ax.plot(sy["year"], sy["inst_combined_count"] * 100,
            color="#7f7f7f", linewidth=1.2, marker="s", markersize=3,
            linestyle="--", alpha=0.6, label="Combined odd-lots (ref)")
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, color="lightgrey")

axes[0].set_ylabel("Institutional share (%)", fontsize=11)
axes[0].legend(fontsize=9)
plt.suptitle("Robustness: Institutional Signal on Block Trades (≥$1M) by Trade Type")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_block_combined.pdf"), bbox_inches="tight")
plt.show()


" Robustness 4: Combined signal by par-size bucket over time "
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

# Right: four-measure grouped bar chart for most recent year
recent_yr = sy["year"].max()
row = sy[sy["year"] == recent_yr].iloc[0]
bucket_labels = [d for _, d, _ in BUCKET_META]

MEASURES = [
    ("inst_cluster",      "Clustering-based",   "#9467bd"),
    ("inst_flag",  "Flag-based",         "#1f77b4"),
    ("inst_markup",   "Markup-based",       "#2ca02c"),
    ("inst_combined", "Combined",           "#d62728"),
]
n_measures = len(MEASURES)
x = np.arange(len(BUCKET_META))
w = 0.18

for i, (key, label, color) in enumerate(MEASURES):
    rates = [row.get(f"{key}_{l}", float("nan")) * 100 for l, _, _ in BUCKET_META]
    offset = (i - (n_measures - 1) / 2) * w
    axes[1].bar(x + offset, rates, w, label=label, color=color, alpha=0.85)

axes[1].set_xticks(x)
axes[1].set_xticklabels(bucket_labels)
axes[1].set_xlabel("Par-size bucket")
axes[1].set_ylabel("Institutional share (%)", fontsize=11)
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].grid(True, axis="y", color="lightgrey")
axes[1].legend(fontsize=10)
axes[1].set_title(f"All measures by size bucket ({recent_yr})")

plt.suptitle("Robustness: Combined Signal by Par-Size Bucket")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_bucket_combined.pdf"), bbox_inches="tight")
plt.show()


" Robustness 5: Signal specificity — customer vs dealer odd-lots "
# inst_cluster and inst_markup are computed only on customer trades, so they
# should be ~0 for dealer (D) odd-lots.  inst_flag (ATS/BB/NTBC/WAP) fires on
# D trades, so it should be high.  Validating this confirms signals are not
# accidentally picking up dealer-dealer activity.
ROB5_SIGNALS = [
    ("inst_flag",     "inst_cust_count",    "inst_flag_dealer_odd_count",     "Flag-based",       "#1f77b4"),
    ("inst_cluster",  "inst_cluster_count", "inst_cluster_dealer_odd_count",  "Clustering-based", "#9467bd"),
    ("inst_markup",   "inst_markup_count",  "inst_markup_dealer_odd_count",   "Markup-based",     "#2ca02c"),
    ("inst_combined", "inst_combined_count","inst_combined_dealer_odd_count", "Combined",         "#d62728"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="white", sharey=False)
for ax in axes:
    ax.set_facecolor("white")

for _, cust_col, dealer_col, lbl, color in ROB5_SIGNALS:
    axes[0].plot(sy["year"], sy[cust_col] * 100,
                 color=color, linewidth=1.5, marker="o", markersize=4, label=lbl)
    if dealer_col in sy.columns:
        axes[1].plot(sy["year"], sy[dealer_col] * 100,
                     color=color, linewidth=1.5, marker="o", markersize=4, label=lbl)

for ax, title in zip(axes, ["Customer odd-lots (P/S, par ≤$100K)",
                              "Dealer odd-lots (D, par ≤$100K)"]):
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_ylabel("Signal rate (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, color="lightgrey")
    ax.legend(fontsize=10)
    ax.set_title(title)

plt.suptitle("Robustness: Signal Specificity — Customer vs Dealer Odd-Lots\n"
             "(Clustering- and Markup-based signals should be ~0 for dealer trades)")
plt.tight_layout(rect=[0, 0, 1, 0.93])
#plt.savefig(os.path.join(OUT_DIR, "plots/institutionalization/robustness_dealer_specificity.pdf"), bbox_inches="tight")
plt.show()


