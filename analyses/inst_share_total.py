"""
inst_share_total.py
Estimate state-level total institutional share of municipal bonds by combining
three investor-class measures:

  1. Non-ETF mutual funds:   direct holdings share from CRSP (mf_sq.csv − etf_sq.csv)
  2. ETFs:                   direct holdings share from CRSP (etf_sq.csv)
  3. Insurers:               direct holdings share from NAIC (ins_sq.csv)
  4. Household channel:      odd-lot institutional trading share from MSRB
                             (inst_oddlot_sm.csv), used to scale the residual
                             non-MF / non-ETF / non-insurer ownership slice

Methodology
-----------
For each state s and quarter t:

  holdings_mf_noetf = holdings_mf_all (from mf_sq) − holdings_etf (from etf_sq)
  inst_mf      = holdings_mf_noetf / amt_out                 [non-ETF MF share]
  inst_etf     = holdings_etf      / amt_out                 [ETF share]
  inst_ins     = holdings_ins      / amt_out                 [insurer share]
  inst_mf_ins  = (holdings_mf_all + holdings_ins) / amt_out  [combined MF+ETF+insurer share]
  residual     = max(1 − inst_mf_ins, 0)                     [household / retail / other slice]

  Fed FFA national sector fractions are used to split the state-level residual into:
    ffa_hh    = Households & Nonprofits                       [must apply odd-lot proxy]
    ffa_other = all remaining sectors (MMF, Depository, State/Local, Retirement, Foreign,
                Brokers, GSEs, CEFs, Credit Unions, Other Financial, etc.)
                                                              [all institutional]
    ffa_res   = ffa_hh + ffa_other                            [FFA residual = denominator]
    frac_hh   = ffa_hh  / ffa_res                             [HH share within residual]
    frac_inst = (ffa_mmf + ffa_other) / ffa_res               [fully-inst share within residual]

  residual_hh    = residual × frac_hh
  residual_other = residual × frac_inst
  inst_share_vol = fraction of odd-lot customer par volume
                   that is institutional (MSRB proxy)         [institutionalization within HH channel]
  inst_hh      = residual_hh    × inst_share_vol              [household-channel institutional share]
  inst_other   = residual_other × 1.0                         [Other (all non-HH): fully institutional]
  inst_total   = inst_mf_ins + inst_hh + inst_other           [total institutional share]

Caveats
-------
- Odd-lot MSRB monthly signal is aggregated to quarters by re-summing par_inst_combined
  and par_cust_oddlot, then recomputing inst_share_vol = par_inst_combined / par_cust_oddlot.
  This preserves the correct volume-weighted average rather than averaging quarterly rates.
- inst_share_vol is a *trading* proxy for institutional ownership within the household channel, 
  not a direct ownership measure.
- Coverage window: SAMPLE_START–SAMPLE_END (set at top of script). The full combined
  measure requires all three sources; partial measures are also reported.

Output files
------------
  inst_total_sq.csv    — state × quarter: all component shares and combined estimate
  inst_total_sy.csv    — state × year:    annual averages of quarterly shares
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from utils.set_paths import PROC_DIR, PLOTS_DIR

SAMPLE_START = "2007Q4"
SAMPLE_END   = "2023Q4"


""" Read in data"""
" CRSP "
# All MFs (includes ETFs): state × quarter
mf = pd.read_csv(os.path.join(PROC_DIR, "CRSP/mf_sq.csv"), index_col=0)
mf = mf[["state", "fips_state", "date_q", "holdings_mf", "amt_out"]].copy()
mf.columns = ["state", "fips_state", "date_q", "holdings_mf_all", "amt_out_mf"]
print("All-MF state-quarter:", mf.shape)

# ETFs: state × quarter
etf = pd.read_csv(os.path.join(PROC_DIR, "CRSP/etf_sq.csv"), index_col=0)
etf = etf[["state", "date_q", "holdings_mf", "amt_out"]].copy()
etf.columns = ["state", "date_q", "holdings_etf", "amt_out_etf"]
print("ETF state-quarter:", etf.shape)

" NAIC "
# Insurers: state × quarter
ins = pd.read_csv(os.path.join(PROC_DIR, "NAIC/ins_sq.csv"), index_col=0)
ins = ins[["state", "fips_state", "date_q", "holdings_ins", "amt_out"]].copy()
ins.columns = ["state", "fips_state", "date_q", "holdings_ins", "amt_out_ins"]
print("Insurer state-quarter:", ins.shape)

" MSRB "
# Odd-lot MSRB: month × state → aggregate to quarter and year
odd_m = pd.read_csv(os.path.join(PROC_DIR, "MSRB/inst_oddlot_sm.csv"))
odd_m["date_q"] = pd.to_datetime(odd_m["trade_month"]).dt.to_period("Q").astype(str)
odd_m["year"]   = pd.to_datetime(odd_m["trade_month"]).dt.year
print("Odd-lot state-month:", odd_m.shape)

# Aggregate to state × quarter (re-sum numerators/denominators for correct vol-weighting)
odd_q = (odd_m.groupby(["state", "date_q"], as_index=False)
               .agg(n_cust_oddlot    =("n_cust_oddlot",     "sum"),
                    n_inst_combined  =("n_inst_combined",    "sum"),
                    par_cust_oddlot  =("par_cust_oddlot",    "sum"),
                    par_inst_combined=("par_inst_combined",  "sum")))
odd_q["inst_share_vol"]   = odd_q["par_inst_combined"] / odd_q["par_cust_oddlot"]
odd_q["inst_share_count"] = odd_q["n_inst_combined"]   / odd_q["n_cust_oddlot"]
print("Odd-lot state-quarter:", odd_q.shape)

# Also aggregate to state × year (for sy annual output)
odd_y = (odd_m.groupby(["state", "year"], as_index=False)
               .agg(n_cust_oddlot    =("n_cust_oddlot",     "sum"),
                    n_inst_combined  =("n_inst_combined",    "sum"),
                    par_cust_oddlot  =("par_cust_oddlot",    "sum"),
                    par_inst_combined=("par_inst_combined",  "sum")))
odd_y["inst_share_vol"]   = odd_y["par_inst_combined"] / odd_y["par_cust_oddlot"]
odd_y["inst_share_count"] = odd_y["n_inst_combined"]   / odd_y["n_cust_oddlot"]
print("Odd-lot state-year:", odd_y.shape)

" FFA "
# Fed FFA: national quarterly sector holdings → residual split fractions
FFA_COLS_MF_ETF_INS = ["Mutual Funds", "ETFs", "Prop-Casualty Insurance", "Life Insurance"]
ffa = pd.read_csv(os.path.join(PROC_DIR, "ffa_muni_holdings_sector.csv"))
ffa["date_q"] = pd.to_datetime(ffa["date"]).dt.to_period("Q").astype(str)
ffa = ffa.dropna(subset=["Households & Nonprofits", "Money Market Funds"]).copy()

# Sectors that are fully institutional (everything not HH, MF, ETF, Insurance — MMF included)
ffa_inst_cols = [c for c in ffa.columns
                 if c not in ["date", "date_q", "Households & Nonprofits"]
                    + FFA_COLS_MF_ETF_INS]
ffa["ffa_hh"]    = ffa["Households & Nonprofits"]
ffa["ffa_other"] = ffa[ffa_inst_cols].sum(axis=1, min_count=1)
ffa["ffa_res"]   = ffa["ffa_hh"] + ffa["ffa_other"]
ffa["frac_hh"]   = ffa["ffa_hh"]  / ffa["ffa_res"]
ffa["frac_inst"] = 1.0 - ffa["frac_hh"]
ffa_q = ffa[["date_q", "frac_hh", "frac_inst"]].copy()
print("FFA quarters:", ffa_q.shape)

# Annual version for sy
ffa["year"] = pd.PeriodIndex(ffa["date_q"], freq="Q").year
ffa_y = (ffa.groupby("year")
            .agg(ffa_hh   =("ffa_hh",    "mean"),
                 ffa_other=("ffa_other", "mean"))
            .reset_index())
ffa_y["ffa_res"]   = ffa_y["ffa_hh"] + ffa_y["ffa_other"]
ffa_y["frac_hh"]   = ffa_y["ffa_hh"]  / ffa_y["ffa_res"]
ffa_y["frac_inst"] = 1.0 - ffa_y["frac_hh"]


""" Compute total institutional share """
" Merge all MF + ETF + Insurer on state × quarter "
sq = pd.merge(mf, etf[["state", "date_q", "holdings_etf", "amt_out_etf"]],
              on=["state", "date_q"], how="outer")
sq = pd.merge(sq, ins[["state", "date_q", "holdings_ins", "amt_out_ins"]],
              on=["state", "date_q"], how="outer")

# Coalesce fips_state (fill from either source)
sq["fips_state"] = sq["fips_state"].fillna(
    sq.groupby("state")["fips_state"].transform("first")
)

# Fill missing holdings with 0 (state-quarter not held by that investor class)
sq["holdings_mf_all"] = sq["holdings_mf_all"].fillna(0)
sq["holdings_etf"]    = sq["holdings_etf"].fillna(0)
sq["holdings_ins"]    = sq["holdings_ins"].fillna(0)

# Non-ETF MF holdings = all MF − ETF
sq["holdings_mf"] = (sq["holdings_mf_all"] - sq["holdings_etf"]).clip(lower=0)

# Diagnostic: compare amt_out across sources where all three are available
both = sq[sq["amt_out_mf"].notna() & sq["amt_out_ins"].notna()].copy()
both["ratio"] = both["amt_out_mf"] / both["amt_out_ins"]
both["abs_diff_pct"] = (both["amt_out_mf"] - both["amt_out_ins"]).abs() / both[["amt_out_mf", "amt_out_ins"]].max(axis=1)
print(f"\namt_out_mf vs amt_out_ins (n={len(both):,} state-quarters with both):")
print(f"  ratio (mf/ins) — mean: {both['ratio'].mean():.4f}  median: {both['ratio'].median():.4f}  "
      f"p5: {both['ratio'].quantile(0.05):.4f}  p95: {both['ratio'].quantile(0.95):.4f}")
print(f"  abs pct diff   — mean: {both['abs_diff_pct'].mean():.4f}  median: {both['abs_diff_pct'].median():.4f}  "
      f"p95: {both['abs_diff_pct'].quantile(0.95):.4f}  max: {both['abs_diff_pct'].max():.4f}")
print(f"  only mf:  {sq['amt_out_ins'].isna().sum():,}  |  only ins: {sq['amt_out_mf'].isna().sum():,}")

# Use the largest of the three amt_out as the common denominator (closest to true Mergent total)
sq["amt_out"] = sq[["amt_out_mf", "amt_out_etf", "amt_out_ins"]].max(axis=1)
sq = sq[sq["amt_out"].notna() & (sq["amt_out"] > 0)].copy()

# Restrict to coverage window
sq = sq[(sq["date_q"] >= SAMPLE_START) & (sq["date_q"] <= SAMPLE_END)].copy()

# Component shares against common denominator
sq["inst_mf"]     = sq["holdings_mf"]      / sq["amt_out"]   # non-ETF MF
sq["inst_etf"]    = sq["holdings_etf"]     / sq["amt_out"]
sq["inst_ins"]    = sq["holdings_ins"]     / sq["amt_out"]
sq["inst_mf_ins"] = sq["holdings_mf_all"]  / sq["amt_out"] + sq["inst_ins"]  # all MF + insurer

# Histogram of unclipped values
fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor="white")
for ax, col, label in zip(axes,
                           ["inst_mf", "inst_etf", "inst_ins", "inst_mf_ins"],
                           ["Non-ETF MF share", "ETF share", "Insurer share", "MF+ETF+Insurer share"]):
    ax.set_facecolor("white")
    data = sq[col].dropna()
    ax.hist(data, bins=80, color="#2171b5", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red",    linewidth=1, linestyle="--")
    ax.axvline(1, color="orange", linewidth=1, linestyle="--")
    ax.set_title(label)
    ax.set_xlabel("Unclipped share")
    ax.set_ylabel("State-quarters")
    pct_below = (data < 0).mean()
    pct_above = (data > 1).mean()
    ax.annotate(f"<0: {pct_below:.2%}\n>1: {pct_above:.2%}",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=9)
plt.suptitle("Institutional share distributions before clipping", fontsize=12)
plt.tight_layout()
plt.show()

for col in ["inst_mf", "inst_etf", "inst_ins", "inst_mf_ins"]:
    s = sq[col]
    print(f"{col:14s}  min={s.min():.4f}  p1={s.quantile(.01):.4f}  "
          f"mean={s.mean():.4f}  median={s.median():.4f}  "
          f"p99={s.quantile(.99):.4f}  max={s.max():.4f}  "
          f"pct<0={(s<0).mean():.3%}  pct>1={(s>1).mean():.3%}")

# Clip after inspection
sq["inst_mf"]     = sq["inst_mf"].clip(0, 1)
sq["inst_etf"]    = sq["inst_etf"].clip(0, 1)
sq["inst_ins"]    = sq["inst_ins"].clip(0, 1)
sq["inst_mf_ins"] = sq["inst_mf_ins"].clip(0, 1)

# Residual (ownership slice not captured by MF or insurer holdings data)
sq["residual"] = (1 - sq["inst_mf_ins"])
print("Merged MF+Insurer state-quarter:", sq.shape)


" Join odd-lot signal and FFA fractions "
sq["year"] = pd.PeriodIndex(sq["date_q"], freq="Q").year
sq = pd.merge(sq, odd_q[["state", "date_q", "inst_share_vol"]], on=["state", "date_q"], how="left")
sq = pd.merge(sq, ffa_q, on="date_q", how="left")

pct_oddlot_match = sq["inst_share_vol"].notna().mean()
pct_ffa_match    = sq["frac_hh"].notna().mean()
print(f"Odd-lot match rate: {pct_oddlot_match:.1%}  |  FFA match rate: {pct_ffa_match:.1%}")


" Compute total institutional share "
# Split residual using national FFA sector proportions:
#   residual_hh    → scaled by odd-lot inst proxy
#   residual_other → MMF + Other sectors, all assumed 100% institutional
sq["residual_hh"]    = sq["residual"] * sq["frac_hh"]
sq["residual_other"] = sq["residual"] * sq["frac_inst"]
sq["inst_hh"]        = sq["residual_hh"]    * sq["inst_share_vol"]
sq["inst_other"]     = sq["residual_other"]  # × 1.0

# Full combined estimate
sq["inst_total"] = sq["inst_mf_ins"] + sq["inst_hh"] + sq["inst_other"]
s = sq["inst_total"].dropna()
print(f"inst_total  min={s.min():.4f}  p1={s.quantile(.01):.4f}  "
      f"mean={s.mean():.4f}  median={s.median():.4f}  "
      f"p99={s.quantile(.99):.4f}  max={s.max():.4f}  "
      f"pct<0={(s<0).mean():.3%}  pct>1={(s>1).mean():.3%}")
sq["inst_total"] = sq["inst_total"].clip(0, 1)

print("\nInstitutional share — national quarterly medians (state-level):")
for col in ["inst_mf", "inst_etf", "inst_ins", "inst_mf_ins", "inst_other", "inst_hh", "inst_total"]:
    print(f"  {col:20s}: {sq[col].median():.3f}")


""" Save down data """
" State-quarter level "
sq = sq.rename(columns={'inst_share_vol': 'inst_oddlot'})
out_sq = sq[[
    "state", "fips_state", "date_q", "year",
    "holdings_mf", "holdings_etf", "holdings_ins", "amt_out",
    "inst_mf", "inst_etf", "inst_ins", "inst_mf_ins",
    "residual", "frac_hh", "frac_inst",
    "inst_oddlot", "inst_hh", "inst_other", "inst_total"
]].sort_values(["state", "date_q"]).reset_index(drop=True)

out_sq.to_csv(os.path.join(PROC_DIR, "inst_total_sq.csv"), index=False)


" State-year level "
# Re-aggregate: mean holdings / mean amt_out per state-year, then join annual odd-lot
sy = (sq.groupby(["state", "fips_state", "year"])
        .agg(holdings_mf =("holdings_mf",  "mean"),
             holdings_etf=("holdings_etf", "mean"),
             holdings_ins=("holdings_ins", "mean"),
             amt_out     =("amt_out",      "mean"))
        .reset_index())

odd_y = odd_y.rename(columns={'inst_share_vol': 'inst_oddlot'})
sy = pd.merge(sy, odd_y[["state", "year", "inst_oddlot"]], on=["state", "year"], how="left")
sy = pd.merge(sy, ffa_y[["year", "frac_hh", "frac_inst"]], on="year", how="left")

sy["inst_mf"]     = sy["holdings_mf"]  / sy["amt_out"]
sy["inst_etf"]    = sy["holdings_etf"] / sy["amt_out"]
sy["inst_ins"]    = sy["holdings_ins"] / sy["amt_out"]
sy["inst_mf_ins"] = ((sy["holdings_mf"] + sy["holdings_etf"] + sy["holdings_ins"]) / sy["amt_out"])
sy["residual"]    = (1 - sy["inst_mf_ins"])
sy["inst_hh"]     = sy["residual"] * sy["frac_hh"]    * sy["inst_oddlot"]
sy["inst_other"]  = sy["residual"] * sy["frac_inst"]
sy["inst_total"]  = sy["inst_mf_ins"] + sy["inst_hh"] + sy["inst_other"]

sy.to_csv(os.path.join(PROC_DIR, "inst_total_sy.csv"), index=False)


""" Plots """
INST_PLOT_DIR = os.path.join(PLOTS_DIR, "institutionalization")
os.makedirs(INST_PLOT_DIR, exist_ok=True)

# Shared component color palette
C = {
    "mf":    "#1f77b4",   # non-ETF Mutual Funds
    "etf":   "#2ca02c",   # ETFs
    "ins":   "#ff7f0e",   # Insurers
    "other": "#8c564b",   # Other incl. MMF (FFA, 100% inst)
    "hh":    "#9467bd",   # Household channel (odd-lot proxy)
    "total": "#d62728",   # Total combined
}

" Plot 1: National average time series — stacked component shares "
# Quarterly national average (par-weighted: sum holdings / sum amt_out across states)
nat_q = (sq.groupby("date_q")
           .agg(
               holdings_mf    =("holdings_mf",    "sum"),
               holdings_etf   =("holdings_etf",   "sum"),
               holdings_ins   =("holdings_ins",   "sum"),
               amt_out        =("amt_out",         "sum"),
               inst_hh_par    =("inst_hh",         lambda x: (x * sq.loc[x.index, "amt_out"]).sum()),
               inst_other_par =("inst_other",      lambda x: (x * sq.loc[x.index, "amt_out"]).sum()),
           )
           .reset_index())
nat_q["date"]         = pd.PeriodIndex(nat_q["date_q"], freq="Q").to_timestamp()
nat_q["inst_mf_w"]    = nat_q["holdings_mf"]    / nat_q["amt_out"]
nat_q["inst_etf_w"]   = nat_q["holdings_etf"]   / nat_q["amt_out"]
nat_q["inst_ins_w"]   = nat_q["holdings_ins"]   / nat_q["amt_out"]
nat_q["inst_hh"]      = nat_q["inst_hh_par"]    / nat_q["amt_out"]
nat_q["inst_other_w"] = nat_q["inst_other_par"] / nat_q["amt_out"]
nat_q["inst_total"]   = (nat_q["inst_mf_w"] + nat_q["inst_etf_w"] + nat_q["inst_ins_w"]
                         + nat_q["inst_other_w"] + nat_q["inst_hh"]).clip(0, 1)
nat_q = nat_q.sort_values("date")

nat_plot = nat_q[nat_q["inst_hh"].notna()].copy()

fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
ax.set_facecolor("white")

ax.stackplot(
    nat_plot["date"],
    nat_plot["inst_mf_w"]    * 100,
    nat_plot["inst_etf_w"]   * 100,
    nat_plot["inst_ins_w"]   * 100,
    nat_plot["inst_other_w"] * 100,
    nat_plot["inst_hh"]      * 100,
    labels=["Mutual Funds (non-ETF)", "ETFs", "Insurers",
            "Others", "Household channel (odd-lot proxy)"],
    colors=[C["mf"], C["etf"], C["ins"], C["other"], C["hh"]],
    alpha=0.85,
)
ax.plot(nat_plot["date"], nat_plot["inst_total"] * 100,
        color=C["total"], linewidth=1.8, label="Total (combined)")

ax.set_ylabel("Institutional share of outstanding (%)", fontsize=11)
ax.set_xlabel("Year-Quarter")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey", zorder=0)
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.15),
          ncol=3, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(INST_PLOT_DIR, "inst_total_national_ts.pdf"), bbox_inches="tight")
plt.show()


" Plot 2: State-level time series — total inst share (top-5 + national) "
sq_plot = sq.copy()
sq_plot["date"] = pd.PeriodIndex(sq_plot["date_q"], freq="Q").to_timestamp()
sq_plot = sq_plot[sq_plot["inst_total"].notna()].copy()

# Top-5 states by average amt_out (largest muni markets)
top5_amt  = sq_plot.groupby("state")["amt_out"].mean().nlargest(5)
top5_st   = set(top5_amt.index)
TOP5_COLORS = ["#fc8d59", "#feb24c", "#78c679", "#c994c7", "#bf812d"]
top5_color  = {s: c for s, c in zip(top5_amt.index, TOP5_COLORS)}

# National par-weighted average per quarter
nat_total = (sq_plot.groupby("date")
               .apply(lambda d: (d["inst_total"] * d["amt_out"]).sum() / d["amt_out"].sum())
               .reset_index(name="nat_total"))

fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
ax.set_facecolor("white")

for state, grp in sq_plot.groupby("state"):
    grp_s = grp.sort_values("date")
    if state in top5_st:
        ax.plot(grp_s["date"], grp_s["inst_total"] * 100,
                color=top5_color[state], linewidth=1.2, alpha=0.7, label=state)
    else:
        ax.plot(grp_s["date"], grp_s["inst_total"] * 100,
                color="grey", linewidth=0.4, alpha=0.20, label="_nolegend_")

ax.plot(nat_total["date"], nat_total["nat_total"] * 100,
        color="#d62728", linewidth=2.2, marker="o", markersize=2.5, label="National avg")

ax.set_ylabel("Total institutional share of outstanding (%)", fontsize=11)
ax.set_xlabel("Year-Quarter")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.grid(True, color="lightgrey")
ax.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(INST_PLOT_DIR, "inst_total_state_ts.pdf"), bbox_inches="tight")
plt.show()


" Plot 3: Cross-state bar chart — most recent year with data "
recent_year = int(sy[sy["inst_total"].notna()]["year"].max())
sy_recent   = sy[sy["year"] == recent_year].copy().sort_values("inst_total", ascending=True)

fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
ax.set_facecolor("white")

y_pos = np.arange(len(sy_recent))
ax.barh(y_pos, sy_recent["inst_mf"]  * 100, color=C["mf"],    label="Mutual Funds (non-ETF)", alpha=0.9)
ax.barh(y_pos, sy_recent["inst_etf"] * 100,
        left=sy_recent["inst_mf"] * 100,
        color=C["etf"], label="ETFs", alpha=0.9)
ax.barh(y_pos, sy_recent["inst_ins"] * 100,
        left=(sy_recent["inst_mf"] + sy_recent["inst_etf"]) * 100,
        color=C["ins"], label="Insurers", alpha=0.9)
ax.barh(y_pos, sy_recent["inst_other"] * 100,
        left=(sy_recent["inst_mf"] + sy_recent["inst_etf"] + sy_recent["inst_ins"]) * 100,
        color=C["other"], label="Others", alpha=0.9)
ax.barh(y_pos, sy_recent["inst_hh"]   * 100,
        left=(sy_recent["inst_mf"] + sy_recent["inst_etf"] + sy_recent["inst_ins"]
              + sy_recent["inst_other"]) * 100,
        color=C["hh"], label="Household channel (odd-lot proxy)", alpha=0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels(sy_recent["state"], fontsize=8)
ax.set_xlabel("Institutional share of outstanding (%)", fontsize=11)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title(f"State-Level Total Institutional Share — {recent_year}", fontsize=12)
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.07),
          ncol=3, frameon=True)
ax.grid(True, axis="x", color="lightgrey", zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(INST_PLOT_DIR, f"inst_total_state_bar_{recent_year}.pdf"), bbox_inches="tight")
plt.show()


" Plot 4: Component breakdown — national annual averages "
nat_y = (sy.groupby("year")
           .apply(lambda d: pd.Series({
               "inst_mf":    (d["inst_mf"]    * d["amt_out"]).sum() / d["amt_out"].sum(),
               "inst_etf":   (d["inst_etf"]   * d["amt_out"]).sum() / d["amt_out"].sum(),
               "inst_ins":   (d["inst_ins"]   * d["amt_out"]).sum() / d["amt_out"].sum(),
               "inst_other": (d["inst_other"] * d["amt_out"]).sum() / d["amt_out"].sum(),
               "inst_hh":    (d["inst_hh"]    * d["amt_out"]).sum() / d["amt_out"].sum(),
               "inst_total": (d["inst_total"] * d["amt_out"]).sum() / d["amt_out"].sum(),
           }))
           .reset_index())
nat_y = nat_y[nat_y["inst_hh"].notna()].copy()

fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(nat_y["year"], nat_y["inst_mf"]    * 100, color=C["mf"],    linewidth=1.5,
        marker="o", markersize=4, label="Mutual Funds (non-ETF)")
ax.plot(nat_y["year"], nat_y["inst_etf"]   * 100, color=C["etf"],   linewidth=1.5,
        marker="o", markersize=4, label="ETFs")
ax.plot(nat_y["year"], nat_y["inst_ins"]   * 100, color=C["ins"],   linewidth=1.5,
        marker="o", markersize=4, label="Insurers")
ax.plot(nat_y["year"], nat_y["inst_other"] * 100, color=C["other"], linewidth=1.5,
        marker="o", markersize=4, label="Other incl. MMF (FFA, 100% inst)")
ax.plot(nat_y["year"], nat_y["inst_hh"]    * 100, color=C["hh"],    linewidth=1.5,
        marker="o", markersize=4, label="Household channel (odd-lot proxy)")
ax.plot(nat_y["year"], nat_y["inst_total"] * 100, color=C["total"], linewidth=2.0,
        marker="o", markersize=4, label="Total (combined)")
ax.set_xlabel("Year")
ax.set_ylabel("Institutional share of outstanding (%)", fontsize=11)
ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
plt.tight_layout()
#plt.savefig(os.path.join(INST_PLOT_DIR, "inst_total_components_ts.pdf"), bbox_inches="tight")
plt.show()
