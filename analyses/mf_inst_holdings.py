import pandas as pd
import pyreadr
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go

PROC_DIR = r"D:\Dropbox\project\muni_bonds\data\processed"
RAW_DIR  = r"D:\Dropbox\project\muni_bonds\data\raw"
OUT_DIR  = r"D:\Dropbox\project\muni_bonds\outputs\plots"
TEMP_DIR = r"D:\Dropbox\project\muni_bonds\data\temp"

""" Read in data """
" CRSP MF holdings data "
# Load CRSP muni holdings (cusip8 - crsp_portno - date_q)
crsp = pd.read_csv(f"{PROC_DIR}/crsp_muni_holdings_q.csv", index_col=0)
crsp["cusip6"] = crsp["cusip8"].str[:6]
print("Read CRSP fund-bond-quarter level muni holdings:", crsp.shape)

# Location: cusip_county_info on cusip6 
cty = pd.read_csv(f"{RAW_DIR}/cusip_county_info.csv", dtype={"cusip6": str})
print("Read muni issuer location:", cty.shape)

# MF identifiers
fund = pyreadr.read_r(os.path.join(PROC_DIR, "CRSP/mf_retail_fm.rds"))[None]
print("retail_fund value counts:\n", fund["retail_fund"].value_counts(dropna=False))

fund["date_q"] = pd.to_datetime(fund["date_m"]).dt.to_period("Q").astype(str)  # → "2023Q1"
fund_q = fund.groupby(['crsp_portno', 'date_q']).last().reset_index()


" Mergent amount outstanding 38 GB file, read in chunks and save down once "
# print("Reading Mergent (chunked)...")
# MERGENT_PATH = "D:/Dropbox (Old)/project/muni_bonds/data/processed/muni_amt_outstanding_bm_cleaned.csv"
# chunks = []
# with open(MERGENT_PATH, "r", encoding="utf-8") as f:
#     for chunk in pd.read_csv(
#         f,
#         usecols=["cusip", "date_m", "amt_out"],
#         dtype={"cusip": str},
#         chunksize=500_000,
#     ):
#         chunk["cusip8"] = chunk["cusip"].str[:8]
#         chunk = chunk[chunk["date_m"] >= "2007-06"]
#         if not chunk.empty:
#             chunks.append(chunk)
# mergent = pd.concat(chunks, ignore_index=True)
# print("Read Mergent bond-month level amount outstanding:", mergent.shape)

# mergent = pd.read_csv(os.path.join(PROC_DIR, "Mergent/muni_amt_outstanding_bm_07.csv"))
# mergent["date_q"] = pd.to_datetime(mergent["date_m"]).dt.to_period("Q").astype(str)
# mergent = mergent.sort_values(["cusip8", "date_m"])
# mergent_q = mergent.groupby(["cusip8", "date_q"], as_index=False)["amt_out"].last()
# print("Mergent bond-quarter:", mergent_q.shape)

# # Attach issuer geography to Mergent universe
# mergent_q["cusip6"] = mergent_q["cusip8"].str[:6]
# mergent_q = pd.merge(mergent_q, cty[["cusip6", "state", "fips_state", "county_fips"]],
#                      on="cusip6", how="left")
# print("Mergent match rate:", mergent_q["fips_state"].notnull().mean().round(3))

# mergent_q = mergent_q.loc[
#     (mergent_q["county_fips"].notnull()) | (mergent_q["fips_state"].notnull())
# ].copy()
# mergent_q["is_state"] = mergent_q["county_fips"].isnull().astype(int)
# mergent_q["county_fips"] = mergent_q["county_fips"].fillna(0)
# mergent_q.loc[mergent_q["is_state"] == 1, "muni_issuer_id"] = mergent_q["state"]
# mergent_q.loc[mergent_q["is_state"] == 0, "muni_issuer_id"] = (
#     mergent_q["state"] + "_" + mergent_q["county_fips"].astype(int).astype(str)
# )
# mergent_q["state"] = mergent_q["muni_issuer_id"].str.split("_").str[0]
# print("Mergent bond-quarter with issuer id:", mergent_q.shape)

# mergent.to_csv(os.path.join(PROC_DIR, "Mergent/muni_amt_outstanding_bm_07.csv"))
# mergent_q.to_csv(os.path.join(PROC_DIR, "Mergent/muni_amt_outstanding_bq_07.csv"))

# Read Mergent quarterly with locations
mergent_q = pd.read_csv(os.path.join(PROC_DIR, "Mergent/muni_amt_oustanding_bq_07.csv"))
print("Read quarterly level Mergent muni bond amount oustanding:", mergent_q.shape)


" MSRB muni transaction price data "
# # Read msrb_{year}_month.rds for 2005-2025, concat, aggregate to cusip8-quarter
# msrb_chunks = []
# for yr in range(2005, 2026):
#     fpath = os.path.join(TEMP_DIR, "MSRB", f"msrb_{yr}_month.rds")
#     if not os.path.exists(fpath):
#         print(f"  Skipping {yr}: file not found")
#         continue
#     result = pyreadr.read_r(fpath)
#     df = result[None]  # pyreadr returns a dict; None key = first/only object
#     msrb_chunks.append(df)
#     print(f"  Read {yr}: {df.shape}")

# msrb = pd.concat(msrb_chunks, ignore_index=True)
# print("MSRB all years:", msrb.shape)
# print("Columns:", msrb.columns.tolist())

# # Create cusip8 and date_q
# msrb["cusip8"] = msrb['cusip'].astype(str).str[:8]
# msrb["date_q"] = (
#     pd.to_datetime(msrb["yyyymm"].astype(int).astype(str), format="%Y%m")
#     .dt.to_period("Q")
#     .astype(str)  # → "2023Q1"
# )

# # Aggregate to cusip8-quarter: par-weighted average price
# msrb_clean = msrb.dropna(subset=["prc", "paramt", "yield"])
# msrb_clean = msrb_clean.assign(
#     prc_x_par=msrb_clean["prc"] * msrb_clean["paramt"],
#     yld_x_par=msrb_clean["yield"] * msrb_clean["paramt"],
# )
# g = msrb_clean.groupby(["cusip8", "date_q"], as_index=False)[
#     ["paramt", "prc_x_par", "yld_x_par"]
# ].sum()
# msrb_q = g.assign(
#     prc=g["prc_x_par"] / g["paramt"],
#     yld=g["yld_x_par"] / g["paramt"],
# ).drop(columns=["prc_x_par", "yld_x_par"])
# print("MSRB cusip8-quarter price:", msrb_q.shape)

# # save down
# msrb_q.to_csv(os.path.join(PROC_DIR, "MSRB/muni_price_q.csv"))

# Read MSRB quarterly price with locations
msrb_q = pd.read_csv(os.path.join(PROC_DIR, "MSRB/muni_price_q.csv"))
print("Read quarterly level MSRB muni price", msrb_q.shape)


""" Compute institutional (MF) holdings at fund-bond-quarter level """
# Merge MSRB price into CRSP at fund-bond-quarter level
crsp_p = pd.merge(crsp, msrb_q[["cusip8", "date_q", "prc"]], on=["cusip8", "date_q"], how="left")
print(f"No MSRB price match: {crsp_p['prc'].isna().mean():.1%}")  # 44%

# adjust market value in thousands
crsp_p["mkt_val"] = crsp_p["market_val"] / 1000

# Fix missing/zero mkt_val using nbr_shares
no_mv = (crsp_p["mkt_val"] == 0) | crsp_p["mkt_val"].isna()
has_prc = crsp_p["prc"].notna()
crsp_p.loc[no_mv &  has_prc, "mkt_val"] = (crsp_p.loc[no_mv &  has_prc, "nbr_shares"] / 1000) * (crsp_p.loc[no_mv &  has_prc, "prc"] / 100)
crsp_p.loc[no_mv & ~has_prc, "mkt_val"] = (crsp_p.loc[no_mv & ~has_prc, "nbr_shares"] / 1000)

# par amount in thousands: paramt = mkt_val / (price / 100); fallback = mkt_val if no price
crsp_p["paramt"] = crsp_p["mkt_val"] / (crsp_p["prc"] / 100)
crsp_p.loc[~has_prc, "paramt"] = crsp_p.loc[~has_prc, "mkt_val"]

# merge with retail fund identifiers (include et_flag)
crsp_p = pd.merge(crsp_p.copy(), fund_q[['crsp_portno', 'date_q', 'retail_fund', 'share_tna_retail']],
                  on=['crsp_portno', 'date_q'], how='left')
print("After merging with price and fund identifiers:", crsp_p.shape)


""" Helper functions """
state_fips = cty[["state", "fips_state"]].drop_duplicates().dropna()

def fund_pipeline(crsp_sub, label):
    " bond-quarter aggregation "
    bq = crsp_sub.groupby(["cusip8", "date_q"], as_index=False).agg(holdings_mf=("paramt", "sum"))
    bq["holdings_mf"] = bq["holdings_mf"] * 1000  # thousands → dollars

    " merge with mergent universe "
    bi = pd.merge(
        mergent_q[["cusip8", "muni_issuer_id", "state", "date_q", "amt_out"]],
        bq[["cusip8", "date_q", "holdings_mf"]],
        on=["cusip8", "date_q"], how="left",
    )
    bi["held_by_mf"] = (bi["holdings_mf"].notnull()).astype(int)
    bi["holdings_mf"] = bi["holdings_mf"].fillna(0)
    bi["inst_mf"] = bi["holdings_mf"] / bi["amt_out"]

    " filter valid, add indicators "
    biv = bi[(bi["inst_mf"] >= 0) & (bi["inst_mf"] <= 1)].copy()
    print(f"{label} valid bonds: {len(biv)} / {len(bi)} ({len(biv)/len(bi):.1%})")

    " aggregate to region-quarter "
    def agg_region(df):
        rq = df.groupby(["muni_issuer_id", "date_q"], as_index=False).agg(
            holdings_mf=("holdings_mf", "sum"),
            amt_out=("amt_out", "sum"),
            inst_mf_eq=("inst_mf", "mean"),
            n_bonds=("cusip8", "nunique"),
        )
        rq["inst_mf_w"] = rq["holdings_mf"] / rq["amt_out"]
        return rq

    region_q    = agg_region(biv)
    region_held = agg_region(biv[biv["held_by_mf"] == 1])
    print(f"{label} region_q: {region_q.shape}, region_held: {region_held.shape}")

    " aggregate to state-quarter "
    def agg_state(df):
        sq = df.groupby(["state", "date_q"], as_index=False).agg(
            holdings_mf=("holdings_mf", "sum"),
            amt_out=("amt_out", "sum"),
            inst_mf_eq=("inst_mf", "mean"),
            n_bonds=("cusip8", "nunique"),
        )
        sq = pd.merge(sq, state_fips, on="state", how="left")
        sq["inst_mf_w"] = sq["holdings_mf"] / sq["amt_out"]
        return sq

    state_q    = agg_state(biv)
    state_held = agg_state(biv[biv["held_by_mf"] == 1])
    print(f"{label} state_q: {state_q.shape}, state_held: {state_held.shape}")

    return biv, region_q, region_held, state_q, state_held


def plot_hist(all_df, held_df, level_label, fname):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    for ax in axes.flat:
        ax.set_facecolor("white")
    all_df["inst_mf_w"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[0, 0])
    axes[0, 0].set_title("All bonds"); axes[0, 0].set_xlabel("Outstanding-weighted"); axes[0, 0].set_ylabel("Count")
    all_df["inst_mf_eq"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[0, 1])
    axes[0, 1].set_title("All bonds"); axes[0, 1].set_xlabel("Equal-weighted")
    held_df["inst_mf_w"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[1, 0])
    axes[1, 0].set_title("MF held"); axes[1, 0].set_xlabel("Outstanding-weighted"); axes[1, 0].set_ylabel("Count")
    held_df["inst_mf_eq"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[1, 1])
    axes[1, 1].set_title("MF held"); axes[1, 1].set_xlabel("Equal-weighted")
    #plt.suptitle(level_label)
    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"institutionalization/{fname}.pdf"), facecolor="white", bbox_inches="tight")
    plt.show()


def plot_ts(sq, sh, top5, top5_color, title, fname_stem):
    for df, held_label, fsuffix in [
        (sq, "All Bonds",     fname_stem),
        (sh, "MF-Held Bonds", fname_stem + "_held"),
    ]:
        df = df.copy()
        df["date"] = pd.PeriodIndex(df["date_q"], freq="Q").to_timestamp()
        ts = df[(df["date"] >= "2007-10-01") & (df["date"] <= "2023-10-01")]
        nat = ts.groupby("date")["inst_mf_w"].mean().reset_index()

        fig_s, ax_s = plt.subplots(figsize=(12, 5), facecolor="white")
        ax_s.set_facecolor("white")
        for state, grp in ts.groupby("state"):
            grp_sorted = grp.sort_values("date")
            if state in top5:
                ax_s.plot(grp_sorted["date"], grp_sorted["inst_mf_w"] * 100,
                          color=top5_color[state], linewidth=1.5, alpha=0.5, label=state)
            else:
                ax_s.plot(grp_sorted["date"], grp_sorted["inst_mf_w"] * 100,
                          color="grey", linewidth=0.5, alpha=0.25, label="_nolegend_")
        ax_s.plot(nat["date"], nat["inst_mf_w"] * 100, color="#d62728", linewidth=2.5,
                  marker="o", markersize=3, label="National avg")
        ax_s.set_title(f"{title} — {held_label}")
        ax_s.set_ylabel("MF Institutional Ownership Share (%)", fontsize=12)
        ax_s.tick_params(axis="x", labelsize=12)
        ax_s.grid(True, color="lightgrey")
        ax_s.legend(ncol=2, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"institutionalization/{fsuffix}.pdf"), facecolor="white")
        plt.show()


""" Run pipeline: All MFs, Retail MFs, ETFs """
" All MFs "
biv_all, region_q, region_held, state_q, state_held = fund_pipeline(crsp_p, "All MFs")
# All MFs valid bonds: 76888906 / 77750630 (98.9%)

# save down data
biv_all.to_csv(os.path.join(PROC_DIR, "CRSP/mf_iq.csv"))
region_q.to_csv(os.path.join(PROC_DIR, "CRSP/mf_gq.csv"))
state_q.to_csv(os.path.join(PROC_DIR, "CRSP/mf_sq.csv"))

# top 5 states by avg amt_out (computed once, reused for retail/ETF plots)
_sq_ts = state_q.copy()
_sq_ts["date"] = pd.PeriodIndex(_sq_ts["date_q"], freq="Q").to_timestamp()
_ts0 = _sq_ts[(_sq_ts["date"] >= "2007-10-01") & (_sq_ts["date"] <= "2023-10-01")]
top5_amt = _ts0.groupby("state")["amt_out"].mean().nlargest(5)
top5 = set(top5_amt.index)
COLORS = ["#fc8d59", "#feb24c", "#78c679", "#c994c7", "#bf812d"]
top5_color = {s: c for s, c in zip(top5_amt.index, COLORS)}

# save down plots
plot_hist(region_q, region_held, "Region — All MFs", "region_mf_hist")
plot_hist(state_q,  state_held,  "State — All MFs",  "state_mf_hist")
plot_ts(state_q, state_held, top5, top5_color,
        "MF Institutional Ownership by State: All MFs", "state_mf_ts")


" Retail MFs "
biv_r, region_q_r, region_held_r, state_q_r, state_held_r = fund_pipeline(
    crsp_p[crsp_p["retail_fund"] == 1], "Retail MFs")

# save down data
biv_r.to_csv(os.path.join(PROC_DIR, "CRSP/retail_mf_iq.csv"))
region_q_r.to_csv(os.path.join(PROC_DIR, "CRSP/retail_mf_gq.csv"))
state_q_r.to_csv(os.path.join(PROC_DIR, "CRSP/retail_mf_sq.csv"))

# save down plots
plot_hist(region_q_r, region_held_r, "Region — Retail MFs", "region_retail_mf_hist")
plot_hist(state_q_r,  state_held_r,  "State — Retail MFs",  "state_retail_mf_hist")
plot_ts(state_q_r, state_held_r, top5, top5_color,
        "MF Institutional Ownership by State: Retail MFs", "state_retail_mf_ts")


" ETFs "
# ETFs (et_flag == "F")
biv_e, region_q_e, region_held_e, state_q_e, state_held_e = fund_pipeline(
    crsp_p[crsp_p["et_flag"] == "F"], "ETFs")

# save down data
biv_e.to_csv(os.path.join(PROC_DIR, "CRSP/etf_iq.csv"))
region_q_e.to_csv(os.path.join(PROC_DIR, "CRSP/etf_gq.csv"))
state_q_e.to_csv(os.path.join(PROC_DIR, "CRSP/etf_sq.csv"))

# save down plots
plot_hist(region_q_e, region_held_e, "Region — ETFs", "region_etf_hist")
plot_hist(state_q_e,  state_held_e,  "State — ETFs",  "state_etf_hist")
plot_ts(state_q_e, state_held_e, top5, top5_color,
        "MF Institutional Ownership by State: ETFs", "state_etf_ts")
