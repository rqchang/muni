import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

PROC_DIR = r"D:\Dropbox\project\muni_bonds\data\processed"
RAW_DIR  = r"D:\Dropbox\project\muni_bonds\data\raw"
OUT_DIR  = r"D:\Dropbox\project\muni_bonds\outputs\plots"


""" Read in data """
# NAIC insurer-bond-quarter holdings: entity_key - cusip8 - date_q, paramt in thousands
naic = pd.read_csv(f"{PROC_DIR}/naic_muni_holdings_q.csv", index_col=0,
                   usecols=lambda c: c in ["entity_key", "naic_type", "cusip8", "date_q", "paramt"])
naic["cusip8"] = naic["cusip8"].str[:8]
print("Read NAIC insurer-bond-quarter level muni holdings:", naic.shape)

# Location: cusip_county_info on cusip6
cty = pd.read_csv(f"{RAW_DIR}/cusip_county_info.csv", dtype={"cusip6": str})
print("Read muni issuer location:", cty.shape)

# Mergent quarterly universe (pre-filtered to 2007+)
mergent_q = pd.read_csv(os.path.join(PROC_DIR, "Mergent/muni_amt_oustanding_bq_07.csv"))
mergent_q["state"] = mergent_q["muni_issuer_id"].str.split("_").str[0]
print("Read quarterly level Mergent muni bond amount outstanding:", mergent_q.shape)


""" Helper functions """
state_fips = cty[["state", "fips_state"]].drop_duplicates().dropna()

def ins_pipeline(naic_sub, label):
    " Step 1: aggregate to bond-quarter "
    bq = naic_sub.groupby(["cusip8", "date_q"], as_index=False).agg(
        holdings_ins=("paramt", "sum")
    )
    bq["holdings_ins"] = bq["holdings_ins"] * 1000  # thousands → dollars

    " Step 2: merge with Mergent universe, compute inst share "
    bi = pd.merge(
        mergent_q[["cusip8", "muni_issuer_id", "state", "date_q", "amt_out"]],
        bq[["cusip8", "date_q", "holdings_ins"]],
        on=["cusip8", "date_q"], how="left",
    )
    bi["held_by_ins"] = bi["holdings_ins"].notna().astype(int)
    bi["holdings_ins"] = bi["holdings_ins"].fillna(0)
    bi["inst_ins"] = bi["holdings_ins"] / bi["amt_out"]

    " filter valid bonds (inst_ins in [0, 1]) "
    biv = bi[(bi["inst_ins"] >= 0) & (bi["inst_ins"] <= 1)].copy()
    print(f"{label} valid bonds: {len(biv)} / {len(bi)} ({len(biv)/len(bi):.1%})")

    " Step 3: aggregate to region-quarter "
    def agg_region(df):
        rq = df.groupby(["muni_issuer_id", "date_q"], as_index=False).agg(
            holdings_ins=("holdings_ins", "sum"),
            amt_out=("amt_out", "sum"),
            inst_ins_eq=("inst_ins", "mean"),
            n_bonds=("cusip8", "nunique"),
        )
        rq["inst_ins_w"] = (rq["holdings_ins"] / rq["amt_out"]).clip(0, 1)
        return rq

    region_q    = agg_region(biv)
    region_held = agg_region(biv[biv["held_by_ins"] == 1])
    print(f"{label} region_q: {region_q.shape}, region_held: {region_held.shape}")

    " Step 4: aggregate to state-quarter "
    def agg_state(df):
        sq = df.groupby(["state", "date_q"], as_index=False).agg(
            holdings_ins=("holdings_ins", "sum"),
            amt_out=("amt_out", "sum"),
            inst_ins_eq=("inst_ins", "mean"),
            n_bonds=("cusip8", "nunique"),
        )
        sq = pd.merge(sq, state_fips, on="state", how="left")
        sq["inst_ins_w"] = sq["holdings_ins"] / sq["amt_out"]
        return sq

    state_q    = agg_state(biv)
    state_held = agg_state(biv[biv["held_by_ins"] == 1])
    print(f"{label} state_q: {state_q.shape}, state_held: {state_held.shape}")

    return biv, region_q, region_held, state_q, state_held


def plot_hist(all_df, held_df, level_label, fname):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    for ax in axes.flat:
        ax.set_facecolor("white")
    all_df["inst_ins_w"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[0, 0])
    axes[0, 0].set_title("All bonds"); axes[0, 0].set_xlabel("Outstanding-weighted"); axes[0, 0].set_ylabel("Count")
    all_df["inst_ins_eq"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[0, 1])
    axes[0, 1].set_title("All bonds"); axes[0, 1].set_xlabel("Equal-weighted")
    held_df["inst_ins_w"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[1, 0])
    axes[1, 0].set_title("Insurer held"); axes[1, 0].set_xlabel("Outstanding-weighted"); axes[1, 0].set_ylabel("Count")
    held_df["inst_ins_eq"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[1, 1])
    axes[1, 1].set_title("Insurer held"); axes[1, 1].set_xlabel("Equal-weighted")
    #plt.suptitle(level_label)
    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"institutionalization/{fname}.pdf"), facecolor="white", bbox_inches="tight")
    plt.show()


def plot_ts(sq, sh, top5, top5_color, title, fname_stem):
    for df, held_label, fsuffix in [
        (sq, "All Bonds",          fname_stem),
        (sh, "Insurer-Held Bonds", fname_stem + "_held"),
    ]:
        df = df.copy()
        df["date"] = pd.PeriodIndex(df["date_q"], freq="Q").to_timestamp()
        ts = df[(df["date"] >= "2007-10-01") & (df["date"] <= "2023-10-01")]
        nat = ts.groupby("date")["inst_ins_w"].mean().reset_index()

        fig_s, ax_s = plt.subplots(figsize=(12, 5), facecolor="white")
        ax_s.set_facecolor("white")
        for state, grp in ts.groupby("state"):
            grp_sorted = grp.sort_values("date")
            if state in top5:
                ax_s.plot(grp_sorted["date"], grp_sorted["inst_ins_w"] * 100,
                          color=top5_color[state], linewidth=1.5, alpha=0.5, label=state)
            else:
                ax_s.plot(grp_sorted["date"], grp_sorted["inst_ins_w"] * 100,
                          color="grey", linewidth=0.5, alpha=0.25, label="_nolegend_")
        ax_s.plot(nat["date"], nat["inst_ins_w"] * 100, color="#d62728", linewidth=2.5,
                  marker="o", markersize=3, label="National avg")
        ax_s.set_title(f"{title} — {held_label}")
        ax_s.set_ylabel("Insurer Institutional Ownership Share (%)", fontsize=12)
        ax_s.tick_params(axis="x", labelsize=12)
        ax_s.grid(True, color="lightgrey")
        ax_s.legend(ncol=2, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"institutionalization/{fsuffix}.pdf"), facecolor="white")
        plt.show()


""" Run pipeline """
" All Insurers "
biv, region_q, region_held, state_q, state_held = ins_pipeline(naic, "All Insurers")

# save down data
biv.to_csv(os.path.join(PROC_DIR, "NAIC/ins_iq.csv"))
region_q.to_csv(os.path.join(PROC_DIR, "NAIC/ins_gq.csv"))
state_q.to_csv(os.path.join(PROC_DIR, "NAIC/ins_sq.csv"))

# top 5 states by avg amt_out
_sq_ts = state_q.copy()
_sq_ts["date"] = pd.PeriodIndex(_sq_ts["date_q"], freq="Q").to_timestamp()
_ts0 = _sq_ts[(_sq_ts["date"] >= "2007-10-01") & (_sq_ts["date"] <= "2023-10-01")]
top5_amt = _ts0.groupby("state")["amt_out"].mean().nlargest(5)
top5 = set(top5_amt.index)
COLORS = ["#fc8d59", "#feb24c", "#78c679", "#c994c7", "#bf812d"]
top5_color = {s: c for s, c in zip(top5_amt.index, COLORS)}

# save down plots
plot_hist(region_q, region_held, "Region — All Insurers", "region_ins_hist")
plot_hist(state_q,  state_held,  "State — All Insurers",  "state_ins_hist")
plot_ts(state_q, state_held, top5, top5_color,
        "Insurer Institutional Ownership by State", "state_ins_ts")


" Life Insurers "
biv_l, region_q_l, region_held_l, state_q_l, state_held_l = ins_pipeline(
    naic[naic["naic_type"] == "Life"], "Life Insurers")

# save down data
biv_l.to_csv(os.path.join(PROC_DIR, "NAIC/life_ins_iq.csv"))
region_q_l.to_csv(os.path.join(PROC_DIR, "NAIC/life_ins_gq.csv"))
state_q_l.to_csv(os.path.join(PROC_DIR, "NAIC/life_ins_sq.csv"))

# save down plots
plot_hist(region_q_l, region_held_l, "Region — Life Insurers", "region_life_ins_hist")
plot_hist(state_q_l,  state_held_l,  "State — Life Insurers",  "state_life_ins_hist")
plot_ts(state_q_l, state_held_l, top5, top5_color,
        "Life Insurer Institutional Ownership by State", "state_life_ins_ts")


" P&C Insurers "
biv_pc, region_q_pc, region_held_pc, state_q_pc, state_held_pc = ins_pipeline(
    naic[naic["naic_type"] == "P&C"], "P&C Insurers")

# save down data
biv_pc.to_csv(os.path.join(PROC_DIR, "NAIC/pc_ins_iq.csv"))
region_q_pc.to_csv(os.path.join(PROC_DIR, "NAIC/pc_ins_gq.csv"))
state_q_pc.to_csv(os.path.join(PROC_DIR, "NAIC/pc_ins_sq.csv"))

# save down plots
plot_hist(region_q_pc, region_held_pc, "Region — P&C Insurers", "region_pc_ins_hist")
plot_hist(state_q_pc,  state_held_pc,  "State — P&C Insurers",  "state_pc_ins_hist")
plot_ts(state_q_pc, state_held_pc, top5, top5_color,
        "P&C Insurer Institutional Ownership by State", "state_pc_ins_ts")


" Share of muni bonds held by insurers vs MFs over time "
ins_held_ts = (
    biv.groupby("date_q").agg(
        pct_held=("held_by_ins", "mean"),
        n_held=("held_by_ins", "sum"),
    ).reset_index()
)
ins_held_ts["date"] = pd.PeriodIndex(ins_held_ts["date_q"], freq="Q").to_timestamp()
ins_held_ts = ins_held_ts[
    (ins_held_ts["date"] >= "2007-10-01") & (ins_held_ts["date"] <= "2023-10-01")
].sort_values("date")

mf_iq = pd.read_csv(os.path.join(PROC_DIR, "CRSP/mf_iq.csv"), index_col=0,
                    usecols=lambda c: c in ["cusip8", "date_q", "holdings_mf"])
mf_iq["held_by_mf"] = (mf_iq["holdings_mf"] > 0).astype(int)
mf_held_ts = (
    mf_iq.groupby("date_q").agg(
        pct_held=("held_by_mf", "mean"),
        n_held=("held_by_mf", "sum"),
    ).reset_index()
)
mf_held_ts["date"] = pd.PeriodIndex(mf_held_ts["date_q"], freq="Q").to_timestamp()
mf_held_ts = mf_held_ts[
    (mf_held_ts["date"] >= "2007-10-01") & (mf_held_ts["date"] <= "2023-10-01")
].sort_values("date")

# total bonds in universe per quarter
n_total = (
    mergent_q.groupby("date_q")["cusip8"].nunique()
    .reset_index().rename(columns={"cusip8": "n_total"})
)
n_total["date"] = pd.PeriodIndex(n_total["date_q"], freq="Q").to_timestamp()
n_total = n_total[(n_total["date"] >= "2007-10-01") & (n_total["date"] <= "2023-10-01")].sort_values("date")

ins_held_ts = pd.merge(ins_held_ts, n_total[["date_q", "n_total"]], on="date_q", how="left")
mf_held_ts  = pd.merge(mf_held_ts,  n_total[["date_q", "n_total"]], on="date_q", how="left")
ins_held_ts["share_held"] = ins_held_ts["n_held"] / ins_held_ts["n_total"]
mf_held_ts["share_held"]  = mf_held_ts["n_held"]  / mf_held_ts["n_total"]

# number of bonds held
fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(ins_held_ts["date"], ins_held_ts["n_held"],
        color="#2171b5", linewidth=1.5, label="Insurers")
ax.plot(mf_held_ts["date"], mf_held_ts["n_held"],
        color="#d62728", linewidth=1.5, label="Mutual Funds")
ax.plot(n_total["date"], n_total["n_total"],
        color="black", linewidth=1.5, linestyle="--", label="Total universe")
ax.set_ylabel("Number of bonds held", fontsize=12)
ax.tick_params(axis="x", labelsize=11)
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "institutionalization/nbonds_inst_ts.pdf"), facecolor="white")
plt.show()

# share of total bonds held
fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(ins_held_ts["date"], ins_held_ts["share_held"] * 100,
        color="#2171b5", linewidth=1.5, label="Insurers")
ax.plot(mf_held_ts["date"], mf_held_ts["share_held"] * 100,
        color="#d62728", linewidth=1.5, label="Mutual Funds")
ax.set_ylabel("Share of total bonds held (%)", fontsize=12)
ax.tick_params(axis="x", labelsize=11)
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "institutionalization/share_nbonds_inst_ts.pdf"), facecolor="white")
plt.show()
