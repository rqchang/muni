import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

PROC_DIR = r"D:\Dropbox\project\muni_bonds\data\processed"
RAW_DIR  = r"D:\Dropbox\project\muni_bonds\data\raw"
OUT_DIR  = r"D:\Dropbox\project\muni_bonds\outputs\plots"


""" Read in data """
# Load NAIC muni holdings (entity_key - cusip8 - date_q), paramt = par amount held
naic = pd.read_csv(f"{PROC_DIR}/naic_muni_holdings_q.csv", index_col=0,
                   usecols=lambda c: c in ["entity_key", "cusip8", "date_q", "paramt"])
naic['cusip8'] = naic["cusip8"].str[:8]
naic["cusip6"] = naic["cusip8"].str[:6]
print("Read NAIC insurer-bond-quarter level muni holdings:", naic.shape)

# Location: cusip_county_info on cusip6
cty = pd.read_csv(f"{RAW_DIR}/cusip_county_info.csv", dtype={"cusip6": str})
print("Read muni issuer location:", cty.shape)

# Mergent amount outstanding (pre-filtered to 2007+, saved by mf_inst_holdings.py)
mergent_q = pd.read_csv(os.path.join(PROC_DIR, "Mergent/muni_amt_oustanding_bq_07.csv"))
print("Read quarterly level Mergent muni bond amount oustanding:", mergent_q.shape)


""" Compute institutional (insurer) holdings """
" Step 1: Aggregate NAIC to bond-quarter (total holdings across all insurers) "
bond_q = (
    naic.groupby(["cusip8", "date_q"], as_index=False)
    .agg(holdings_ins=("paramt", "sum"))  # paramt in thousands
)
bond_q["holdings_ins"] = bond_q["holdings_ins"] * 1000  # convert to dollars
print("Bond-quarter insurer holdings:", bond_q.shape)


" Step 2: Merge and compute bond-level inst share "
# Use Mergent as base — all muni bonds universe; unmatched bonds have holdings_ins = 0
bond_inst = pd.merge(
    mergent_q[["cusip8", "muni_issuer_id", "date_q", "amt_out"]],
    bond_q[["cusip8", "date_q", "holdings_ins"]],
    on=["cusip8", "date_q"], how="left",
)
bond_inst["holdings_ins"] = bond_inst["holdings_ins"].fillna(0)
print("Bond-quarter full universe:", bond_inst.shape)

bond_inst["inst_ins"] = bond_inst["holdings_ins"] / bond_inst["amt_out"]
print("inst_ins raw:", bond_inst["inst_ins"].describe())

fig, ax = plt.subplots(figsize=(8, 4))
bond_inst["inst_ins"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=ax)
ax.set_xlabel("Insurer inst share (raw)")
plt.tight_layout()
plt.show()

# Drop bonds oustside inst_ins = [0, 1]
bond_inst_valid = bond_inst[(bond_inst["inst_ins"] >= 0) & (bond_inst["inst_ins"] <= 1)].copy()
print(f"Valid bonds: {len(bond_inst_valid)} / {len(bond_inst)} ({len(bond_inst_valid)/len(bond_inst):.1%})")  # (99.8%)

# Indicator: bond-quarter ever held by an insurer
bond_inst_valid["held_by_ins"] = (bond_inst_valid["holdings_ins"] > 0).astype(int)
print("Held by insurer:", bond_inst_valid["held_by_ins"].value_counts())

fig, ax = plt.subplots(figsize=(8, 4))
bond_inst_valid["inst_ins"].replace([np.inf, -np.inf], np.nan).dropna().hist(
    bins=50, ax=ax, alpha=0.5, label="All bonds")
bond_inst_valid.loc[bond_inst_valid["held_by_ins"] == 1, "inst_ins"].replace(
    [np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=ax, alpha=0.5, label="Held by insurer")
ax.set_xlabel("Insurer inst share")
ax.legend()
plt.tight_layout()
plt.show()


" Step 3: Aggregate to region-quarter level "
# all sample
region_q = (
    bond_inst_valid.groupby(["muni_issuer_id", "date_q"], as_index=False)
    .agg(
        holdings_ins=("holdings_ins", "sum"),
        amt_out=("amt_out", "sum"),
        inst_ins_eq=("inst_ins", "mean"),
        n_bonds=("cusip8", "nunique"),
    )
)
region_q["inst_ins_w"] = (region_q["holdings_ins"] / region_q["amt_out"]).clip(0, 1)
print("Region-quarter all sample shape:", region_q.shape)

# held by insurers
region_held = (
    bond_inst_valid[bond_inst_valid["held_by_ins"]==1]
    .groupby(["muni_issuer_id", "date_q"], as_index=False)
    .agg(
        holdings_ins=("holdings_ins", "sum"),
        amt_out=("amt_out", "sum"),
        inst_ins_eq=("inst_ins", "mean"),
        n_bonds=("cusip8", "nunique"),
    )
)
region_held["inst_ins_w"] = (region_held["holdings_ins"] / region_held["amt_out"]).clip(0, 1)
print("Region-quarter held by insurer shape:", region_held.shape)

# Plot histograms: 2x2 (row = all / ins.held, col = outstanding-weighted / equal-weighted)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
for ax in axes.flat:
    ax.set_facecolor("white")

region_q["inst_ins_w"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[0, 0])
axes[0, 0].set_title("All bonds")
axes[0, 0].set_xlabel("Outstanding-weighted")
axes[0, 0].set_ylabel("Count")

region_q["inst_ins_eq"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[0, 1])
axes[0, 1].set_title("All bonds")
axes[0, 1].set_xlabel("Equal-weighted")

region_held["inst_ins_w"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[1, 0])
axes[1, 0].set_title("Insurer held")
axes[1, 0].set_xlabel("Outstanding-weighted")
axes[1, 0].set_ylabel("Count")

region_held["inst_ins_eq"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[1, 1])
axes[1, 1].set_title("Insurer held")
axes[1, 1].set_xlabel("Equal-weighted")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "institutionalization/region_inst_ins_hist.pdf"), facecolor="white")
plt.show()


" Step 4: Aggregate to state-quarter level "
bond_inst_valid["state"] = bond_inst_valid["muni_issuer_id"].str.split("_").str[0]

# all sample
state_q = (
    bond_inst_valid.groupby(["state", "date_q"], as_index=False)
    .agg(
        holdings_ins=("holdings_ins", "sum"),
        amt_out=("amt_out", "sum"),
        inst_ins_eq=("inst_ins", "mean"),
        n_bonds=("cusip8", "nunique"),
    )
)

state_fips = cty[["state", "fips_state"]].drop_duplicates().dropna()
state_q = pd.merge(state_q.copy(), state_fips, on="state", how="left")
state_q["inst_ins_w"] = state_q["holdings_ins"] / state_q["amt_out"]
print("State-quarter all sample shape:", state_q.shape)
print(state_q.count())

# held by insurers
state_held = (
    bond_inst_valid[bond_inst_valid["held_by_ins"]==1]
    .groupby(["state", "date_q"], as_index=False)
    .agg(
        holdings_ins=("holdings_ins", "sum"),
        amt_out=("amt_out", "sum"),
        inst_ins_eq=("inst_ins", "mean"),
        n_bonds=("cusip8", "nunique"),
    )
)

state_held = pd.merge(state_held.copy(), state_fips, on="state", how="left")
state_held["inst_ins_w"] = state_held["holdings_ins"] / state_held["amt_out"]
print("State-quarter held by insurers shape:", state_held.shape)
print(state_held.count())

# Plot histogram
fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
for ax in axes.flat:
    ax.set_facecolor("white")

state_q["inst_ins_w"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[0, 0])
axes[0, 0].set_title("All bonds")
axes[0, 0].set_xlabel("Outstanding-weighted")
axes[0, 0].set_ylabel("Count")

state_q["inst_ins_eq"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[0, 1])
axes[0, 1].set_title("All bonds")
axes[0, 1].set_xlabel("Equal-weighted")

state_held["inst_ins_w"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[1, 0])
axes[1, 0].set_title("Insurer held")
axes[1, 0].set_xlabel("Outstanding-weighted")
axes[1, 0].set_ylabel("Count")

state_held["inst_ins_eq"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50, ax=axes[1, 1])
axes[1, 1].set_title("Insurer held")
axes[1, 1].set_xlabel("Equal-weighted")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "institutionalization/state_inst_ins_hist.pdf"), facecolor="white")
plt.show()


""" Save down data """
" Insurer inst share at bond-quarter level "
bond_inst_valid.to_csv(os.path.join(PROC_DIR, "NAIC/ins_inst_iq.csv"))

" Insurer inst share at region-quarter level "
region_q.to_csv(os.path.join(PROC_DIR, "NAIC/ins_inst_gq.csv"))

" Insurer inst share at state-quarter level "
state_q.to_csv(os.path.join(PROC_DIR, "NAIC/ins_inst_sq.csv"))


""" Plot time series """
" Average insurer ownership over time "
state_q["date"] = pd.PeriodIndex(state_q["date_q"], freq="Q").to_timestamp()

ts = state_q[(state_q["date"] >= "2007-10-01") & (state_q["date"] <= "2023-10-01")].copy()
nat = ts.groupby("date")["inst_ins_w"].mean().reset_index()

# top 5 states by average amt_out (largest muni outstanding)
top5_amt = (
    ts.groupby("state")["amt_out"].mean().nlargest(5)
)
top5 = set(top5_amt.index)
COLORS = ["#fc8d59", "#feb24c", "#78c679", "#c994c7", "#bf812d"]
top5_color = {s: c for s, c in zip(top5_amt.index, COLORS)}

# Interactive plotly
fig = go.Figure()
for state, grp in ts.groupby("state"):
    color, width, opacity = (top5_color[state], 2.0, 1.0) if state in top5 else ("grey", 0.7, 0.25)
    fig.add_trace(go.Scatter(
        x=grp["date"], y=grp["inst_ins_w"] * 100,
        mode="lines", name=state,
        line=dict(color=color, width=width),
        opacity=opacity, showlegend=True, legendgroup=state,
        hovertemplate=f"{state}: %{{y:.1f}}%<extra></extra>",
    ))
fig.add_trace(go.Scatter(
    x=nat["date"], y=nat["inst_ins_w"] * 100,
    mode="lines+markers", name="National avg",
    line=dict(color="#d62728", width=3.5),
    marker=dict(size=5, color="#d62728"),
    hovertemplate="National avg: %{y:.1f}%<extra></extra>",
))
fig.update_layout(
    title="Insurer Institutional Ownership by State",
    yaxis_title="Insurer Institutional Ownership Share (%)",
    plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
    xaxis=dict(showgrid=True, gridcolor="lightgrey"),
    yaxis=dict(showgrid=True, gridcolor="lightgrey"),
    legend=dict(ncols=2),
)
fig.write_html(os.path.join(OUT_DIR, "institutionalization/state_inst_ins_ts.html"))
fig.show()

# Static matplotlib
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
ax_s.set_ylabel("Insurer Institutional Ownership Share (%)", fontsize=12)
ax_s.tick_params(axis="x", labelsize=12)
ax_s.grid(True, color="lightgrey")
ax_s.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "institutionalization/state_inst_ins_ts.pdf"), facecolor="white")
plt.show()


" Share of muni bonds held by insurers vs MFs over time "
# % of bonds held by insurers
ins_held_ts = (
    bond_inst_valid.groupby("date_q")["held_by_ins"]
    .mean()
    .reset_index()
    .rename(columns={"held_by_ins": "pct_held"})
)
ins_held_ts["date"] = pd.PeriodIndex(ins_held_ts["date_q"], freq="Q").to_timestamp()
ins_held_ts = ins_held_ts[
    (ins_held_ts["date"] >= "2007-10-01") & (ins_held_ts["date"] <= "2023-10-01")
].sort_values("date")

# % of bonds held by MFs (read from mf_inst_iq.csv)
mf_iq = pd.read_csv(os.path.join(PROC_DIR, "CRSP/mf_inst_iq.csv"), index_col=0,
                    usecols=lambda c: c in ["cusip8", "date_q", "holdings_mf"])
mf_iq["held_by_mf"] = (mf_iq["holdings_mf"] > 0).astype(int)
mf_held_ts = (
    mf_iq.groupby("date_q")["held_by_mf"]
    .mean()
    .reset_index()
    .rename(columns={"held_by_mf": "pct_held"})
)
mf_held_ts["date"] = pd.PeriodIndex(mf_held_ts["date_q"], freq="Q").to_timestamp()
mf_held_ts = mf_held_ts[
    (mf_held_ts["date"] >= "2007-10-01") & (mf_held_ts["date"] <= "2023-10-01")
].sort_values("date")

fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.set_facecolor("white")
ax.plot(ins_held_ts["date"], ins_held_ts["pct_held"] * 100,
        color="#2171b5", linewidth=1.5, label="Insurers")
ax.plot(mf_held_ts["date"], mf_held_ts["pct_held"] * 100,
        color="#d62728", linewidth=1.5, label="Mutual Funds")
ax.set_ylabel("% of bonds held (%)", fontsize=12)
ax.tick_params(axis="x", labelsize=11)
ax.grid(True, color="lightgrey")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "institutionalization/muni_held_ins_mf_ts.pdf"), facecolor="white")
plt.show()
