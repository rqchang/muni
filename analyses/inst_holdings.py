import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROC_DIR = r"D:\Dropbox\project\muni_bonds\data\processed"
RAW_DIR  = r"D:\Dropbox\project\muni_bonds\data\raw"

""" Read in data """
# Load CRSP muni holdings (cusip8 - crsp_portno - date_q)
crsp = pd.read_csv(f"{PROC_DIR}/crsp_muni_holdings_q.csv", index_col=0)
crsp["cusip6"] = crsp["cusip8"].str[:6]
print("Read CRSP fund-bond-quarter level muni holdings:", crsp.shape)

# Location: cusip_county_info on cusip6 
cty = pd.read_csv(f"{RAW_DIR}/cusip_county_info.csv", dtype={"cusip6": str})
print("Read muni issuer location:", cty.shape)

# Mergent amount outstanding 38 GB file, read in chunks
print("Reading Mergent (chunked)...")
MERGENT_PATH = "D:/Dropbox (Old)/project/muni_bonds/data/processed/muni_amt_outstanding_bm_cleaned.csv"
chunks = []
with open(MERGENT_PATH, "r", encoding="utf-8") as f:
    for chunk in pd.read_csv(
        f,
        usecols=["cusip", "date_m", "amt_out"],
        dtype={"cusip": str},
        chunksize=500_000,
    ):
        chunk["cusip8"] = chunk["cusip"].str[:8]
        chunk = chunk[chunk["date_m"] >= "2007-06"]
        if not chunk.empty:
            chunks.append(chunk)
mergent = pd.concat(chunks, ignore_index=True)
print("Read Mergent bond-month level amount outstanding:", mergent.shape)
mergent.to_csv(os.path.join(PROC_DIR, "Mergent/muni_amt_outstanding_bm_07.csv"))


""" Compute instutional (MF) holdings """
" Create issuer identifier "
# Merge with cusip6
base = pd.merge(
    crsp.copy(),
    cty[["cusip6", "state", "fips_state", "county_fips"]],
    on="cusip6", how="left", suffixes=("", "_cty"),
)
print("After merge:", base.shape)
print("Match rate:", base["fips_state"].notnull().mean().round(3))

# Create muni_issuer_id at state / county level
base = base.loc[(base["county_fips"].notnull()) | (base["fips_state"].notnull())].copy()
print("After keeping obs with valid fips:", base.shape)

# if county_fips is null, then the issuer is a state
base["is_state"] = base["county_fips"].isnull().astype(int)
print(base["is_state"].value_counts())

# else it's a county
base["county_fips"] = base["county_fips"].fillna(0)

# create issuer_id
base.loc[base["is_state"] == 1, "muni_issuer_id"] = base["state"]
base.loc[base["is_state"] == 0, "muni_issuer_id"] = (
    base["state"] + "_" + base["county_fips"].astype(int).astype(str)
)
print("Number of issuers:", base["muni_issuer_id"].nunique())

" Step 1: Aggregate CRSP to bond-quarter "
bond_q = (
    base.groupby(["cusip8", "muni_issuer_id", "date_q"], as_index=False)
    .agg(holdings_mf=("market_val", "sum"))  # market_val in thousands 
)
print("Bond-quarter MF holdings:", bond_q.shape)

" Step 2: Collapse Mergent amount outstanding to bond-quarter "
# convert to thousands
mergent["amt_out_k"] = mergent["amt_out"] / 1000 

# date_m is "YYYY-MM"; date_q = quarter of each month (last() picks last available month per quarter)
mergent["date_q"] = (
    pd.to_datetime(mergent["date_m"]).dt.year.astype(str)
    + "Q"
    + pd.to_datetime(mergent["date_m"]).dt.quarter.astype(str)
)

# collapse to bond-quarter level
mergent = mergent.sort_values(by=['cusip8, date_m'])
mergent_q = (
    mergent.groupby(["cusip8", "date_q"], as_index=False)["amt_out"]
    .last()
)
print("Mergent bond-quarter :", mergent_q.shape)

" Step 3: Merge and compute bond-quarter level inst share "
bond_inst = pd.merge(bond_q, mergent_q, on=["cusip8", "date_q"], how="left")
print("After merging CRSP and Mergent:", bond_inst.shape)

# Bond-level: compute inst_mf = holdings / amt_out, clip to [0, 1]
bond_inst["inst_mf"] = bond_inst["holdings_mf"] / bond_inst["amt_out"]
print("inst_mf raw:", bond_inst["inst_mf"].describe())

bond_inst_valid = bond_inst[(bond_inst["inst_mf"] >= 0) & (bond_inst["inst_mf"] <= 1)].copy()
print(f"Valid bonds: {len(bond_inst_valid)} / {len(bond_inst)} ({len(bond_inst_valid)/len(bond_inst):.1%})")
print("inst_mf dropped:", bond_inst_valid["inst_mf"].describe())

" Step 4: Aggregate to region-quarter level "
# Outstanding-weighted: Inst_{g,t}^{MF,w} = sum(holdings) / sum(amt_out)
# Equal-weighted:       Inst_{g,t}^{MF,eq} = mean(Inst_{i,t}^{MF})
region_q = (
    bond_inst_valid.groupby(["muni_issuer_id", "date_q"], as_index=False)
    .agg(
        holdings_mf=("holdings_mf", "sum"),
        amt_out=("amt_out", "sum"),
        inst_mf_eq=("inst_mf", "mean"),
        n_bonds=("cusip8", "nunique"),
    )
)

region_q["inst_mf_w"] = (region_q["holdings_mf"] / region_q["amt_out"]).clip(0, 1)
print("After aggregating to region-quarter level:", region_q.shape)

" Step 5: Aggregate to state-quarter level "
# Extract state from muni_issuer_id (state issuers: "CA"; county issuers: "CA_75")
bond_inst_valid["state"] = bond_inst_valid["muni_issuer_id"].str.split("_").str[0]

state_q = (
    bond_inst_valid.groupby(["state", "date_q"], as_index=False)
    .agg(
        holdings_mf=("holdings_mf", "sum"),
        amt_out=("amt_out", "sum"),
        inst_mf_eq=("inst_mf", "mean"),
        n_bonds=("cusip8", "nunique"),
    )
)

# add state fips
state_fips = cty[["state", "fips_state"]].drop_duplicates().dropna()
state_q = pd.merge(state_q.copy(), state_fips, on="state", how="left")

state_q["inst_mf_w"] = state_q["holdings_mf"] / state_q["amt_out"]
print("State-quarter shape:", state_q.shape)


""" Save down data """
" MF inst share at bond-quarter level "
bond_inst_valid.to_csv(os.path.join(PROC_DIR, "CRSP/mf_inst_iq.csv"))

" MF inst share at region-quarter level"
region_q.to_csv(os.path.join(PROC_DIR, "CRSP/mf_inst_gq.csv"))

" MF inst share at state-quarter level"
state_q.to_csv(os.path.join(PROC_DIR, "CRSP/mf_inst_sq.csv"))


""" Plot time series """
state_q["date"] = pd.PeriodIndex(state_q["date_q"], freq="Q").to_timestamp()

# national average (equal-weighted across states)
nat = state_q.groupby("date")["inst_mf_w"].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 5))

# individual states (thin, grey)
for state, grp in state_q.groupby("state"):
    ax.plot(grp["date"], grp["inst_mf_w"], color="grey", alpha=0.2, linewidth=0.7)

# national average (bold)
ax.plot(nat["date"], nat["inst_mf_w"], color="steelblue", linewidth=2, label="National avg")

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("MF Institutional Ownership Share")
ax.set_xlabel("")
ax.set_title("Mutual Fund Institutional Ownership by State")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mf_inst_state_ts.png"), dpi=150)
plt.show()
