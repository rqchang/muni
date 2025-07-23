import os
import pandas as pd
import numpy as np
import zipfile

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.4f}'.format

os.chdir(r'/nfs/sloanlab007/projects/muni_bonds_proj')


#%%
""" Functions """
# rename columns: delete contents after the last '_'
def rename_columns(columns):
    return [col.rsplit('_', 1)[0] for col in columns]

# winsorize variables
def remove_outliers(df, var):
    pct1 = df[var].describe(percentiles=[0.01,0.05,0.80,0.9,.95,.99,.995]).T['1%']
    pct99 = df[var].describe(percentiles=[0.01,0.05,0.80,0.9,.95,.99,.995]).T['99%']
    df.loc[(df[var]>pct99),var] = pct99
    df.loc[(df[var]<pct1),var] = pct1
    return df

# generate the list of month-end dates for each row
def generate_dates(row):
    base_date = row['dated_month_end']
    period = row['outstanding_period']
    # one month before, month range, and one month after
    return (
        [base_date - pd.DateOffset(months=1)] +
        [base_date + pd.DateOffset(months=m) for m in range(period + 1)] +
        [base_date + pd.DateOffset(months=period + 1)]
    )


" Parameters "
begin_year = 1980
end_year = 2025


" Documentation "
# Redemption type:
# A: Called --> amount outstanding decreases by the face value of the bonds that were called and redeemed
# B: Called Due To Default --> amount outstanding decreases by the face value of the bonds that were called and redeemed.
# C: Cross-Over Refunded -- amount outstanding stays the same until the crossover date
# D: Cross-Over Refunding - ETM --> amount outstanding stays on the books until maturity
# E: Escrowed To Conversion Date --> ???
# F: Escrowed to Maturity --> amount outstanding stays on the books until maturity
# G: ETM - Interest Only --> amount outstanding stays on the books until maturity
# H: ETM - Principal Only --> amount outstanding stays on the books until maturity
# I: ETM - Waiting Close Of Muni-Forward --> amount outstanding stays on the books until maturity
# J: No Redemption --> amount outstanding stays on the books until maturity
# K: Partially Prerefunded --  amount outstanding stays the same overall, but a portion of it is economically defeased ???
# L: Pre-Refunded --> amount outstanding stays the same until the original bonds are actually redeemed
# M: Prerefunded - Waiting Close Of Muni Forward --> amount outstanding stays the same until the original bonds are actually redeemed
# N: Remarketing  --> amount outstanding stays the same until the original bonds are actually redeemed
# O: Tendered --> amount outstanding decreases by the amount tendered and redeemed ???


#%%
""" Read Mergent FISD muni bond data """
" Muni bond offering information "
# read in data
munis = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/bondinfo.csv', encoding='latin-1') # raw data from website
munis['cusip8'] = munis['cusip_c'].str[:8]
print(munis.columns.to_list())

# select and rename columns
# offering_price_f: expressed as a percentage of par
cols = ['issue_id_l', 'cusip_c', 'cusip8', 'dated_date_d', 'maturity_date_d', 'maturity_id_l', 
        'offering_price_f', 'offering_yield_f', 'total_maturity_offering_amt_f', 'tot_mat_amt_outstanding_f',
        'active_maturity_flag_i', 'redemption_flag_i', 'call_schedule_number_l', 'redeem_method_c']
munis1 = munis[cols].copy()

# format and rename columns
munis1.columns = rename_columns(munis1.columns)
munis1['maturity_date'] = pd.to_datetime(munis1['maturity_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
munis1['dated_date'] = pd.to_datetime(munis1['dated_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
munis1['date_m'] = pd.to_datetime(munis1['dated_date']).dt.to_period('M').astype(str)

# issue_id : cusip = 1 : m
print("Read muni bond information data:", munis1.shape, munis1['dated_date'].min(), munis1['dated_date'].max())
print("Check that data is at cusip9 level:", len(munis1)==len(munis1.drop_duplicates(subset=['cusip'])))


" Muni bond issue information "
issue = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/issueinfo.csv')
issue.columns = rename_columns(issue.columns)
issue['offering_date'] = pd.to_datetime(issue['offering_date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
print("Read muni bond issue information data:", issue.shape, issue['offering_date'].min(), issue['offering_date'].max())
print("Check that data is at issue_id level:", len(issue)==len(issue.drop_duplicates(subset=['issue_id'])))


" Muni bond call schedule information "
# with zipfile.ZipFile('./RA_Ruiquan/data/raw/mergent_muni/dataverse_files.zip', 'r') as zip_ref:
#     zip_ref.extractall('./RA_Ruiquan/data/raw/mergent_muni/dataverse_files')
# column_names = ['issue_id_l', 'maturity_id_l', 'call_schedule_number_l', 'call_date_d', 'call_price_f']
# call = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/dataverse_files/CALLSCHD.DLM', 
#                    delimiter='|', names=column_names, index_col=False)
# call.to_csv('./RA_Ruiquan/data/raw/mergent_muni/callschd.csv', index=False)
call = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/callschd.csv')

# format date and rename columns
call.columns = rename_columns(call.columns)
call['call_date'] = pd.to_datetime(call['call_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
print("Read muni bond call schedule data:", call.shape, call['call_date'].min(), call['call_date'].max())

# aggregate to issue_id-maturity_id level, keep last call_date
call = call.sort_values(by=['issue_id', 'maturity_id', 'call_date'])
call = call.groupby(['issue_id', 'maturity_id', 'call_schedule_number'], as_index=False).last()
print("After keeping the last call date:", call.shape)
print("Check that data is at issue_id-maturity_id-call_number level:", len(call)==len(call.drop_duplicates(subset=['issue_id', 'maturity_id', 'call_schedule_number'])))


" Muni bond redemption information "
# column_names = ['issue_id_l', 'maturity_id_l', 'redemption_type_i', 'redemption_date_d', 'redemption_price_f',
#                 'redemption_amt_f', 'redemption_rate_at_cav_f', 'refunding_cusip_c', 'escrow_type_i',
#                 'calls_defeased_flag_i', 'sink_defeased_flag_i', 'refunding_issue_dtd_d', 'ref_issue_settlement_date_d',
#                 'note_c', 'escrow_percentage_c']
# redemptn = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/dataverse_files/REDEMPTN.DLM', 
#                        delimiter='|', names=column_names, index_col=False)
# redemptn.to_csv('./RA_Ruiquan/data/raw/mergent_muni/redemptn.csv', index=False)
redemptn = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/redemptn.csv')

# format date and rename columns
redemptn.columns = rename_columns(redemptn.columns)
redemptn['redemption_date'] = pd.to_datetime(redemptn['redemption_date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
print("Read muni bond redemption information data:", redemptn.shape, 
      pd.to_datetime(redemptn['redemption_date']).min(), pd.to_datetime(redemptn['redemption_date']).max())
print("Check that data is at issue_id-maturity_id level:", len(redemptn)==len(redemptn.drop_duplicates(subset=['issue_id', 'maturity_id'])))


" Muni bond put/tender schedule "
# column_names = ['issue_id_l', 'maturity_id_l', 'put_or_tender_type_i', 'put_or_tender_date_d', 'put_or_tender_price_f',
#                 'put_or_tender_frequency_i', 'next_put_or_tender_date_d', 'final_put_or_tender_date_d', 'put_or_tender_window1_l',
#                 'put_or_tender_window2_l', 'put_fee_f', 'note_c']
# putsched = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/dataverse_files/PUTSCHED.DLM', 
#                        delimiter='|', names=column_names, index_col=False)
# putsched.to_csv('./RA_Ruiquan/data/raw/mergent_muni/putsched.csv', index=False)
# putsched = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/putsched.csv')

# # format date and rename columns
# putsched.columns = rename_columns(putsched.columns)
# putsched['put_or_tender_date'] = pd.to_datetime(putsched['put_or_tender_date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
# print("Read muni bond put/tender schedule data:", putsched.shape, 
#       pd.to_datetime(putsched['put_or_tender_date']).min(), pd.to_datetime(putsched['put_or_tender_date']).max())

# # keep only type == "S" (tender option) or "J" (mandatory tendor); other types only have ~10 obs
# putsched = putsched.loc[(putsched['put_or_tender_type'].isin(['S', 'J']))]
# putsched['put_or_tender_type'] = pd.Categorical(putsched['put_or_tender_type'], categories=["S", "J"], ordered=True)

# # aggregate to issue_id-maturity_id level, keep "J" first
# putsched = putsched.sort_values(by=['issue_id', 'maturity_id', 'put_or_tender_date', 'put_or_tender_type'])
# putsched = putsched.groupby(['issue_id', 'maturity_id'], as_index=False).last()
# print("After aggregating data:", putsched.shape)
# print("Check that data is at issue_id-maturity_id-date level:", len(putsched)==len(putsched.drop_duplicates(subset=['issue_id', 'maturity_id'])))


" Muni bond partial redemption information "
# column_names = ['issue_id_l', 'maturity_id_l', 'partial_call_type_i', 'partial_call_date_d', 'row_number_l',
#                 'partial_call_rate_f', 'prtl_call_rate_at_cav_f', 'prtl_call_amt_f', 'prtl_call_amt_at_cav_f',
#                 'source_agent_l', 'note_c']
# partredm = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/dataverse_files/PARTREDM.DLM', 
#                        delimiter='|', names=column_names, index_col=False)
# partredm.to_csv('./RA_Ruiquan/data/raw/mergent_muni/partredm.csv', index=False)
# partredm = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/partredm.csv')

# # format date and rename columns
# partredm.columns = rename_columns(partredm.columns)
# partredm['partial_call_date'] = pd.to_datetime(partredm['partial_call_date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
# print("Read muni bond partial redemption information data:", partredm.shape, 
#       pd.to_datetime(partredm['partial_call_date']).min(), pd.to_datetime(partredm['partial_call_date']).max())
# print("Check that data is at issue_id-maturity_id level:", len(partredm)==len(partredm.drop_duplicates(subset=['issue_id', 'maturity_id', 'partial_call_date', 'partial_call_type'])))


#%%
""" Read other data """
" Longstaff liquidity premium "
# bond-date level credit liquidity
lp = pd.read_csv('./RA_Ruiquan/data/raw/Longstaff.csv')
lp['date'] = pd.to_datetime(lp[['Year', 'Month', 'Day']])
lp['date_m'] = pd.to_datetime(lp['date']).dt.to_period('M').astype(str)

# compute bond-date level liquidity premium
lp = lp.rename(columns={'CUSIP':'cusip', 'YearsToMat':'years_to_maturity', 'Cpn':'cpn'})
lp['liq_prem'] = lp['YldTaxexCredLiquAdj'] - lp['YldParBenchmarkBond']
print("Read Longstaff credit liquidity data %s, from %s to %s."%(lp.shape, lp['date'].min(), lp['date'].max()))

# aggregate to bond-month level
lp = lp.sort_values(by=['cusip', 'date_m'])
lp = lp.groupby(['cusip', 'date_m'], as_index=False).last()
print("After aggregating to bond-month level:", lp.shape)
print("Check that Longstaff's data is at bond-month level:", len(lp)==len(lp.drop_duplicates(subset=['date_m', 'cusip'])))


" Read cusip6-county map "
# cusip6-fips_state level
cusip_cty_info = pd.read_csv('./data/raw/cusip_county_info.csv')
print("Read state-issuer map:", cusip_cty_info.shape)


#%%
""" Create base amount outstanding table at the cusip-month level """
" Add dates and state info "
# add call date
munis2 = pd.merge(munis1.copy(), call[['issue_id', 'maturity_id', 'call_schedule_number', 'call_date', 'call_price']].copy(),
                  on=['issue_id', 'maturity_id', 'call_schedule_number'], how='left')
print("After adding call date:", munis2.shape)

# add redemption date
munis2 = pd.merge(munis2.copy(), redemptn[['issue_id', 'maturity_id', 'redemption_type', 'redemption_date', 'redemption_price', 'redemption_amt']].copy(),
                  on=['issue_id', 'maturity_id'], how='left')
print("After adding redemption date:", munis2.shape)

# add offering date
munis2 = pd.merge(munis2.copy(), issue[['issue_id', 'offering_date', 'total_offering_amount']].copy(),
                  on=['issue_id'], how='left')
print("After adding offering date:", munis2.shape)

# add tender date
# munis2 = pd.merge(munis2.copy(), putsched[['issue_id', 'maturity_id', 'put_or_tender_date', 'put_or_tender_price']].copy(),
#                   on=['issue_id', 'maturity_id'], how='left')
# print("After adding offering date:", munis2.shape)

# add state info
munis2['cusip6'] = munis2['cusip'].str[:6]
munis2 = pd.merge(munis2.copy(), cusip_cty_info.copy(), on=['cusip6'], how='left')
print("After adding state information:", munis2.shape)
print(munis2.count())


" Create base table at cusip9-month level "
# initially, we set outstanding period as the months between dated_date and maturity_date
# exclude null maturity_date and dated_date
munis2['year'] = pd.to_datetime(munis2['dated_date']).dt.year
munis3 = munis2.loc[(munis2['dated_date'].notnull()) & (munis2['maturity_date'].notnull()) & (munis2['total_maturity_offering_amt'].notnull()) &
                    (munis2['year']>=begin_year) & (munis2['year']<=end_year)].reset_index(drop=True)
print("After keeping valid obs:", munis3.shape)

# format dates
munis3['dated_date'] = pd.to_datetime(munis3['dated_date'], errors='coerce')
munis3['maturity_date'] = pd.to_datetime(munis3['maturity_date'], errors='coerce')
munis3['call_date'] = pd.to_datetime(munis3['call_date'], errors='coerce')
munis3['redemption_date'] = pd.to_datetime(munis3['redemption_date'], errors='coerce')

# create next and previous dates
munis3['previous_dated_date'] = pd.to_datetime(munis3['dated_date'])-pd.DateOffset(months=1)+pd.offsets.MonthEnd(0)
munis3['next_maturity_date'] = pd.to_datetime(munis3['maturity_date'])+pd.DateOffset(months=1)+pd.offsets.MonthEnd(0)

# create date_m
munis3['dated_month_end'] = munis3['dated_date'] + pd.offsets.MonthEnd(0)
munis3['redemption_date_m'] = munis3['redemption_date'].dt.to_period('M').astype(str)
munis3['maturity_date_m'] = munis3['maturity_date'].dt.to_period('M').astype(str)

# a more efficient way to create cusip-month level panel
# 1. get the minimum dated_date from munis3
min_date = munis3['dated_date'].min()
print("The min dated date in the data is:", min_date)

# 2. create a date range from min_date to 2025-12-31 with monthly frequency (month-end dates)
date_range = pd.date_range(start=min_date, end='2025-12-31', freq='M')
months_df = pd.DataFrame({'date': date_range})

# 3. create a dataframe with all unique cusips
cusip_df = pd.DataFrame({'cusip': munis3['cusip'].unique()})

# 4. cross join the cusip_df with months_df
base = cusip_df.merge(months_df, how='cross')
print("After creating base table at the cusip-month level from 1980 to 2025:", base.shape)


" Add bond info to base table "
# first merge dated_date and maturity_date by cusip
base1 = pd.merge(base.copy(), munis3[['cusip', 'dated_date', 'maturity_date', 'previous_dated_date', 'next_maturity_date']].copy(), 
                 on=['cusip'], how='left')
del base
print("After merging dated_date and maturity_date:", base1.shape)
print(base1[['dated_date', 'maturity_date']].describe())

# keep obs that dated_date-1mo <= date <= maturity_date+1mo
base1['date'] = pd.to_datetime(base1['date'], errors='coerce')
base1 = base1.loc[(base1['date']>=base1['previous_dated_date']) & (base1['date']<=base1['next_maturity_date'])].copy()
print("After keeping obs only between dated_date-1mo and maturity_date+1mo for each cusip:", base1.shape)

# then merge other info
base1 = pd.merge(base1.copy(), munis3.drop(columns=['dated_date', 'maturity_date', 'date_m', 'previous_dated_date', 'next_maturity_date'], errors='ignore'), 
                 on=['cusip'], how='left')
base1['date_m'] = base1['date'].dt.to_period('M').astype(str)
print("After merging cleaned bond info:", base1.shape)


" Compute amount outstanding "
# set amt_outstanding = total_maturity_offering_amt
base1['amt_out'] = base1['total_maturity_offering_amt']

# for one month before issuance and after redemption, amt_outstanding = 0
mask_outside = (base1['date'] < base1['dated_date']) | (base1['date'] > base1['maturity_date'])
base1.loc[mask_outside, 'amt_out'] = 0

# adjust amount outstanding
# get the min date between call_date and redemption_date
base1['end_date'] = np.where(
    base1['call_date'].notnull() & base1['redemption_date'].notnull(),
    base1[['call_date', 'redemption_date']].min(axis=1),
    np.where(base1['call_date'].notnull(), base1['call_date'], base1['redemption_date'])
)

# if end_date exists, create end_amt = redemption_amt | total_maturity_offering_amt (if null redemption_amt)
base1['end_amt'] = np.where(
    base1['end_date'].notnull() & base1['redemption_amt'].notnull(),
    base1['redemption_amt'],
    base1['total_maturity_offering_amt']
)

# create masks for the redemption and call conditions:
mask_redemption = (
    (base1['date'] >= base1['end_date']) &
    (base1['end_date'] == base1['redemption_date']) &
    (base1['date_m'] <= base1['maturity_date_m'])
)
mask_call = (base1['date'] >= base1['end_date']) & (base1['end_date'] == base1['call_date'])

# adjust amt_out for redemption and call events:
# if redemption: date >= end_date & end_date == redemption_date, amt_out == min(amt_out - redemption_amt, 0)
# if call: date >= end_date & end_date == call_date, amt_out = 0
base1.loc[mask_redemption, 'amt_out'] = np.maximum(base1['total_maturity_offering_amt'] - base1['end_amt'], 0)
base1.loc[mask_call, 'amt_out'] = 0
print("After computing amount outstanding:")
print(base1.count())

# check an example:
#base1.loc[base1['cusip']=='00036TAQ7'].sort_values(by=['date'])


" Compute amount issuance "
# create lagged amount outstanding for the balanced panel
print("Check that base table is at the bond-month level:", len(base1)==len(base1.drop_duplicates(subset=['cusip', 'date_m'])))
base1.sort_values(by=['cusip', 'date_m'], inplace=True)
base1['amt_out_lag'] = base1.groupby('cusip')['amt_out'].shift()

# compute amt_iss = amt_out - amt_out_lag
base1['amt_iss'] = base1['amt_out'] - base1['amt_out_lag']

# keep only rows with valid amt_iss
base2 = base1.loc[base1['amt_iss'].notnull()].copy()
del base1
print("After keeping valid amt_iss (dropping observations after redemption):", base2.shape)


" Add liquidity premium "
lp.rename(columns={'date_m':'date_m_longstaff'}, inplace=True)
cols = ['cusip', 'date_m_longstaff', 'liq_prem', 'years_to_maturity', 'cpn']
base2 = pd.merge(base2.copy(), lp[cols].copy(), left_on=['cusip', 'date_m'], right_on=['cusip', 'date_m_longstaff'], how='left')
print("After adding liquidity premium:", base2.shape)
print(base2.count())

# check an example:
#base2.loc[base2['cusip']=='00036TAQ7'].sort_values(by=['date'])
# save file
base2.to_csv('./RA_Ruiquan/data/processed/FISD/muni_amt_outstanding.csv', index=False)
print("Saved down bond-month level muni amount outstanding:", base2.shape, base2['date'].min(), base2['date'].max())
base2.sample(n=1000000).to_csv('./RA_Ruiquan/data/processed/FISD/muni_amt_outstanding_small.csv', index=False)
print("Saved down sample data.")


#%%
""" Aggregate the base table to the issuer-month level """
base2 = pd.read_csv('./RA_Ruiquan/data/processed/FISD/muni_amt_outstanding.csv')
print("Read data:", base2.shape)


" Create dummy variables for state and county "
# keep obs with valid fips location
base3 = base2.loc[((base2['county_fips'].notnull()) | (base2['fips_state'].notnull()))].copy()
del base2
print("After keeping obs with valid fips:", base3.shape)

# if county_fips is null, then the issuer is at state level
base3['is_state'] = 0
base3.loc[base3['county_fips'].isnull(), 'is_state'] = 1
print(base3['is_state'].value_counts())

# create identifier for issuer either at county or state level
# if issuer is state, issuer_id == state
# if issuer is county, issuer_id == state + county_fips
base3['county_fips'] = base3['county_fips'].fillna(0)
base3.loc[base3['is_state']==1, 'muni_issuer_id'] = base3['state'].astype(str).copy()
base3.loc[base3['is_state']==0, 'muni_issuer_id'] = base3['state'].astype(str).copy() + '_' + base3['county_fips'].astype(int).astype(str).copy()
print("Number of issuers:", len(base3['muni_issuer_id'].unique()))


" Collapse from bond-month level to issuer-month level "
# weighted base3
base3['total_amt_out'] = base3.groupby(['muni_issuer_id', 'date_m'])['amt_out'].transform('sum')
base3['w'] = base3['amt_out'] / base3['total_amt_out']
base3['liq_prem_weighted'] = base3['liq_prem'] * base3['w']

# columns needed: amt_net_iss, num_iss, amt_out, liq_prem
base3 = base3.sort_values(by=['cusip', 'date'])
f = {'date':'last', 'is_state':'last', 'amt_out':'sum', 'amt_out_lag':'sum', 'amt_iss':'sum', 'liq_prem_weighted':'sum'}
base_sm = base3.loc[(base3['muni_issuer_id'].notnull())].groupby(['muni_issuer_id', 'date_m'], as_index=False).agg(f)
base_sm = base_sm.rename(columns={'amt_iss':'amt_net_iss'})
print("After aggregating to the issuer-month level:", base_sm.shape)

# num_out
base_out = base3.loc[(base3['amt_out']>0)].copy()
f = {'cusip':'nunique'}
base_sm_out = base_out.loc[(base_out['muni_issuer_id'].notnull())].groupby(['muni_issuer_id', 'date_m'], as_index=False).agg(f)
base_sm_out = base_sm_out.rename(columns={'cusip':'num_out'})

# num_new_iss and amt_new_iss
base_new = base3.loc[(base3['amt_iss']>0)].copy()
f = {'cusip':'nunique', 'amt_iss':'sum'}
base_sm_new = base_new.loc[(base_new['muni_issuer_id'].notnull())].groupby(['muni_issuer_id', 'date_m'], as_index=False).agg(f)
base_sm_new = base_sm_new.rename(columns={'cusip':'num_new_iss', 'amt_iss':'amt_new_iss'})

# merge base_sm + base_sm_new + base_sm_out
base_sm = pd.merge(base_sm.copy(), base_sm_new.copy(), on=['muni_issuer_id', 'date_m'], how='left')
# fill NA num_new_iss with 0
base_sm['num_new_iss'].fillna(0, inplace=True)
base_sm['num_new_iss'] = base_sm['num_new_iss'].astype(int)
# fill NA amt_new_iss with 0
base_sm['amt_new_iss'].fillna(0, inplace=True)
print("After adding num_new_iss and amt_new_iss:", base_sm.shape)

base_sm = pd.merge(base_sm.copy(), base_sm_out.copy(), on=['muni_issuer_id', 'date_m'], how='left')
# fill NA num_out with 0
base_sm['num_out'].fillna(0, inplace=True)
base_sm['num_out'] = base_sm['num_out'].astype(int)
print("After adding num_out:", base_sm.shape)

# filter time range
base_sm['year'] = pd.to_datetime(base_sm['date']).dt.year
base_sm = base_sm.loc[(base_sm['year']<=2025)].copy()
print("There are %s unique muni issuers, from %s to %s."%(len(base_sm['muni_issuer_id'].unique()), base_sm['date_m'].min(), base_sm['date_m'].max()))

# adjust units to millions
base_sm['amt_out_mm'] = base_sm['amt_out'] / 1000000
base_sm['amt_out_mm_lag'] = base_sm['amt_out_lag'] / 1000000
base_sm['amt_net_iss_mm'] = base_sm['amt_net_iss'] / 1000000
base_sm['amt_new_iss_mm'] = base_sm['amt_new_iss'] / 1000000


" Save final data "
base_sm.to_csv('./RA_Ruiquan/data/processed/FISD/muni_issuer_month_bonds.csv', index=False)
print("Saved down muni issuer-month level muni amount outstanding:", base_sm.shape, base_sm['date_m'].min(), base_sm['date_m'].max())



#%%
""" Sum of new issuance by month """
" Consider only state issuers "
# select 1980-2023
munis4 = munis3.loc[(munis3['year']>=1980) & (munis3['year']<=2023)].copy()

# create date_m and month
munis4['date_m'] = pd.to_datetime(munis4['dated_date']).dt.to_period('M').astype(str)
munis4['month'] = pd.to_datetime(munis4['dated_date']).dt.month

# if county_fips is null, then the issuer is at state level
munis4['is_state'] = 0
munis4.loc[(munis4['county_fips'].isnull()) & (munis4['fips_state'].notnull()), 'is_state'] = 1
print(munis4['is_state'].value_counts())

# select state issuance
munis_state = munis4.loc[(munis4['is_state']==1)].copy()
print("After selecting states issuance:", munis_state.shape)

# aggregate to date_m level
munis_y = munis_state.groupby(['date_m']).agg(
    year=('year', 'last'),
    month=('month', 'last'),
    total_maturity_offering_amt=('total_maturity_offering_amt', 'sum'),
    num_new_iss=('cusip', 'nunique'),
    num_state=('state', 'nunique')
).reset_index()
print("After aggregating to date_m level:", munis_y.shape)

# aggregate to state-month level
munis_m = munis_y.groupby(['month']).agg(
    sum_offering_amt=('total_maturity_offering_amt', 'sum'),
    median_offering_amt=('total_maturity_offering_amt', 'median'),
    median_num_iss=('num_new_iss', 'median')
).reset_index()
print("After aggregating to 12-mo level:", munis_m.shape)


" Total amount of new issuance "
fig = px.bar(munis_m, x='month', y='sum_offering_amt', 
             labels={'month': 'Month', 'sum_offering_amt': 'Sum of state-issued offering amount'},
             title='Total state-issued offering amount: 1980-2023')

fig.update_layout(width=1000, height=600)
fig.update_xaxes(dtick=1)
fig.show()


" Median new issuance across years "
# amount
fig = px.bar(munis_m, x='month', y='median_offering_amt',
             labels={'month': 'Month', 'median_offering_amt': 'Median of state-issued offering amount'},
             title='Median state-issued offering amount across years: 1980-2023')

fig.update_layout(width=1000, height=600)
fig.update_xaxes(dtick=1)
fig.show()

# number
fig = px.bar(munis_m, x='month', y='median_num_iss',
             labels={'month': 'Month', 'median_num_iss': 'Median number of state new issues'},
             title='Median number of state new issue events across years: 1980-2023')

fig.update_layout(width=1000, height=600)
fig.update_xaxes(dtick=1)
fig.show()

