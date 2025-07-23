# <codecell>
import os
import pandas as pd
import numpy as np
import datetime

# import wrds
# db = wrds.Connection()
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.4f}'.format

os.chdir(r'/nfs/sloanlab007/projects/muni_bonds_proj')


#<codecell>
""" Read fund and bond data """
# " WRDS monthly bond returns for prices, amounts outstanding, and credit ratings "
# wbr = pd.read_csv('./munis/data/raw/WBR/bond_returns_master.csv', encoding='latin-1')
# wbr['cusip8'] = wbr['cusip'].str[:8]
# wbr['date_m'] = pd.to_datetime(wbr['date']).dt.to_period('M').astype(str)
# print("Read bond-month level WRDS Bond returns data %s, from %s to %s."%(wbr.shape, str(wbr.date.min()), str(wbr.date.max())))

" Read Mergent muni bond data "
munis = pd.read_csv('./RA_Ruiquan/data/raw/mergent_muni/bondinfo.csv', encoding='latin-1') # raw data from website
munis['cusip8'] = munis['cusip_c'].str[:8]
print("Read bond level municipal bond issuance data:", munis.shape)
print(munis.head())
print("Bondinfo goes from %s to %s."%(str(munis.dated_date_d.min()), str(munis.dated_date_d.max())))
munis['year'] = pd.to_datetime(munis['dated_date_d'], format = '%Y%m%d').dt.year

print("Number of issuances per year: ", munis.groupby(['year'], as_index=False).agg({'cusip8':'count'}))
# " Muni bonds location 1 "
# cusip_link = pd.read_csv('./munis/data/cusip_link.csv')
# cusip_link = cusip_link.loc[:, ~cusip_link.columns.str.contains('^Unnamed')].copy()
# cusip_link['cusip8'] = cusip_link['cusip_c'].str[:8]
# print("Read munis bond location data 1:", cusip_link.shape)

# " Muni bonds location "
# # data is at cusip9-call_date level
# cusip_link2 = pd.read_csv('./munis/data/cusip_link2.csv')
# cusip_link2 = cusip_link2.loc[:, ~cusip_link2.columns.str.contains('^Unnamed')].copy()
# cusip_link2['cusip8'] = cusip_link2['cusip'].str[:8]
# print("Read munis bond location data 2:", cusip_link2.shape)

# # aggregate to cusip8 level
# f = {'state':'last', 'total_maturity_offering_amt':'last','offering_price':'last', 
#      'offering_yield':'last', 'yield_treas_final':'last',  'tax_code':'last', 
#      'debt_type':'last', 'capital_purpose':'last', 'state_tax':'last'}
# vcusip_link2 = cusip_link2.groupby(['cusip8'], as_index=False).agg(f)
# print("After aggregating location data 2 to bond level:", vcusip_link2.shape)

" Read Mergent muni bond rating information "
# data is at cusip9-call_date level
rating = pd.read_csv('./munis/data/rating_merged.csv', encoding='latin-1') # raw data from website
rating['cusip8'] = rating['cusip'].str[:8]
print("Read munis rating information data:", rating.shape)

# aggregate to cusip8 level
f = {'state':'last', 'tax_code':'last', 'debt_type':'last', 'capital_purpose':'last', 
     'state_tax':'last', 'yield_treas_final':'last', 'grade':'last','grade_LT':'last', 
     'grade_ST':'last', 'grade_ST_ENH':'last', 'grade_LT_ENH':'last', 'invgrade':'last'}
vrating = rating.groupby(['cusip8'], as_index=False).agg(f)
print("After aggregating rating information to bond level:", vrating.shape)

" CRSP MF quarterly summary "
# # nav_latest: BEGINS 1961 WITH YEAREND VALUES SWITCHES TO QUARTERLY VALUES IN 2000
# # tna_latest: BEGINS 1961, REPORTED IN MILLIONS, BY CONVENTION, 0.1 REFLECTS ALL TOTAL NET ASSET VALUES <=$100,000
# fund_sum_cols = ['caldt', 'crsp_portno', 'crsp_fundno', 'per_corp', 'per_govt', 'per_cash', 
#                  'tna_latest', 'index_fund_flag', 'et_flag', 'retail_fund', 'inst_fund']
# fund_sum = db.get_table(library='crsp', table='fund_summary2', columns=fund_sum_cols)   

# # keep only with crsp_portno
# fund_sum = fund_sum.loc[~fund_sum['crsp_portno'].isnull()].copy()
# fund_sum['date_q'] = pd.to_datetime(fund_sum['caldt']).dt.to_period('Q').astype(str)
# fund_sum['date_m'] = pd.to_datetime(fund_sum['caldt']).dt.to_period('M').astype(str)
# print("Downloaded CRSP MF fund summary %s, from %s to %s."%(fund_sum.shape, str(fund_sum.caldt.min()), str(fund_sum.caldt.max())))

# # crps_fundno-quarter level
# fund_sum = fund_sum.groupby(['crsp_fundno', 'date_q'], as_index=False).last()
# print("After cleaning fundsum to crsp_fundno-quarter level:", fund_sum.shape)

# # compute retail vs institutional portion of fund
# fund_sum.loc[(fund_sum['retail_fund']=="Y"), 'retail_portion'] = fund_sum['tna_latest'].copy()
# fund_sum.loc[(fund_sum['inst_fund']=="Y"), 'inst_portion'] = fund_sum['tna_latest'].copy()

# # aggregate fund_sum by crsp_PORTNO-quarter
# f = {'caldt':'last', 'per_corp':'last', 'per_govt':'last', 'per_cash':'last', 'tna_latest':'sum', 
#      'index_fund_flag':'last', 'et_flag':'last', 'retail_portion':'sum', 'inst_portion':'sum'}
# vfund_sum = fund_sum.groupby(['crsp_portno', 'date_q'], as_index=False).agg(f)
# vfund_sum.to_csv('./munis/data/raw/CRSP/fund_sum_portnoq.csv')
# print("After cleaning fundsum to crsp_PORTNO-quarter level:", vfund_sum.shape)

fund_sum = pd.read_csv('./munis/data/raw/CRSP/fund_sum_portnoq.csv')
fund_sum = fund_sum.loc[:, ~fund_sum.columns.str.contains('^Unnamed')].copy()
print("Read crsp_portno-quarter level MF fund summary data %s, from %s to %s."%(fund_sum.shape, str(fund_sum.caldt.min()), str(fund_sum.caldt.max())))


#<codecell>
""" Read and clean CRSP MF monthly holdings data """
" Read holdings data that cusip is in munis "
chunksize = 1000000 
cmf1_temp = []
crsp_cols = ['crsp_portno', 'report_dt', 'market_val', 'cusip', 'nbr_shares']

for cmf in pd.read_csv('./munis/data/raw/CRSP/crsp_mf_holdings_cusip_m.csv', usecols=crsp_cols, chunksize=chunksize):
    # create relevant columns
    cmf['year'] = pd.to_datetime(cmf['report_dt']).dt.year
    cmf['qtr'] = pd.to_datetime(cmf['report_dt']).dt.quarter
    cmf['month'] = pd.to_datetime(cmf['report_dt']).dt.month
    cmf['date_m'] = pd.to_datetime(cmf['report_dt']).dt.to_period('M').astype(str)
    cmf['date_q'] = pd.to_datetime(cmf['report_dt']).dt.to_period('Q').astype(str)
    cmf['cusip8'] = cmf['cusip'].copy()
    cmf.loc[(cmf['cusip'].isnull()), 'cusip'] = 'Other'

    # only keep if there is a cusip8 and is in the munis
    cmf1 = cmf.loc[cmf['cusip8'].notnull()].copy()
    cmf1 = cmf1.loc[cmf1['cusip8'].isin(munis['cusip8'])].copy()
    cmf1_temp.append(cmf1.copy())

vcmf = pd.concat(cmf1_temp)
print("Read CRSP MF muni bond holdings from CRSP:", vcmf.shape)
print("Number of unique crsp_portno funds that have munis:", len(set(vcmf.crsp_portno)))

" Aggregate muni holdings to fund-bond-month level "
print("Check that MF holdings is at fund-bond-month level:", len(vcmf)==len(vcmf.drop_duplicates(subset=['crsp_portno', 'cusip8', 'date_m'])))
print(vcmf.groupby(['date_m'], as_index=False).agg({'crsp_portno':'count'}))

vcmf = vcmf.sort_values(by=['crsp_portno', 'cusip8', 'date_m'], na_position='first')
f = {'report_dt':'last', 'year':'last', 'qtr':'last', 'month':'last', 'date_q':'last', 
     'nbr_shares':'sum', 'market_val':'sum'}
vdata = vcmf.groupby(['crsp_portno', 'cusip8', 'date_m'], as_index=False).agg(f)

print("After aggregating to fund-bond-month level:", vdata.shape)
print("Check that CRSP muni holdings is at fund-bond-month level:", 
      len(vdata)==len(vdata.drop_duplicates(subset=['crsp_portno', 'cusip8', 'date_m'])))


#<codecell>
""" Merge with bond characteristics """
" Add muni bond issuance "
vdata = pd.merge(vdata.copy(), munis.copy(), how='left', on=['cusip8'], suffixes=['', '_FISD'])
print("After adding FISD muni bond issuance data:", vdata.shape)

" Add muni bond rating "
vdata = pd.merge(vdata.copy(), vrating.copy(), how='left', on=['cusip8'], suffixes=['', '_FISD'])
print("After adding FISD muni rating information:", vdata.shape)

" Add fund_sum "
vdata = pd.merge(vdata.copy(), fund_sum.copy(), how='left', on=['crsp_portno', 'date_q'])
print("After adding quarterly fundsum:", vdata.shape)

# " Add WRDS bond returns "
# cols = ['cusip8', 'date_m', 'offering_amt', 'rating_cat', 'tmt', 'ret_eom', 'yield', 't_yld_pt',
#         't_spread', 'price_eom', 'cusip', 'amt_outstanding', 'yc_yield', 'cds_spread', 'rating_rank',
#         'rcat', 'seniority3', 'cs_yield', 'basis_yield', 'bid_ask', 'sicf']
# vdata = pd.merge(vdata.copy(), wbr[cols], how='left', on=['cusip8', 'date_m'], suffixes=['', '_WBR'])
# print("After adding monthly WBR bond issuance data:", vdata.shape)
print(vdata.loc[vdata['date_m']>='2002-07'].count())

" Select columns "
vcols = ['crsp_portno', 'cusip8', 'date_m', 'report_dt', 'year', 'qtr', 'month', 'date_q', 
         'nbr_shares', 'market_val', 'cusip_c', 'issue_id_l', 'isin_c', 'dated_date_d', 'maturity_date_d', 
         'total_maturity_offering_amt_f', 'offering_price_f', 'offering_yield_f', 'moody_long_rating_c', 
         'moody_short_rating_c', 'sp_long_rating_c', 'sp_short_rating_c', 'grade', 'grade_LT', 'grade_ST', 
         'invgrade', 'state', 'yield_treas_final', 'tax_code', 'debt_type', 'capital_purpose', 'state_tax', 
         'index_fund_flag', 'et_flag', 'retail_portion', 'inst_portion', 'tna_latest', 'per_corp', 'per_govt', 'per_cash']
vdata_ibm = vdata[vcols].copy()
print("After select columns:", vdata_ibm.shape)


#<codecell>
""" Aggregate muni holdings to fund-bond-quarter level """
vdata_ibm = vdata_ibm.sort_values(by=['crsp_portno', 'cusip8', 'date_m'], na_position='first')
f = {'report_dt':'last', 'year':'last', 'qtr':'last', 'month':'last', 'date_m':'last', 'nbr_shares':'last', 'market_val':'last', 
     'cusip_c':'last', 'issue_id_l':'last', 'isin_c':'last', 'dated_date_d':'last', 'maturity_date_d':'last', 
     'total_maturity_offering_amt_f':'last', 'offering_price_f':'last', 'offering_yield_f':'last', 'moody_long_rating_c':'last', 
     'moody_short_rating_c':'last', 'sp_long_rating_c':'last', 'sp_short_rating_c':'last', 'grade':'last', 'grade_LT':'last', 
     'grade_ST':'last', 'invgrade':'last', 'state':'last', 'yield_treas_final':'last', 'tax_code':'last', 'debt_type':'last', 
     'capital_purpose':'last', 'state_tax':'last', 'index_fund_flag':'last', 'et_flag':'last', 'retail_portion':'last', 
     'inst_portion':'last', 'tna_latest':'last', 'per_corp':'last', 'per_govt':'last', 'per_cash':'last'}
vdata_ibq = vdata_ibm.groupby(['crsp_portno', 'cusip8', 'date_q'], as_index=False).agg(f)

print("After aggregating to fund-bond-quarter level:", vdata_ibq.shape)
print("Check that CRSP muni holdings is at fund-bond-quarter level:", 
      len(vdata_ibq)==len(vdata_ibq.drop_duplicates(subset=['crsp_portno', 'cusip8', 'date_q'])))
print(vdata_ibq.loc[vdata_ibq['date_q']>='2002Q3'].count())


#<codecell>
""" Save data """
#vcmf.sample(n=1000000).to_csv('./munis/data/processed/CRSP/crsp_muni_holdings_m_all_small.csv')
#vcmf.to_csv('./munis/data/processed/CRSP/crsp_muni_holdings_m_all.csv')

vdata.sample(n=1000000).to_csv('./munis/data/processed/CRSP/crsp_muni_holdings_m_small.csv')
vdata_ibm.to_csv('./munis/data/processed/CRSP/crsp_muni_holdings_m.csv')
print("Saved fund-muni-month level data:", vdata_ibm.shape)

vdata_ibq.sample(n=1000000).to_csv('./munis/data/processed/CRSP/crsp_muni_holdings_q_small.csv')
vdata_ibq.to_csv('./munis/data/processed/CRSP/crsp_muni_holdings_q.csv')
print("Saved fund-muni-quarter level data:", vdata_ibq.shape)

" Check data "
print(vdata.columns.tolist())
print("Data has: %s unique funds, %s unique bonds, %s unique date_m, from %s to %s."%(
    len(set(vdata_ibm.crsp_portno)),
    len(set(vdata_ibm.cusip8)),
    len(set(vdata_ibm.date_m)),
    str(min(vdata_ibm.date_m)),
    str(min(vdata_ibm.date_m))
     ))

print(vdata_ibq.columns.tolist())
print("Data has: %s unique funds, %s unique bonds, %s unique date_q, from %s to %s."%(
    len(set(vdata_ibq.crsp_portno)),
    len(set(vdata_ibq.cusip8)),
    len(set(vdata_ibq.date_q)),
    str(min(vdata_ibq.date_q)),
    str(min(vdata_ibq.date_q))
     ))

