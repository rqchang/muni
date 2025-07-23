import os
import pandas as pd
import numpy as np
import datetime
import pyodbc
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.4f}'.format

os.chdir(r'/nfs/sloanlab007/projects/emaxx_rq_proj')


#<codecell>
""" Read fund and bond data """
" Mergent muni bond data "
munis = pd.read_csv('./munis/data/raw/mergent_muni/bondinfo.csv', encoding='latin-1')
munis['cusip'] = munis['cusip_c'].str[:9]
print("Read bond level municipal bond issuance data:", munis.shape)
print(munis.columns.to_list())

" Mergent muni bond rating information "
# data is at cusip9-call_date level
rating = pd.read_csv('./munis/data/rating_merged.csv', encoding='latin-1')
print("Read munis rating information data:", rating.shape)

# aggregate to cusip9 level
f = {'state':'last', 'tax_code':'last', 'debt_type':'last', 'capital_purpose':'last', 
     'state_tax':'last', 'yield_treas_final':'last', 'grade':'last','grade_LT':'last', 
     'grade_ST':'last', 'grade_ST_ENH':'last', 'grade_LT_ENH':'last', 'invgrade':'last'}
vrating = rating.groupby(['cusip'], as_index=False).agg(f)
print("After aggregating rating information to bond level:", vrating.shape)


#<codecell>
""" Read and clean NAIC Insurance quarterly holdings data """
" Read holdings data that cusip is in munis "
chunksize = 1000000 
naic1_temp = []

for naic in pd.read_csv('./munis/data/raw/naic_clean.csv', encoding='latin-1', chunksize=chunksize):
    # create relevant columns
    naic['date_q'] = pd.to_datetime(naic['date_q'].str.replace(' ', '')).dt.to_period('Q').astype(str)
    naic['date_m'] = pd.to_datetime(naic['date_q']).dt.to_period('M').astype(str)
    naic['cusip8'] = naic['cusip'].copy()
    
    naic['year'] = pd.to_datetime(naic['date_q']).dt.year
    naic['qtr'] = pd.to_datetime(naic['date_q']).dt.quarter
    naic['month'] = pd.to_datetime(naic['date_q']).dt.month

    # only keep if there is a cusip and is in the munis
    naic1 = naic.loc[naic['cusip'].notnull()].copy()
    naic1 = naic1.loc[naic1['cusip'].isin(munis['cusip'])].copy()
    naic1_temp.append(naic1.copy())

vnaic = pd.concat(naic1_temp)
print("Read NAIC Insurance muni bond holdings:", vnaic.shape)
print("Number of unique funds that hold munis:", len(set(vnaic.entity_key)))


#<codecell>
""" Merge with bond characteristics """
" Add muni bond issuance "
vdata = pd.merge(vnaic.copy(), munis.copy(), how='left', on=['cusip'], suffixes=['', '_FISD'])
print("After adding FISD muni bond issuance data:", vdata.shape)

" Add muni bond rating "
vdata = pd.merge(vdata.copy(), vrating.copy(), how='left', on=['cusip'], suffixes=['', '_FISD'])
print("After adding FISD muni rating information:", vdata.shape)

" Check that data is at entity_key-cusip9-quarter level "
vdata = vdata.sort_values(by=['entity_key', 'cusip', 'date_q'])
print("Check that data is at entity_key-cusip9-quarter level:",
      len(vdata)==len(vdata.drop_duplicates(subset=['entity_key', 'cusip', 'date_q'])))

" Subsample P&C insurers "
vpc = vdata.loc[vdata['naic_type']=='P&C'].copy()
print("After subsampling P&C insurers:", vpc.shape)


#<codecell>
""" Save data """
vdata.sample(n=1000000).to_csv('./munis/data/processed/NAIC/naic_muni_holdings_q_small.csv')
vdata.to_csv('./munis/data/processed/NAIC/naic_muni_holdings_q.csv')
print("Saved fund-muni-quarter level data:", vnaic.shape)

vpc.to_csv('./munis/data/processed/NAIC/naic_pc_muni_holdings_q.csv')
print("Saved fund-muni-quarter level P&C data:", vpc.shape)

" Check "
print(vdata.columns.tolist())
print("Full NAIC muni holdings data has: %s unique funds, %s unique bonds, %s unique date_q, from %s to %s."%(
    len(set(vdata.entity_key)),
    len(set(vdata.cusip)),
    len(set(vdata.date_q)),
    str(min(vdata.date_q)),
    str(max(vdata.date_q))
     ))
print(vdata.count())
print(vdata.groupby(['date_q'], as_index=False).agg({'entity_key':'count'}))

print("P&C muni holdings data has: %s unique funds, %s unique bonds, %s unique date_q, from %s to %s."%(
    len(set(vpc.entity_key)),
    len(set(vpc.cusip)),
    len(set(vpc.date_q)),
    str(min(vpc.date_q)),
    str(max(vpc.date_q))
     ))
print(vpc.count())
print(vpc.groupby(['date_q'], as_index=False).agg({'entity_key':'count'}))

