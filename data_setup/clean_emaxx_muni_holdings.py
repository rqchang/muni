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


""" Read fund and bond data """
" WRDS monthly bond returns for prices, amounts outstanding, and credit ratings "
wbr = pd.read_csv('./munis/data/raw/WBR/bond_returns_master.csv', encoding='latin-1')
wbr['cusip8'] = wbr['cusip'].str[:8]
wbr['date_m'] = pd.to_datetime(wbr['date']).dt.to_period('M').astype(str)
print("Read bond-month level WRDS Bond returns data %s, from %s to %s."%(wbr.shape, str(wbr.date.min()), str(wbr.date.max())))

" Read Mergent muni bond data "
munis = pd.read_csv('./munis/data/raw/mergent_muni/bondinfo.csv', encoding='latin-1') # raw data from website
munis['cusip9'] = munis['cusip_c'].copy()
print("Read bond level municipal bond issuance data:", munis.shape)
print(munis.columns.to_list())


""" Read and clean Emaxx quarterly holdings data """
" Read holdings data that cusip is in munis "
chunksize = 1000000 
emaxx1_temp = []
emaxx_cols = ['fundid', 'cusip9', 'date_q', 'paramt', 'aum', 'fundclass', 'fundgroup']

for emaxx in pd.read_csv('./munis/data/processed/Emaxx/emaxx_clean.csv', usecols=emaxx_cols, chunksize=chunksize):
    # create relevant columns
    emaxx['date_q'] = emaxx['date_q'].str.replace(' ', '')
    emaxx['date_m'] = pd.to_datetime(emaxx['date_q']).dt.to_period('M').astype(str)
    emaxx['year'] = pd.to_datetime(emaxx['date_q']).dt.year
    emaxx['qtr'] = pd.to_datetime(emaxx['date_q']).dt.quarter
    emaxx['month'] = pd.to_datetime(emaxx['date_q']).dt.month
    emaxx['cusip8'] = emaxx['cusip9'].str[:8]

    # only keep if there is a cusip8 and is in the munis
    emaxx1 = emaxx.loc[emaxx['cusip9'].notnull()].copy()
    emaxx1 = emaxx1.loc[emaxx1['cusip9'].isin(munis['cusip9'])].copy()
    emaxx1_temp.append(emaxx1.copy())

vemaxx = pd.concat(emaxx1_temp)
print("Read Emaxx quarterly muni bond holdings:", vemaxx.shape)
print("Number of unique fundids that have munis:", len(set(vemaxx.fundid)))


""" Save data """
vemaxx.to_csv('./munis/data/processed/Emaxx/emaxx_muni_holdings_q.csv')

" Check "
print(vemaxx.columns.tolist())
print("Data has: %s unique funds, %s unique bonds, %s unique date_q, from %s to %s."%(
    len(set(vemaxx.fundid)),
    len(set(vemaxx.cusip9)),
    len(set(vemaxx.date_q)),
    str(min(vemaxx.date_q)),
    str(min(vemaxx.date_q))
     ))
print(vemaxx.count())