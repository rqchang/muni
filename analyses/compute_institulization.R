# ================================================================= #
# compute_institulization.R ####
# ================================================================= #


# ================================================================= #
# Environment ####
# ================================================================= #
# Clear workspace
rm(list = ls())

# Import libraries
library(data.table)
library(parallel)
library(zoo)
library(lubridate)

# Source helper scripts
source('utils/setPaths.R')


# ================================================================= #
# Read data ####
# ================================================================= #
# issuer-month level amount outstanding
base <- fread(paste0(DIR, '/data_old/raw/bondinfo.csv')) |> as.data.table()
issue_data <- readRDS('D:/Dropbox/project/fund_contagion/data/raw/fisd/fisd_issue.rds')
