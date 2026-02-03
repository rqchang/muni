# ================================================================= #
# import_msrb.R ####
# ================================================================= #
# Description:
# ------------
#   This file downloads and saves MSRB muni bonds transaction data from 2005 till now.
#   It uses RPostgres direct download.
#
# Output(s):
# ------------
#   Dropbox data/raw/MSRB/:
#     msrb_year.rds
# 
# Date:
# -----
#   2026-02-03
#   update: 
#
# Author(s):
# -------
#   Ruiquan Chang, chang.2590@osu.edu
#
# Additional note(s):
# ----------
#   https://wrds-www.wharton.upenn.edu/data-dictionary/msrb_all/msrb/
#
# ================================================================= #


# ================================================================= #
# Environment ####
# ================================================================= #
# Clear workspace
rm(list = ls())

# Import libraries
library(data.table)
library(RPostgres)
library(zoo)
library(lubridate)
library(DBI)

# Source helper scripts
source('utils/setPaths.R')
source('utils/wrds_credentials.R')

# Create database connections
creds <- get_wrds_credentials()
wrds <- dbConnect(Postgres(),
                  host='wrds-pgdata.wharton.upenn.edu',
                  port=9737,
                  user = creds$username,
                  password = creds$password,
                  sslmode='require',
                  dbname='wrds')


# ================================================================= #
# Import data ####
# ================================================================= #
# Check columns
query <- "
  SELECT column_name,
         data_type,
         is_nullable
  FROM information_schema.columns
  WHERE table_schema = 'msrb'
    AND table_name   = 'msrb'
  ORDER BY ordinal_position
"
cols <- as.data.table(dbGetQuery(wrds, query))

# for each year, save down to dropbox
years <- 2005:2025
t0_all <- Sys.time()

for (i in seq_along(years)) {
  year <- years[i]
  t0 <- Sys.time()
  
  message(sprintf(
    "[%02d/%02d] %d | START  (%s)",
    i, length(years), year, format(t0, "%H:%M:%S")
  ))
  
  query <- sprintf("
    SELECT * FROM msrb.msrb
    WHERE EXTRACT(YEAR FROM trade_date::date) = %d
  ", year)
  
  # save down data
  dt <- as.data.table(dbGetQuery(wrds, query))
  saveRDS(dt, sprintf(paste0(RAWDIR, "MSRB/msrb_%d.rds"), year))
  
  message(sprintf(
    "[%02d/%02d] %d | DONE   rows=%s | %.1f sec",
    i, length(years), year,
    format(nrow(dt), big.mark = ","),
    as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ))
}

message(sprintf(
  "ALL DONE | total elapsed = %.1f minutes",
  as.numeric(difftime(Sys.time(), t0_all, units = "mins"))
))

