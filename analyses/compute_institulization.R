# ================================================================= #
# compute_institulization.R ####
# ================================================================= #
# define odd lot as par traded <= 100000

# define customer trade
#Type of trade: an inter-dealer trade (D), a purchase from a customer by a dealer (P) 
# or a sale to acustomer by a dealer (S). Format: one character.

# ATS trades: used by institutional traders and SMAs
# An indicator (Y) showing that an inter-dealer transaction was executed with or 
# using the services of an alternative trading system (ATS) with Form ATS on file with the SEC.

# Broker's broker: used by dealers sourcing liquidity for institutions
# An indicator used in inter-dealer transactions that were executed by a broker's broker, 
# including whether it was a purchase (P) or sale (S) by the broker's broker.

# NTBC trades: institutional trades are typically principal trades and no explicit markup reported
# An indicator (Y) showing that a customer transaction did not include 
# a mark-up, mark -down or commission.

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
library(ggplot2)

# Source helper scripts
source('utils/setPaths.R')


# ================================================================= #
# Compute institulization ####
# ================================================================= #
years   <- 2005:2024
results <- vector("list", length(years))
pb <- txtProgressBar(min = 0, max = length(years), style = 3)

for (i in seq_along(years)) {
  y <- years[i]
  dt <- readRDS(file.path(TEMPDIR, "MSRB", sprintf("msrb_clean_%d.rds", y)))
  setDT(dt)
  
  # Determine odd lot and customer trades
  dt[, odd_lot        := par_traded <= 100000]
  dt[, customer_trade := trade_type_indicator %chin% c("P","S")]
  
  # Institutional signals
  dt[, inst_ats     := ats_indicator %chin% "Y"]
  dt[, inst_bb      := brokers_broker_indicator %chin% c("P","S")]
  dt[, inst_ntbc    := ntbc_indicator %chin% "Y"]
  dt[, inst_channel := inst_ats | inst_bb | inst_ntbc]
  
  # ---- Year-level summary ----
  # All trades
  n_trades  <- dt[, .N]
  par_total <- dt[, sum(par_traded, na.rm = TRUE)]
  
  n_oddlot   <- dt[odd_lot == TRUE, .N]
  par_oddlot <- dt[odd_lot == TRUE, sum(par_traded, na.rm = TRUE)]
  
  inst_all_count <- dt[
    odd_lot == TRUE,
    if (.N > 0) mean(inst_channel, na.rm = TRUE) else NA_real_
  ]
  
  inst_all_vol <- dt[
    odd_lot == TRUE,
    {
      denom <- sum(par_traded, na.rm = TRUE)
      if (denom > 0) sum(par_traded * as.integer(inst_channel), na.rm = TRUE) / denom else NA_real_
    }
  ]
  
  # Customer trades
  n_cust_oddlot   <- dt[odd_lot == TRUE & customer_trade == TRUE, .N]
  par_cust_oddlot <- dt[odd_lot == TRUE & customer_trade == TRUE, sum(par_traded, na.rm = TRUE)]
  
  inst_cust_count <- dt[
    odd_lot == TRUE & customer_trade == TRUE,
    if (.N > 0) mean(inst_channel, na.rm = TRUE) else NA_real_
  ]
  
  inst_cust_vol <- dt[
    odd_lot == TRUE & customer_trade == TRUE,
    {
      denom <- sum(par_traded, na.rm = TRUE)
      if (denom > 0) sum(par_traded * as.integer(inst_channel), na.rm = TRUE) / denom else NA_real_
    }
  ]
  
  results[[i]] <- data.table(
    year = y,
    n_trades = n_trades,
    n_oddlot = n_oddlot,
    n_cust_oddlot = n_cust_oddlot,
    par_total = par_total,
    par_oddlot = par_oddlot,
    par_cust_oddlot = par_cust_oddlot,
    inst_all_count = inst_all_count,
    inst_cust_count = inst_cust_count,
    inst_all_vol = inst_all_vol,
    inst_cust_vol = inst_cust_vol
  )
  
  rm(dt)
  gc()
  setTxtProgressBar(pb, i)
}

close(pb)

summary_year <- rbindlist(results)
summary_year[]

# save down data
saveRDS(summary_year, paste0(PROCDIR, "MSRB/oddlot_sum_year.rds"))

# ================================================================= #
# Read data ####
# ================================================================= #
plot_dt <- melt(
  summary_year,
  id.vars = "year",
  measure.vars = c(
    "inst_all_count",
    "inst_cust_count",
    "inst_all_vol",
    "inst_cust_vol"
  ),
  variable.name = "measure",
  value.name = "value"
)

ggplot(plot_dt[year >= 2016], aes(x = year, y = value, color = measure)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    x = "Year",
    y = "Institutionalization share",
    color = "Measure",
    title = "Institutionalization of Odd-Lot Municipal Bond Trades"
  ) +
  theme_minimal()



