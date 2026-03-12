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
years   <- 2005:2025
results <- vector("list", length(years))
pb <- txtProgressBar(min = 0, max = length(years), style = 3)

logfile <- file.path(PROCDIR, "MSRB", "institutionalization_log.txt")
dir.create(dirname(logfile), recursive = TRUE, showWarnings = FALSE)

# overwrite each run
cat("", file = logfile)

log_line <- function(...) {
  msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste(..., collapse = " "))
  cat(msg, "\n", file = logfile, append = TRUE)
}

log_line("START institutionalization loop. Years:", paste(range(years), collapse = "-"))

for (i in seq_along(years)) {
  y <- years[i]
  f <- file.path(TEMPDIR, "MSRB", sprintf("msrb_clean_%d.rds", y))
  
  log_line("-----")
  log_line("Year", y, "- loading:", f)
  
  if (!file.exists(f)) {
    log_line("WARNING: file not found, skipping year", y)
    results[[i]] <- data.table(year = y)
    setTxtProgressBar(pb, i)
    next
  }
  
  dt <- readRDS(f)
  setDT(dt)
  n_before <- dt[, .N]
  
  # keep secondary market trades (drop primary takedowns)
  if ("offer_price_takedown_indicator" %in% names(dt)) {
    dt <- dt[!offer_price_takedown_indicator %chin% "Y"]
    log_line("Dropped primary takedown trades using offer_price_takedown_indicator.")
  } else {
    log_line("NOTE: offer_price_takedown_indicator not found; no primary-market filter applied.")
  }
  
  n_after <- dt[, .N]
  n_drop  <- n_before - n_after
  pct_drop_rows <- if (n_before > 0) 100 * n_drop / n_before else NA_real_
  log_line(
    "Rows before:", n_before,
    "| after:", n_after,
    "| dropped:", n_drop,
    sprintf("| pct_dropped: %.2f%%", pct_drop_rows)
  )
  
  # Determine odd lot and customer trades
  dt[, odd_lot        := par_traded <= 100000]
  dt[, customer_trade := trade_type_indicator %chin% c("P","S")]
  
  # Institutional signals (NA-safe via %chin%)
  dt[, inst_ats       := ats_indicator %chin% "Y"]
  dt[, inst_bb        := brokers_broker_indicator %chin% c("P","S")]
  dt[, inst_ntbc      := ntbc_indicator %chin% "Y"]
  dt[, inst_wap       := weighted_price_indicator %chin% "Y"]
  dt[, inst_channel   := inst_ats | inst_bb | inst_ntbc | inst_wap]
  
  # ---- Year-level summary ----
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
  
  # Par in billions
  par_total_bn      <- par_total / 1e9
  par_oddlot_bn     <- par_oddlot / 1e9
  par_cust_oddlot_bn<- par_cust_oddlot / 1e9
  
  results[[i]] <- data.table(
    year = y,
    n_trades = n_trades,
    n_oddlot = n_oddlot,
    n_cust_oddlot = n_cust_oddlot,
    
    par_total_bn = par_total_bn,
    par_oddlot_bn = par_oddlot_bn,
    par_cust_oddlot_bn = par_cust_oddlot_bn,
    
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
summary_year <- rbindlist(results, fill = TRUE)

# save down data
saveRDS(summary_year, paste0(PROCDIR, "MSRB/oddlot_sum_year.rds"))


# ================================================================= #
# Read data ####
# ================================================================= #
summary_year <- readRDS(paste0(PROCDIR, "MSRB/oddlot_sum_year.rds"))
summary_year[, share_par_oddlot := par_oddlot_bn / par_total_bn]
summary_year[, share_count_oddlot := n_oddlot / n_trades]

k <- max(summary_year$share_count_oddlot, na.rm = TRUE) /
  max(summary_year$share_par_oddlot, na.rm = TRUE)

ggplot(summary_year, aes(x = year)) +
  geom_line(
    aes(y = share_par_oddlot, color = "Par (volume)"),
    linewidth = 1.2
  ) +
  geom_point(
    aes(y = share_par_oddlot, color = "Par (volume)"),
    size = 2
  ) +
  geom_line(
    aes(y = share_count_oddlot / k, color = "Count (trades)"),
    linewidth = 1.2
  ) +
  geom_point(
    aes(y = share_count_oddlot / k, color = "Count (trades)"),
    size = 2
  ) +
  scale_y_continuous(
    labels = scales::percent,
    sec.axis = sec_axis(
      ~ . * k,
      name = "Count (trades) share",
      labels = scales::percent
    )
  ) +
  scale_color_manual(
    values = c("Par (volume)" = "#1f77b4", "Count (trades)" = "#ff7f0e")
  ) +
  labs(
    x = "Year",
    y = "Par (volume) share",
    color = "Measure",
    title = "Odd-Lot Share of Municipal Bond Trades"
  ) +
  theme_minimal()

plot_dt <- melt(
  summary_year,
  id.vars = "year",
  measure.vars = c(
    "inst_all_count",
    "inst_all_vol"
  ),
  variable.name = "measure",
  value.name = "value"
)

ggplot(plot_dt, aes(x = year, y = value, color = measure)) +
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



