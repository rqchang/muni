# ================================================================= #
# clean_msrb.R ####
# ================================================================= #
# Description:
# ------------
#   This file cleans the MSRB transaction data from 2005 till now.
#   Conditional on:
#   (1) cusips with >= 3 trades across 2005 till 2025
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
library(zoo)
library(lubridate)
library(timeDate)
library(ggplot2)

# Source helper scripts
source('utils/setPaths.R')

# function
get_business_days <- function(year) {
  start_date <- as.Date(sprintf("%d-01-01", year))
  end_date   <- as.Date(sprintf("%d-12-31", year))
  
  all_days <- seq.Date(start_date, end_date, by = "day")
  
  # weekdays only
  is_weekday <- !(weekdays(all_days) %in% c("Saturday", "Sunday"))
  
  # NYSE holidays
  nyse_holidays <- as.Date(holidayNYSE(year))
  
  # exclude holidays
  business_days <- all_days[is_weekday & !(all_days %in% nyse_holidays)]
  
  data.table(trade_date = as.IDate(business_days))
}


# ================================================================= #
# Clean data ####
# ================================================================= #
# ------------------------------------------------- #
# List of CUSIPs with <= 3 trades across 2005-2025
# ------------------------------------------------- #
years <- 2005:2025
total_counts <- data.table(cusip = character(), counts = integer())
nyears <- length(years)

for (i in seq_along(years)) {
  year <- years[i]
  message(sprintf("[%02d/%02d] Processing %d ...", i, nyears, year))
  
  # read data for each year
  dt <- readRDS(sprintf(paste0(RAWDIR, "MSRB/msrb_%d.rds"), year))
  if (!"cusip" %in% names(dt)) {
    stop(sprintf("Missing column 'cusip' in MSRB/msrb_%d.rds", year))
  }
  
  message(sprintf(
    "        Rows: %s | Unique CUSIPs (year): %s",
    format(nrow(dt), big.mark = ","),
    format(uniqueN(dt$cusip), big.mark = ",")
  ))
  
  # count trades in THIS file
  cty <- dt[, .(counts_y = .N), by = cusip]
  
  # accumulate to total
  setkey(cty, cusip)
  setkey(total_counts, cusip)
  
  if (i == 1) {
    total_counts <- cty[, .(cusip, counts = as.integer(counts_y))]
  } else {
    # outer join total_counts with cty to get counts_y
    total_counts <- merge(total_counts, cty, by  = "cusip", all = TRUE)
    
    total_counts[, counts :=
                   fifelse(is.na(counts),   0L, counts) +
                   fifelse(is.na(counts_y), 0L, counts_y)
    ]
    total_counts[, counts_y := NULL]
  }
  
  message(sprintf(
    "        Cumulative unique CUSIPs so far: %s\n",
    format(nrow(total_counts), big.mark = ",")
  ))
}

t_lowtrade <- total_counts[counts < 3]
cusipno <- nrow(t_lowtrade)

# save down low trade cusips
message(sprintf("Total # CUSIPs with < 3 trades across 2005-2025: %d", cusipno))
saveRDS(t_lowtrade, paste0(TEMPDIR, "MSRB/cusip_less_than_3.rds"))

# for fast anti-join later:
lowtrade_set <- t_lowtrade[, unique(cusip)]

# ------------------------------------------------- #
# Per-year cleaning
# ------------------------------------------------- #
# log file
logfile <- file.path(TEMPDIR, "msrb_cleaning_log.txt")
cat("", file = logfile)  # overwrite each run

# open sinks (append=FALSE to overwrite; TRUE to append)
zz_out <- file(logfile, open = "wt")
zz_msg <- file(logfile, open = "at")  # keep messages appended neatly

sink(zz_out, type = "output")
sink(zz_msg, type = "message")

on.exit({
  sink(type = "message")
  sink(type = "output")
  close(zz_msg)
  close(zz_out)
}, add = TRUE)

# run the loop for each year
data_summary_list <- vector("list", length(years))
for (k in seq_along(years)) {
  year <- years[k]
  dt <- readRDS(sprintf(paste0(RAWDIR, "MSRB/msrb_%d.rds"), year))
  message(sprintf("===== [%02d/%02d] Processing %d ... =====",k, nyears, year))
  
  original_rows <- nrow(dt)
  starting_rows <- original_rows
  message(sprintf("Read row data rows: %d", original_rows))
  
  ## --- Step 1.1: business days + non-holidays ---
  # Convert trade_date
  if (!"trade_date" %in% names(dt)) stop(sprintf("Missing 'trade_date' in msrb_%d.rds", year))
  dt[, trade_date := as.IDate(trade_date)]  # expects YYYY-mm-dd; adjust if needed
  bd <- get_business_days(year)
  message(sprintf("Business days in %d (ex-holidays): %d", year, nrow(bd)))
  
  setkey(bd, trade_date)
  setkey(dt, trade_date)
  
  dt <- dt[bd, nomatch = 0L]  # inner join
  new_rows <- nrow(dt)
  message(sprintf("Business-day restriction drops %d rows. Retain %.2f%% rows.",
                  starting_rows - new_rows, 100 * new_rows / original_rows))
  starting_rows <- new_rows
  
  ## --- Step 1.2: compute time-to-maturity ---
  if (!"maturity_date" %in% names(dt)) stop(sprintf("Missing 'maturity_date' in msrb_%d.rds", year))
  dt[, maturity_date := as.IDate(maturity_date)]  # invalid -> NA
  
  dt <- dt[!is.na(maturity_date) & !is.na(trade_date)]
  dt[, ttm_days := as.integer(maturity_date - trade_date)]
  dt[, ttm := ttm_days / 365.25]
  
  # drop maturities < 0 and > 100 years
  dt <- dt[ttm_days >= 0L & ttm_days <= 36525L]
  new_rows <- nrow(dt)
  message(sprintf("Valid time-to-maturity restriction drops %d rows. Retain %.2f%% rows.",
                  starting_rows - new_rows, 100 * new_rows / original_rows))
  starting_rows <- new_rows
  
  ## --- Step 1.3: numeric cols + drop NA ---
  need_num <- c("dollar_price", "coupon", "yield", "par_traded")
  miss <- setdiff(need_num, names(dt))
  if (length(miss)) stop(sprintf("Missing numeric cols in msrb_%d.rds: %s", year, paste(miss, collapse = ", ")))
  
  for (v in need_num) {
    suppressWarnings(dt[, (v) := as.numeric(get(v))])
  }
  dt <- dt[!is.na(dollar_price) & !is.na(coupon) & !is.na(yield) & !is.na(par_traded)]
  
  new_rows <- nrow(dt)
  message(sprintf("Valid price/coupon/yield/par restriction drops %d rows. Retain %.2f%% rows.",
                  starting_rows - new_rows, 100 * new_rows / original_rows))
  starting_rows <- new_rows
  
  ## --- Step 1.4: drop coupons in excess of 20%, and price < 50 and > 150 ---
  dt <- dt[coupon <= 20]
  new_rows <- nrow(dt)
  message(sprintf("Coupon restriction drops %d rows. Retain %.2f%% rows.",
                  starting_rows - new_rows, 100 * new_rows / original_rows))
  starting_rows <- new_rows
  
  dt <- dt[dollar_price >= 50 & dollar_price <= 150]
  new_rows <- nrow(dt)
  message(sprintf("Price restriction drops %d rows. Retain %.2f%% rows. Remaining rows: %d",
                  starting_rows - new_rows, 100 * new_rows / original_rows, new_rows))
  starting_rows <- new_rows
  
  ## --- Step 1.5: drop bonds with < 3 trades across full sample ---
  # keep cusips NOT in lowtrade_set.
  dt <- dt[!(cusip %chin% lowtrade_set)]
  new_rows <- nrow(dt)
  message(sprintf("Low-trade CUSIP filter drops %d rows. Retain %.2f%% rows. Remaining rows: %d",
                  starting_rows - new_rows, 100 * new_rows / original_rows, new_rows))
  starting_rows <- new_rows
  
  ## --- Step 2: compute weighted price/yield ---
  dt[, `:=`(
    weighted_price = dollar_price * par_traded,
    weighted_yield = yield        * par_traded
  )]
  
  # trade_month like "200501"
  dt[, trade_year := year(as.Date(trade_date))]
  dt[, yyyymm := trade_year * 100 + month(as.Date(trade_date))]
  
  # aggregate to month level
  dt_month <- dt[, {
    pt <- sum(par_traded, na.rm = TRUE)
    .(
      prc = if (pt > 0) sum(weighted_price, na.rm = TRUE) / pt else NA_real_,
      yield = if (pt > 0) sum(weighted_yield, na.rm = TRUE) / pt else NA_real_,
      paramt = pt
    )
  }, by = .(yyyymm, cusip)]
  
  # aggregate to day level
  dt_day <- dt[, {
    pt <- sum(par_traded, na.rm = TRUE)
    .(
      prc = if (pt > 0) sum(weighted_price, na.rm = TRUE) / pt else NA_real_,
      yield = if (pt > 0) sum(weighted_yield, na.rm = TRUE) / pt else NA_real_,
      paramt = pt
    )
  }, by = .(trade_date, cusip)]
  
  # save price data
  saveRDS(dt_month, sprintf(paste0(TEMPDIR, "MSRB/msrb_%d_month.rds"), year))
  saveRDS(dt_day, sprintf(paste0(TEMPDIR, "MSRB/msrb_%d_day.rds"), year))
  message("Day and month levels price data created.")
  
  ## --- Step 3: merge prices back ---
  dt <- merge(dt, dt_month[, .(yyyymm, cusip, prc_m = prc, yield_m = yield, paramt_m = paramt)],
              by = c('yyyymm', 'cusip'), all.x = TRUE)
  dt <- merge(dt, dt_day[, .(trade_date, cusip, prc_d = prc, yield_d = yield, paramt_d = paramt)],
              by = c('trade_date', 'cusip'), all.x = TRUE)
  
  ## --- Step 4: save cleaned file for each year ---
  saveRDS(dt, sprintf(paste0(TEMPDIR, "MSRB/msrb_clean_%d.rds"), year))
  
  ## --- Step 5: summary vars ---
  year_data <- dt[, .(
    year = trade_year[1L],
    num_cusip        = uniqueN(cusip),
    num_trade        = .N,
    num_trade_retail = sum(par_traded <= 100000, na.rm = TRUE),
    
    totparamt = sum(par_traded, na.rm = TRUE) / 1e9,
    paramt_D  = sum(par_traded[trade_type_indicator == "D"], na.rm = TRUE) / 1e9,
    paramt_S  = sum(par_traded[trade_type_indicator == "S"], na.rm = TRUE) / 1e9,
    paramt_P  = sum(par_traded[trade_type_indicator == "P"], na.rm = TRUE) / 1e9,
    
    paramt_ATS = sum(par_traded[ats_indicator == "Y"], na.rm = TRUE) / 1e9,
    paramt_BBS = sum(par_traded[brokers_broker_indicator == "S"], na.rm = TRUE) / 1e9,
    paramt_BBP = sum(par_traded[brokers_broker_indicator == "P"], na.rm = TRUE) / 1e9
  )]
  
  data_summary_list[[k]] <- year_data
  message(sprintf("%d processed and saved.", year))
}

sum <- rbindlist(data_summary_list, use.names = TRUE, fill = TRUE)
saveRDS(sum, paste0(PROCDIR, "MSRB/trade_sum_year.rds"))


# ================================================================= #
# Plots ####
# ================================================================= #
# ------------------------------------------------- #
# Share of retail trade
# ------------------------------------------------- #
sum <- sum[year <= 2024]
sum[, share_retail := num_trade_retail / num_trade]

p1 <- ggplot(sum, aes(x = year, y = share_retail)) +
  geom_line(linewidth = 1, color = "steelblue") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_x_continuous(
    breaks = seq(min(sum$year, na.rm = TRUE),
                 max(sum$year, na.rm = TRUE),
                 by = 2)
  ) +
  labs(
    title = NULL,
    x = "Year",
    y = "Retail share (by trade count)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    axis.line.x = element_line(color = "black", linewidth = 0.5),
    axis.line.y = element_line(color = "black", linewidth = 0.5),
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(3, "pt"),
    panel.grid.major = element_line(color = "grey85"),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(OUTDIR, "plots/trade/share_retail_y.pdf"),
  plot     = p1,
  width    = 7,
  height   = 4.5
)

# ------------------------------------------------- #
# Share of P vs S vs D
# ------------------------------------------------- #
sum[, `:=`(
  share_P = paramt_P / totparamt,
  share_S = paramt_S / totparamt,
  share_D = paramt_D / totparamt
)]
sum[, max(share_P + share_S + share_D, na.rm = TRUE)]
sum[, sum_share := share_P + share_S + share_D]

plot_dt <- melt(
  sum,
  id.vars = "year",
  measure.vars = c("share_P", "share_S", "share_D"),
  variable.name = "type",
  value.name = "share"
)

plot_dt[, type := factor(
  type,
  levels = c("share_D", "share_S", "share_P"),
  labels = c("Inter-dealer ", "Sale to a customer by a dealer ", "Purchase from a customer by a dealer")
)]

p2 <- ggplot(plot_dt, aes(x = year, y = share, color = type)) +
  geom_line(linewidth = 1) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_x_continuous(
    breaks = seq(min(plot_dt$year, na.rm = TRUE),
                 max(plot_dt$year, na.rm = TRUE),
                 by = 2)
  ) +
  labs(
    title = NULL,
    x = "Year",
    y = "Share of total par traded",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    axis.line.x      = element_line(color = "black", linewidth = 0.5),
    axis.line.y      = element_line(color = "black", linewidth = 0.5),
    axis.ticks       = element_line(color = "black"),
    panel.grid.major = element_line(color = "grey85"),
    panel.grid.minor = element_blank(),
    legend.position  = "top"
  )

ggsave(
  filename = file.path(OUTDIR, "plots/trade/share_trade_type_y.pdf"),
  plot     = p2,
  width    = 7,
  height   = 4.5
)

# ------------------------------------------------- #
# Number of trade and TOTPARAMT
# ------------------------------------------------- #
scale_factor <- max(sum$num_trade, na.rm = TRUE) / 
  max(sum$totparamt,  na.rm = TRUE)

p3 <- ggplot(sum, aes(x = year)) +
  geom_line(
    aes(y = num_trade / 1e3, color = "Number of trades"),
    linewidth = 1.2
  ) +
  geom_line(
    aes(y = totparamt * scale_factor / 1e3, color = "Total par traded"),
    linewidth = 1.2,
    linetype = "dashed"
  ) +
  scale_y_continuous(
    name = "Number of trades (thousands)",
    labels = scales::comma,
    sec.axis = sec_axis(
      ~ . / scale_factor * 1e3,
      name = "Total par traded (USD, billions)"
    )
  ) +
  scale_x_continuous(
    breaks = seq(min(sum$year), max(sum$year), by = 2)
  ) +
  scale_color_manual(
    values = c(
      "Number of trades" = "steelblue",
      "Total par traded" = "darkred"
    )
  ) +
  labs(
    x = "Year",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    axis.line.x      = element_line(color = "black", linewidth = 0.6),
    axis.line.y      = element_line(color = "black", linewidth = 0.6),
    axis.ticks       = element_line(color = "black"),
    panel.grid.major = element_line(color = "grey85"),
    panel.grid.minor = element_blank(),
    legend.position  = "top"
  )

ggsave(
  filename = file.path(OUTDIR, "plots/trade/totparamt_y.pdf"),
  plot     = p3,
  width    = 7,
  height   = 4.5
)

