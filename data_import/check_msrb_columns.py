"""
check_msrb_columns.py
Connect to WRDS PostgreSQL and list all columns in msrb.msrb.
Checks specifically for reporting_dealer_id and other dealer identifiers.
"""

import os
import psycopg2
import pandas as pd

# ── Connect ───────────────────────────────────────────────────────────────────
conn = psycopg2.connect(
    host="wrds-pgdata.wharton.upenn.edu",
    port=9737,
    dbname="wrds",
    user="rqchang99",
    password="Crq-19990711",
    sslmode="require",
)

# ── Query all columns in msrb.msrb ────────────────────────────────────────────
query = """
    SELECT column_name,
           data_type,
           is_nullable,
           ordinal_position
    FROM information_schema.columns
    WHERE table_schema = 'msrb'
      AND table_name   = 'msrb'
    ORDER BY ordinal_position
"""

cols = pd.read_sql(query, conn)
conn.close()

print(f"\nTotal columns: {len(cols)}\n")
print(cols[["ordinal_position", "column_name", "data_type"]].to_string(index=False))

# ── Check for dealer identifiers ──────────────────────────────────────────────
dealer_cols = cols[cols["column_name"].str.contains("dealer|mpid|reporter|firm", case=False)]
print(f"\n--- Dealer-related columns ---")
if dealer_cols.empty:
    print("None found.")
else:
    print(dealer_cols[["column_name", "data_type"]].to_string(index=False))

has_dealer_id = "reporting_dealer_id" in cols["column_name"].values
print(f"\nhas reporting_dealer_id: {has_dealer_id}")
