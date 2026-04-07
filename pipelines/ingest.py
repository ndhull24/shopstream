import pandas as pd
import sqlite3
import duckdb
import os

RAW_DIR = "data/raw"
DB_PATH = "data/shopstream.db"
DUCK_PATH = "data/shopstream.duckdb"

print("Phase 2 — ingestion starting...")

# ── 1. Load all CSVs ─────────────────────────────────────────
print("\n  Loading CSVs...")
df_customers   = pd.read_csv(f"{RAW_DIR}/customers.csv")
df_products    = pd.read_csv(f"{RAW_DIR}/products.csv")
df_orders      = pd.read_csv(f"{RAW_DIR}/orders.csv")
df_order_items = pd.read_csv(f"{RAW_DIR}/order_items.csv")
df_events      = pd.read_csv(f"{RAW_DIR}/events.csv")

# ── 2. Basic type casting ────────────────────────────────────
print("  Casting types...")
df_customers["signup_date"] = pd.to_datetime(df_customers["signup_date"])
df_orders["order_date"]     = pd.to_datetime(df_orders["order_date"])
df_events["event_ts"]       = pd.to_datetime(df_events["event_ts"])
df_products["is_active"]    = df_products["is_active"].astype(bool)

# ── 3. Load into SQLite ──────────────────────────────────────
print(f"\n  Writing to SQLite → {DB_PATH}")
conn = sqlite3.connect(DB_PATH)

df_customers.to_sql("raw_customers",   conn, if_exists="replace", index=False)
df_products.to_sql("raw_products",     conn, if_exists="replace", index=False)
df_orders.to_sql("raw_orders",         conn, if_exists="replace", index=False)
df_order_items.to_sql("raw_order_items", conn, if_exists="replace", index=False)
df_events.to_sql("raw_events",         conn, if_exists="replace", index=False)

# ── 4. Quick SQL validation queries ─────────────────────────
print("\n  Running validation queries...")
cur = conn.cursor()

checks = [
    ("Total customers",       "SELECT COUNT(*) FROM raw_customers"),
    ("Duplicate customers",   "SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM raw_customers"),
    ("Missing emails",        "SELECT COUNT(*) FROM raw_customers WHERE email IS NULL"),
    ("Total orders",          "SELECT COUNT(*) FROM raw_orders"),
    ("Orders by status",      "SELECT status, COUNT(*) FROM raw_orders GROUP BY status"),
    ("Total order items",     "SELECT COUNT(*) FROM raw_order_items"),
    ("Missing shipping city", "SELECT COUNT(*) FROM raw_orders WHERE shipping_city IS NULL"),
    ("Total events",          "SELECT COUNT(*) FROM raw_events"),
    ("Anonymous events",      "SELECT COUNT(*) FROM raw_events WHERE customer_id IS NULL"),
    ("Revenue (completed)",   """
        SELECT ROUND(SUM(order_total), 2)
        FROM raw_orders WHERE status = 'completed'
    """),
]

for label, query in checks:
    cur.execute(query)
    result = cur.fetchall()
    print(f"    {label:<25}: {result}")

conn.close()

# ── 5. Load into DuckDB ──────────────────────────────────────
print(f"\n  Writing to DuckDB → {DUCK_PATH}")
duck = duckdb.connect(DUCK_PATH)

duck.execute("""
    CREATE OR REPLACE TABLE raw_customers AS
    SELECT * FROM read_csv_auto('data/raw/customers.csv')
""")
duck.execute("""
    CREATE OR REPLACE TABLE raw_products AS
    SELECT * FROM read_csv_auto('data/raw/products.csv')
""")
duck.execute("""
    CREATE OR REPLACE TABLE raw_orders AS
    SELECT * FROM read_csv_auto('data/raw/orders.csv')
""")
duck.execute("""
    CREATE OR REPLACE TABLE raw_order_items AS
    SELECT * FROM read_csv_auto('data/raw/order_items.csv')
""")
duck.execute("""
    CREATE OR REPLACE TABLE raw_events AS
    SELECT * FROM read_csv_auto('data/raw/events.csv')
""")

# ── 6. Run an analytical query in DuckDB ─────────────────────
print("\n  DuckDB — revenue by product category:")
result = duck.execute("""
    SELECT
        p.category,
        COUNT(DISTINCT o.order_id)          AS num_orders,
        ROUND(SUM(oi.line_total), 2)        AS total_revenue,
        ROUND(AVG(oi.line_total), 2)        AS avg_line_value
    FROM raw_order_items oi
    JOIN raw_products p  ON oi.product_id = p.product_id
    JOIN raw_orders   o  ON oi.order_id   = o.order_id
    WHERE o.status = 'completed'
    GROUP BY p.category
    ORDER BY total_revenue DESC
""").df()

print(result.to_string(index=False))

duck.close()

print("\n Phase 2 complete!")
print(f"   SQLite  → {DB_PATH}")
print(f"   DuckDB  → {DUCK_PATH}")