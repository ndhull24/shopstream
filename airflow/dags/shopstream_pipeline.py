from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import os
import sys

default_args = {
    "owner": "navdeep",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

def run_ingestion():
    result = subprocess.run(
        ["python", "/opt/airflow/pipelines/ingest.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Ingestion failed:\n{result.stderr}")

def run_transformation():
    import pandas as pd
    import duckdb
    import os

    duck = duckdb.connect("/opt/airflow/data/shopstream.duckdb")
    os.makedirs("/opt/airflow/data/warehouse", exist_ok=True)

    # dim_customers — deduplicated and cleaned
    duck.execute("""
        CREATE OR REPLACE TABLE dim_customers AS
        WITH deduped AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY customer_id
                    ORDER BY signup_date DESC
                ) AS rn
            FROM raw_customers
        )
        SELECT
            customer_id,
            first_name || ' ' || last_name     AS full_name,
            first_name, last_name,
            COALESCE(email, 'unknown@shopstream.com') AS email,
            city, state, country, segment,
            CAST(signup_date AS DATE)           AS signup_date,
            COALESCE(CAST(age AS INTEGER), 0)   AS age
        FROM deduped WHERE rn = 1 AND customer_id IS NOT NULL
    """)

    # dim_products — with margin and price tier
    duck.execute("""
        CREATE OR REPLACE TABLE dim_products AS
        SELECT
            product_id, product_name, category,
            price, cost,
            ROUND((price - cost) / price, 4)    AS margin,
            stock_qty,
            CAST(is_active AS BOOLEAN)           AS is_active,
            CASE
                WHEN price < 20  THEN 'budget'
                WHEN price < 100 THEN 'mid'
                WHEN price < 300 THEN 'premium'
                ELSE 'luxury'
            END                                  AS price_tier
        FROM raw_products WHERE product_id IS NOT NULL
    """)

    # fct_orders — joined and enriched
    duck.execute("""
        CREATE OR REPLACE TABLE fct_orders AS
        SELECT
            o.order_id, o.customer_id,
            CAST(o.order_date AS DATE)              AS order_date,
            YEAR(CAST(o.order_date AS DATE))        AS order_year,
            MONTH(CAST(o.order_date AS DATE))       AS order_month,
            o.status,
            COALESCE(o.shipping_city, 'Unknown')    AS shipping_city,
            COALESCE(o.shipping_state, 'Unknown')   AS shipping_state,
            COUNT(i.item_id)                        AS num_items,
            ROUND(SUM(i.line_total), 2)             AS order_total,
            ROUND(AVG(i.discount), 4)               AS avg_discount,
            ROUND(AVG((p.price - p.cost) / p.price), 4) AS avg_margin
        FROM raw_orders o
        LEFT JOIN raw_order_items i ON o.order_id   = i.order_id
        LEFT JOIN raw_products    p ON i.product_id = p.product_id
        WHERE o.order_id IS NOT NULL
        GROUP BY 1,2,3,4,5,6,7,8
    """)

    # fct_customer_metrics — RFM scoring
    duck.execute("""
        CREATE OR REPLACE TABLE fct_customer_metrics AS
        WITH completed AS (
            SELECT customer_id,
                DATEDIFF('day', MAX(CAST(order_date AS DATE)),
                    (SELECT MAX(CAST(order_date AS DATE)) FROM raw_orders)) AS recency_days,
                COUNT(order_id)          AS frequency,
                ROUND(SUM(order_total), 2) AS monetary
            FROM raw_orders
            WHERE status = 'completed'
            GROUP BY customer_id
        ),
        scored AS (
            SELECT *,
                NTILE(4) OVER (ORDER BY recency_days DESC) AS r_score,
                NTILE(4) OVER (ORDER BY frequency)         AS f_score,
                NTILE(4) OVER (ORDER BY monetary)          AS m_score
            FROM completed
        )
        SELECT
            c.customer_id, c.full_name, c.segment,
            s.recency_days, s.frequency, s.monetary,
            s.r_score, s.f_score, s.m_score,
            s.r_score + s.f_score + s.m_score AS rfm_score,
            CASE
                WHEN s.r_score + s.f_score + s.m_score >= 10 THEN 'Champions'
                WHEN s.r_score + s.f_score + s.m_score >= 7  THEN 'Loyal'
                WHEN s.r_score + s.f_score + s.m_score >= 5  THEN 'At Risk'
                ELSE 'Lost'
            END AS rfm_segment
        FROM scored s
        JOIN dim_customers c ON s.customer_id = c.customer_id
    """)

    # Verify
    counts = duck.execute("""
        SELECT 'dim_customers' AS tbl, COUNT(*) AS rows FROM dim_customers UNION ALL
        SELECT 'dim_products',         COUNT(*) FROM dim_products          UNION ALL
        SELECT 'fct_orders',           COUNT(*) FROM fct_orders            UNION ALL
        SELECT 'fct_customer_metrics', COUNT(*) FROM fct_customer_metrics
    """).df()
    print(counts.to_string(index=False))
    duck.close()
    print("Transformations complete!")

def check_data_quality():
    import duckdb
    duck = duckdb.connect("/opt/airflow/data/shopstream.duckdb")

    checks = [
        ("No null order IDs",
         "SELECT COUNT(*) FROM raw_orders WHERE order_id IS NULL",
         0),
        ("No negative revenue",
         "SELECT COUNT(*) FROM raw_orders WHERE order_total < 0",
         0),
        ("Customer count sanity",
         "SELECT COUNT(DISTINCT customer_id) FROM raw_customers",
         900),  # at least 900
    ]

    failures = []
    for name, query, expected in checks:
        result = duck.execute(query).fetchone()[0]
        if name.startswith("Customer count"):
            if result < expected:
                failures.append(f"FAIL: {name} — got {result}, expected >= {expected}")
            else:
                print(f"PASS: {name} — {result}")
        else:
            if result != expected:
                failures.append(f"FAIL: {name} — got {result}, expected {expected}")
            else:
                print(f"PASS: {name}")

    duck.close()

    if failures:
        raise Exception("Data quality checks failed:\n" + "\n".join(failures))
    print("All data quality checks passed!")

def log_pipeline_stats():
    import duckdb
    duck = duckdb.connect("/opt/airflow/data/shopstream.duckdb")

    stats = duck.execute("""
        SELECT
            COUNT(*)                        as total_orders,
            ROUND(SUM(order_total), 2)      as total_revenue,
            COUNT(DISTINCT customer_id)     as unique_customers
        FROM raw_orders
        WHERE status = 'completed'
    """).fetchone()

    print(f"""
    ╔══════════════════════════════════╗
    ║   ShopStream Pipeline Stats      ║
    ╠══════════════════════════════════╣
    ║  Completed orders : {stats[0]:>10,}  ║
    ║  Total revenue    : ${stats[1]:>10,.2f}  ║
    ║  Unique customers : {stats[2]:>10,}  ║
    ╚══════════════════════════════════╝
    """)
    duck.close()

def run_dbt():                                          # ← moved outside DAG block
    import subprocess
    import os

    dbt_project_dir = "/opt/airflow/dbt/shopstream_dbt"

    # Check dbt is installed, install if not
    result = subprocess.run(
        ["pip", "install", "dbt-duckdb", "-q"],
        capture_output=True, text=True
    )
    print(result.stdout)

    # Create a profiles.yml inside the container pointing to the right DB
    profiles_dir = "/opt/airflow/dbt/shopstream_dbt"
    profiles_content = """shopstream_dbt:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: /opt/airflow/data/shopstream.duckdb
      threads: 4
"""
    os.makedirs(profiles_dir, exist_ok=True)
    with open(f"{profiles_dir}/profiles.yml", "w") as f:
        f.write(profiles_content)

    print("Running dbt...")
    result = subprocess.run(
        ["dbt", "run",
         "--project-dir", dbt_project_dir,
         "--profiles-dir", profiles_dir],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"dbt run failed:\n{result.stderr}")

    print("Running dbt tests...")
    result = subprocess.run(
        ["dbt", "test",
         "--project-dir", dbt_project_dir,
         "--profiles-dir", profiles_dir],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"dbt test failed:\n{result.stderr}")

    print("dbt complete!")

with DAG(
    dag_id="shopstream_pipeline",
    default_args=default_args,
    description="Full ShopStream ETL pipeline",
    schedule="0 6 * * *",      # runs every day at 6am
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["shopstream", "etl"],
) as dag:

    t1_ingest = PythonOperator(
        task_id="ingest_raw_data",
        python_callable=run_ingestion,
    )

    t2_quality = PythonOperator(
        task_id="data_quality_checks",
        python_callable=check_data_quality,
    )

    t3_transform = PythonOperator(
        task_id="run_pyspark_transforms",
        python_callable=run_transformation,
    )

    t4_dbt = PythonOperator(                           # ← moved inside DAG block
        task_id="run_dbt_models",
        python_callable=run_dbt,
    )

    t5_stats = PythonOperator(                         # ← moved inside DAG block
        task_id="log_pipeline_stats",
        python_callable=log_pipeline_stats,
    )

    # ── Task dependencies (the DAG) ──────────────────────────
    # ingest → quality check → transform → dbt → stats
    t1_ingest >> t2_quality >> t3_transform >> t4_dbt >> t5_stats