from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess
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
    result = subprocess.run(
        ["python", "/opt/airflow/pipelines/transform.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Transformation failed:\n{result.stderr}")

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

    t4_dbt = BashOperator(
        task_id="run_dbt_models",
        bash_command=(
            "cd /opt/airflow && "
            "pip install dbt-duckdb -q && "
            "dbt run --project-dir /opt/airflow/dbt/shopstream_dbt "
            "--profiles-dir /opt/airflow/dbt/shopstream_dbt"
        ),
    )

    t5_stats = PythonOperator(
        task_id="log_pipeline_stats",
        python_callable=log_pipeline_stats,
    )

    # ── Task dependencies (the DAG) ──────────────────────────
    # ingest → quality check → transform → dbt → stats
    t1_ingest >> t2_quality >> t3_transform >> t4_dbt >> t5_stats