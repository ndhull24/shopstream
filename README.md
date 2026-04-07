# 🛍️ ShopStream — End-to-End Data Engineering & Data Science Project

> A production-pattern data engineering and machine learning project built on a fictional e-commerce company, **ShopStream**. Covers the full modern data stack from synthetic data generation to an interactive ML-powered dashboard — mirroring how companies like Amazon, Netflix, Uber, Airbnb, and Spotify operate their data infrastructure.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Phases](#phases)
  - [Phase 1 — Data Generation](#phase-1--data-generation)
  - [Phase 2 — Ingestion & Storage](#phase-2--ingestion--storage)
  - [Phase 3A — PySpark ETL](#phase-3a--pyspark-etl)
  - [Phase 3B — dbt Transformations](#phase-3b--dbt-transformations)
  - [Phase 4 — Airflow Orchestration](#phase-4--airflow-orchestration)
  - [Phase 5 — Machine Learning](#phase-5--machine-learning)
  - [Phase 6 — Streamlit Dashboard](#phase-6--streamlit-dashboard)
- [Getting Started](#getting-started)
- [Key Learnings](#key-learnings)
- [Industry Patterns Implemented](#industry-patterns-implemented)

---

## Project Overview

ShopStream is a comprehensive, hands-on data engineering project that simulates a real e-commerce company's data infrastructure. The project generates realistic synthetic data, ingests it into a database, transforms it through a multi-layer pipeline, orchestrates it with Airflow, and surfaces ML-powered insights through a Streamlit dashboard.

**Dataset summary:**

| Table | Rows | Description |
|---|---|---|
| customers | 1,000 | Customers with segments, demographics, signup dates |
| products | 200 | Products across 5 categories with pricing and margins |
| orders | 5,000 | Orders over 2 years with statuses and revenue |
| order_items | ~15,000 | Line items per order with discounts |
| events | 50,000 | Clickstream events (page views, add-to-cart, purchases) |

Data is intentionally messy — duplicates, nulls, skewed distributions — to mirror real-world data quality challenges.

---

## Architecture

```
Raw CSVs (Faker)
      │
      ▼
SQLite + DuckDB          ← Phase 2: Ingestion
      │
      ▼
PySpark ETL              ← Phase 3A: Cleaning, joins, star schema, RFM
      │
      ▼
dbt Models               ← Phase 3B: Staging → Marts, tests, docs
      │
      ▼
Parquet Warehouse        ← Star schema: dim_customers, dim_products, fct_orders
      │
      ▼
Airflow DAG              ← Phase 4: Scheduled orchestration (Docker)
      │
      ▼
ML Notebooks             ← Phase 5: Churn, segmentation, forecasting
      │
      ▼
Streamlit Dashboard      ← Phase 6: Interactive business intelligence
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data generation | Python, Faker, NumPy, pandas |
| Storage | SQLite, DuckDB, Parquet |
| ETL | PySpark, SQL |
| Transformation | dbt (dbt-duckdb adapter) |
| Orchestration | Apache Airflow 2.9 (Docker) |
| Machine learning | scikit-learn (Random Forest, K-Means, Linear Regression) |
| Visualisation | Streamlit, Plotly |
| Version control | Git |
| Environment | VS Code, Python venv |

---

## Project Structure

```
shopstream/
├── data/
│   ├── raw/                    # Generated CSV files
│   └── warehouse/              # Parquet files (star schema)
├── pipelines/
│   ├── generate_data.py        # Phase 1: Synthetic data generation
│   ├── ingest.py               # Phase 2: CSV → SQLite + DuckDB
│   ├── transform.py            # Phase 3A: PySpark ETL
│   └── dashboard.py            # Phase 6: Streamlit dashboard
├── dbt/
│   └── shopstream_dbt/
│       ├── models/
│       │   ├── staging/        # stg_customers, stg_orders, stg_products, stg_order_items
│       │   ├── intermediate/
│       │   └── marts/          # mart_orders (final business table)
│       ├── dbt_project.yml
│       └── profiles.yml
├── airflow/
│   ├── dags/
│   │   └── shopstream_pipeline.py   # Full DAG: ingest → quality → transform → dbt → stats
│   ├── docker-compose.yml
│   └── logs/
├── notebooks/
│   └── shopstream_ml.ipynb     # Phase 5: EDA + ML models
├── requirements.txt
└── README.md
```

---

## Phases

### Phase 1 — Data Generation

Generates a realistic, intentionally messy e-commerce dataset using Python's `Faker` library.

**Key features:**
- 1,000 customers with 4 segments (VIP, Regular, At-Risk, New)
- 200 products across 5 categories with realistic price ranges
- 5,000 orders with 4 statuses (completed, pending, cancelled, refunded)
- ~15,000 order line items with variable discounts
- 50,000 clickstream events across 6 event types
- Intentional data quality issues: ~10 duplicate customers, 2% missing emails, 3% missing shipping info, 15% anonymous events

```bash
python pipelines/generate_data.py
```

---

### Phase 2 — Ingestion & Storage

Loads raw CSVs into both SQLite (transactional queries) and DuckDB (analytical queries), mirrors how production systems separate OLTP and OLAP workloads.

**Key concepts:** Schema design, primary/foreign keys, data type casting, SQL validation queries.

```bash
python pipelines/ingest.py
```

**Sample output:**
```
Total customers     : 1,010  (incl. 10 duplicates)
Missing emails      : 19
Revenue (completed) : $5,113,032.80
```

---

### Phase 3A — PySpark ETL

Transforms raw tables into a star schema using PySpark, implementing the same distributed processing patterns used by Amazon EMR and Databricks.

**Transformations:**
- `dim_customers` — deduplication via `ROW_NUMBER()`, null filling, full name construction
- `dim_products` — margin calculation, price tier classification (budget/mid/premium/luxury)
- `fct_orders` — joined fact table with order metrics
- `fct_customer_metrics` — **RFM scoring** (Recency, Frequency, Monetary) with quartile-based segmentation
- `fct_events` — cleaned clickstream with anonymity flag and time features

**RFM segments produced:**

| Segment | Customers | Avg Spend | Avg Recency |
|---|---|---|---|
| Champions | 257 | $9,311 | 70 days |
| Loyal | 340 | $5,447 | 157 days |
| At Risk | 211 | $3,026 | 237 days |
| Lost | 156 | $1,472 | 372 days |

```bash
python pipelines/transform.py
```

---

### Phase 3B — dbt Transformations

Implements a three-layer dbt project on top of DuckDB, mirroring how Airbnb and Spotify structure their transformation layer.

**Models:**

| Model | Type | Description |
|---|---|---|
| `stg_customers` | View | Cleaned, deduplicated customers |
| `stg_orders` | View | Cleaned orders with date parts |
| `stg_products` | View | Products with derived margin and price tier |
| `stg_order_items` | View | Cleaned line items with net price |
| `mart_orders` | Table | Final business-ready joined order table |

**dbt tests (11 passing):**
- `unique` and `not_null` on all primary keys
- `accepted_values` on `segment` and `status` columns
- Caught and fixed 10 duplicate customer records via deduplication model

```bash
cd dbt/shopstream_dbt
dbt run      # builds all models
dbt test     # runs 11 data quality tests
dbt docs serve  # opens auto-generated data catalog at localhost:8080
```

---

### Phase 4 — Airflow Orchestration

Schedules and automates the full pipeline as a daily DAG using Apache Airflow running in Docker, mirroring Netflix's Airflow setup.

**DAG:** `shopstream_pipeline` — runs daily at 6am UTC

```
ingest_raw_data → data_quality_checks → run_pyspark_transforms → run_dbt_models → log_pipeline_stats
```

**Task breakdown:**

| Task | Operator | Description |
|---|---|---|
| `ingest_raw_data` | PythonOperator | Loads CSVs into SQLite + DuckDB |
| `data_quality_checks` | PythonOperator | Validates nulls, row counts, business rules |
| `run_pyspark_transforms` | PythonOperator | Runs DuckDB star schema transformations |
| `run_dbt_models` | PythonOperator | Runs dbt models and tests |
| `log_pipeline_stats` | PythonOperator | Prints final revenue and order stats |

```bash
cd airflow
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler
# UI at http://localhost:8081  (admin / admin)
```

---

### Phase 5 — Machine Learning

Three ML models trained on the warehouse data in a Jupyter notebook, implementing patterns from Uber's Michelangelo and Spotify's ML platform.

**Model 1 — Churn Prediction (Random Forest)**
- Binary classification: will a customer churn (no order in 180+ days)?
- Features: recency, frequency, monetary, RFM scores
- Output: churn probability per customer with risk tier (Low / Medium / High)

**Model 2 — Customer Segmentation (K-Means)**
- Clusters customers into 4 behavioural segments
- Features: recency, frequency, monetary (StandardScaler normalised)
- Elbow method + silhouette score used to select optimal k=4
- Segments: Champions, Loyal, At Risk, Dormant

**Model 3 — Sales Forecasting (Linear Regression)**
- Predicts monthly revenue for the next 3 months
- Features: time index, Fourier sin/cos seasonality terms
- Produces MAE and RMSE on held-out test set

```bash
# Open in VS Code with .venv kernel selected
notebooks/shopstream_ml.ipynb
```

---

### Phase 6 — Streamlit Dashboard

Interactive business intelligence dashboard with 5 pages, reading live from DuckDB, mirroring Airbnb's Superset setup.

**Pages:**

| Page | Contents |
|---|---|
| 📊 Overview | Revenue KPIs, monthly trend, order status, day-of-week revenue |
| 🛍️ Products | Revenue by category, price tier distribution, top 20 products |
| 👥 Customers | Segment breakdown, RFM distribution, recency vs monetary scatter |
| 🤖 ML Insights | Churn risk distribution, ML segments, top 20 at-risk customers |
| 📈 Forecast | 3-month revenue forecast with confidence, MAE metric |

**Features:**
- Global filters (year, product category) applied across all pages
- Green / Yellow / Red semantic colour coding across all KPI cards
- Dynamic cancellation rate card changes colour based on actual rate

```bash
streamlit run pipelines/dashboard.py
# Opens at http://localhost:8501
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Java 17 (for PySpark) — [Adoptium Temurin 17](https://adoptium.net)
- Docker Desktop (for Airflow)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/YOURUSERNAME/shopstream.git
cd shopstream

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Phase 1: Generate data
python pipelines/generate_data.py

# Phase 2: Ingest into databases
python pipelines/ingest.py

# Phase 3A: PySpark transformations
python pipelines/transform.py

# Phase 3B: dbt models and tests
cd dbt/shopstream_dbt
dbt run && dbt test
cd ../..

# Phase 4: Start Airflow (Docker required)
cd airflow
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler
cd ..

# Phase 6: Launch dashboard
streamlit run pipelines/dashboard.py
```

---

## Key Learnings

- **Intentional data messiness** — building pipelines that handle real-world issues (nulls, duplicates, type mismatches) from day one
- **Separation of concerns** — raw → staging → intermediate → mart model layers in dbt
- **RFM analysis** — a fundamental customer analytics pattern used across e-commerce and SaaS
- **Orchestration vs execution** — Airflow coordinates work, it doesn't do the work itself
- **Star schema design** — fact and dimension tables optimised for analytical queries
- **Feature engineering** — transforming raw data into ML-ready features (RFM scores, seasonality terms)

---

## Industry Patterns Implemented

| Pattern | Inspired by | Implementation |
|---|---|---|
| Data lake (Parquet + columnar storage) | Amazon S3 + Redshift | `data/warehouse/*.parquet` |
| Distributed ETL | Amazon EMR / Databricks | PySpark `transform.py` |
| Event streaming simulation | Netflix Kafka | `events.csv` clickstream |
| Pipeline orchestration DAG | Netflix Airflow | `airflow/dags/shopstream_pipeline.py` |
| SQL transformation layers | Uber dbt + Hive | `dbt/shopstream_dbt/models/` |
| ML feature store | Uber Michelangelo | RFM features in `fct_customer_metrics` |
| Data quality checks | Airbnb Great Expectations | dbt tests + Airflow quality task |
| BI dashboard | Airbnb Superset | Streamlit + Plotly dashboard |
| Recommendation / segmentation | Spotify ML platform | K-Means clustering + collaborative patterns |
| A/B test readiness | Spotify experimentation | Statistical foundations in ML notebook |

---

## Author

**Navdeep Singh**

Built as a comprehensive learning project covering the full modern data stack — from raw data generation to production-pattern ML-powered dashboards.

---

*Built with Python, PySpark, dbt, Airflow, scikit-learn, DuckDB, and Streamlit.*
