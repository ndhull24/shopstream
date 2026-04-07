from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, BooleanType, TimestampType
)
import os

# Windows: set Hadoop home programmatically
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

# Spark session 
spark = SparkSession.builder \
    .appName("ShopStream ETL") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

RAW     = "data/raw"
WAREHOUSE = "data/warehouse"
os.makedirs(WAREHOUSE, exist_ok=True)

print("Phase 3A — PySpark transformation starting...\n")

# 1. LOAD RAW CSVs INTO SPARK

print("  Loading raw CSVs into Spark...")

customers = spark.read.csv(f"{RAW}/customers.csv",   header=True, inferSchema=True)
products  = spark.read.csv(f"{RAW}/products.csv",    header=True, inferSchema=True)
orders    = spark.read.csv(f"{RAW}/orders.csv",      header=True, inferSchema=True)
items     = spark.read.csv(f"{RAW}/order_items.csv", header=True, inferSchema=True)
events    = spark.read.csv(f"{RAW}/events.csv",      header=True, inferSchema=True)


# 2. CLEAN: CUSTOMERS - dim_customers
print("  Building dim_customers...")

dim_customers = customers \
    .dropDuplicates(["customer_id"]) \
    .fillna({"email": "unknown@shopstream.com"}) \
    .fillna({"age": 0}) \
    .withColumn("full_name",
        F.concat_ws(" ", F.col("first_name"), F.col("last_name"))) \
    .withColumn("signup_date", F.to_date("signup_date")) \
    .withColumn("age", F.col("age").cast(IntegerType())) \
    .select(
        "customer_id", "full_name", "first_name", "last_name",
        "email", "city", "state", "country",
        "segment", "signup_date", "age"
    )

print(f"    Customers before dedup: {customers.count():,}")
print(f"    Customers after dedup : {dim_customers.count():,}")


# 3. CLEAN: PRODUCTS to dim_products

print("  Building dim_products...")

dim_products = products \
    .withColumn("margin",
        F.round((F.col("price") - F.col("cost")) / F.col("price"), 4)) \
    .withColumn("price_tier",
        F.when(F.col("price") < 20,   "budget")
         .when(F.col("price") < 100,  "mid")
         .when(F.col("price") < 300,  "premium")
         .otherwise("luxury")) \
    .select(
        "product_id", "product_name", "category",
        "price", "cost", "margin", "price_tier",
        "stock_qty", "is_active"
    )


# 4. CLEAN: ORDERS to staging_orders

print("  Building staging_orders...")

staging_orders = orders \
    .fillna({"shipping_city": "Unknown", "shipping_state": "Unknown"}) \
    .withColumn("order_date", F.to_date("order_date")) \
    .withColumn("order_year",  F.year("order_date")) \
    .withColumn("order_month", F.month("order_date")) \
    .withColumn("order_dow",   F.dayofweek("order_date"))


# 5. BUILD: FACT TABLE to fct_orders

print("  Building fct_orders...")

fct_orders = staging_orders \
    .join(items, on="order_id", how="left") \
    .join(
        dim_products.select("product_id", "category", "price_tier", "margin"),
        on="product_id", how="left"
    ) \
    .groupBy(
        "order_id", "customer_id", "order_date",
        "order_year", "order_month", "order_dow",
        "status", "shipping_city", "shipping_state"
    ) \
    .agg(
        F.round(F.sum("line_total"), 2).alias("order_total"),
        F.count("item_id").alias("num_items"),
        F.round(F.avg("discount"), 4).alias("avg_discount"),
        F.round(F.avg("margin"), 4).alias("avg_margin"),
        F.collect_set("category").alias("categories_purchased")
    )


# 6. BUILD: CUSTOMER FEATURES to fct_customer_metrics
#    (RFM = Recency, Frequency, Monetary — used by Amazon/Uber)

print("  Building fct_customer_metrics (RFM)...")

from pyspark.sql.window import Window

max_date = orders.agg(F.max("order_date")).collect()[0][0]

rfm = staging_orders \
    .filter(F.col("status") == "completed") \
    .groupBy("customer_id") \
    .agg(
        F.datediff(F.lit(max_date), F.max("order_date")).alias("recency_days"),
        F.count("order_id").alias("frequency"),
        F.round(F.sum("order_total"), 2).alias("monetary"),
        F.min("order_date").alias("first_order_date"),
        F.max("order_date").alias("last_order_date"),
    )

# Score each dimension 1–4 using ntile (quartiles)
w = Window.orderBy("recency_days")
rfm = rfm \
    .withColumn("r_score", (5 - F.ntile(4).over(w)).cast(IntegerType())) \
    .withColumn("f_score",
        F.ntile(4).over(Window.orderBy("frequency"))) \
    .withColumn("m_score",
        F.ntile(4).over(Window.orderBy("monetary"))) \
    .withColumn("rfm_score",
        F.col("r_score") + F.col("f_score") + F.col("m_score")) \
    .withColumn("rfm_segment",
        F.when(F.col("rfm_score") >= 10, "Champions")
         .when(F.col("rfm_score") >= 7,  "Loyal")
         .when(F.col("rfm_score") >= 5,  "At Risk")
         .otherwise("Lost"))

fct_customer_metrics = dim_customers \
    .join(rfm, on="customer_id", how="left")


# 7. CLEAN: EVENTS to fct_events

print("  Building fct_events...")

fct_events = events \
    .withColumn("event_ts", F.to_timestamp("event_ts")) \
    .withColumn("event_date", F.to_date("event_ts")) \
    .withColumn("event_hour", F.hour("event_ts")) \
    .withColumn("is_anonymous", F.col("customer_id").isNull().cast(BooleanType())) \
    .fillna({"customer_id": "ANONYMOUS", "product_id": "NONE"})


# 8. WRITE PARQUET TO WAREHOUSE
print("\n  Writing Parquet files to warehouse...")

tables = {
    "dim_customers":        dim_customers,
    "dim_products":         dim_products,
    "fct_orders":           fct_orders,
    "fct_customer_metrics": fct_customer_metrics,
    "fct_events":           fct_events,
}

for name, df in tables.items():
    path = f"{WAREHOUSE}/{name}"
    df.write.mode("overwrite").parquet(path)
    count = df.count()
    print(f"    {name:<25} → {count:>6,} rows  →  {path}")


# 9. QUICK SANITY CHECK via Spark SQL

print("\n  Sanity checks...")

for name, df in tables.items():
    df.createOrReplaceTempView(name)

print("\n  RFM segment breakdown:")
spark.sql("""
    SELECT rfm_segment,
           COUNT(*)                       AS customers,
           ROUND(AVG(monetary), 2)        AS avg_spend,
           ROUND(AVG(recency_days), 1)    AS avg_recency_days
    FROM   fct_customer_metrics
    WHERE  rfm_segment IS NOT NULL
    GROUP  BY rfm_segment
    ORDER  BY avg_spend DESC
""").show()

print("  Top 5 revenue months:")
spark.sql("""
    SELECT order_year,
           order_month,
           COUNT(order_id)            AS orders,
           ROUND(SUM(order_total), 2) AS revenue
    FROM   fct_orders
    WHERE  status = 'completed'
    GROUP  BY order_year, order_month
    ORDER  BY revenue DESC
    LIMIT  5
""").show()

spark.stop()
print("Phase 3A complete! Star schema written to data/warehouse/")