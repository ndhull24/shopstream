import pandas as pd
import numpy as np
from faker import Faker
import random
import os
from datetime import datetime, timedelta

# Reproducibility
random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("ShopStream data generation starting...")

# 1. CUSTOMERS

print("  Generating customers...")

NUM_CUSTOMERS = 1000

segments    = ["VIP", "Regular", "At-Risk", "New"]
seg_weights = [0.10,  0.55,      0.20,      0.15]

customers = []
for i in range(1, NUM_CUSTOMERS + 1):
    signup = fake.date_between(start_date="-3y", end_date="today")
    customers.append({
        "customer_id":   f"CUST_{i:04d}",
        "first_name":    fake.first_name(),
        "last_name":     fake.last_name(),
        "email":         fake.email() if random.random() > 0.02 else None,  # 2% missing
        "city":          fake.city(),
        "state":         fake.state_abbr(),
        "country":       "US",
        "segment":       random.choices(segments, seg_weights)[0],
        "signup_date":   signup,
        "age":           random.randint(18, 70) if random.random() > 0.05 else None,
    })

# Inject ~10 duplicate rows (real-world dirty data)
for _ in range(10):
    customers.append(random.choice(customers))

df_customers = pd.DataFrame(customers)
df_customers.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
print(f"    customers.csv — {len(df_customers)} rows")

# 2. PRODUCTS

print("  Generating products...")

categories = {
    "Electronics":  (49.99,  899.99),
    "Clothing":     (9.99,   149.99),
    "Home & Garden":(14.99,  299.99),
    "Sports":       (19.99,  399.99),
    "Books":        (4.99,   49.99),
}

products = []
prod_id = 1
for category, (min_price, max_price) in categories.items():
    for _ in range(40):  # 40 products per category = 200 total
        price = round(random.uniform(min_price, max_price), 2)
        products.append({
            "product_id":   f"PROD_{prod_id:04d}",
            "product_name": fake.catch_phrase(),
            "category":     category,
            "price":        price,
            "cost":         round(price * random.uniform(0.3, 0.6), 2),
            "stock_qty":    random.randint(0, 500),
            "is_active":    random.choices([True, False], [0.9, 0.1])[0],
        })
        prod_id += 1

df_products = pd.DataFrame(products)
df_products.to_csv(f"{OUTPUT_DIR}/products.csv", index=False)
print(f"    products.csv — {len(df_products)} rows")


# 3. ORDERS
print("  Generating orders...")

NUM_ORDERS   = 5000
statuses     = ["completed", "pending", "cancelled", "refunded"]
stat_weights = [0.70,        0.15,      0.10,        0.05]

customer_ids = df_customers["customer_id"].unique().tolist()
product_ids  = df_products["product_id"].tolist()

orders = []
order_items = []
item_id = 1

for i in range(1, NUM_ORDERS + 1):
    order_date  = fake.date_between(start_date="-2y", end_date="today")
    status      = random.choices(statuses, stat_weights)[0]
    customer_id = random.choice(customer_ids)

    # Each order has 1–5 line items
    n_items     = random.randint(1, 5)
    order_total = 0

    for _ in range(n_items):
        prod    = df_products.sample(1).iloc[0]
        qty     = random.randint(1, 4)
        price   = prod["price"]
        discount = round(random.uniform(0, 0.3), 2) if random.random() > 0.7 else 0.0
        line_total = round(qty * price * (1 - discount), 2)
        order_total += line_total

        order_items.append({
            "item_id":      f"ITEM_{item_id:06d}",
            "order_id":     f"ORD_{i:05d}",
            "product_id":   prod["product_id"],
            "quantity":     qty,
            "unit_price":   price,
            "discount":     discount,
            "line_total":   line_total,
        })
        item_id += 1

    orders.append({
        "order_id":      f"ORD_{i:05d}",
        "customer_id":   customer_id,
        "order_date":    order_date,
        "status":        status,
        "order_total":   round(order_total, 2),
        # Inject ~3% missing shipping info
        "shipping_city": fake.city() if random.random() > 0.03 else None,
        "shipping_state":fake.state_abbr() if random.random() > 0.03 else None,
    })

df_orders      = pd.DataFrame(orders)
df_order_items = pd.DataFrame(order_items)

df_orders.to_csv(f"{OUTPUT_DIR}/orders.csv",           index=False)
df_order_items.to_csv(f"{OUTPUT_DIR}/order_items.csv", index=False)
print(f"    orders.csv      — {len(df_orders)} rows")
print(f"    order_items.csv — {len(df_order_items)} rows")

# 4. CLICKSTREAM EVENTS

print("  Generating clickstream events...")

NUM_EVENTS  = 50000
event_types = ["page_view", "search", "add_to_cart", "remove_from_cart", "purchase", "wishlist_add"]
evt_weights = [0.50,        0.20,    0.15,           0.05,               0.07,       0.03]

events = []
for i in range(1, NUM_EVENTS + 1):
    event_date = fake.date_time_between(start_date="-2y", end_date="now")
    events.append({
        "event_id":    f"EVT_{i:07d}",
        "customer_id": random.choice(customer_ids) if random.random() > 0.15 else None,  # 15% anonymous
        "session_id":  f"SESS_{random.randint(1, 15000):06d}",
        "event_type":  random.choices(event_types, evt_weights)[0],
        "product_id":  random.choice(product_ids) if random.random() > 0.3 else None,
        "event_ts":    event_date,
        "device":      random.choice(["mobile", "desktop", "tablet"]),
        "os":          random.choice(["iOS", "Android", "Windows", "macOS"]),
    })

df_events = pd.DataFrame(events)
df_events.to_csv(f"{OUTPUT_DIR}/events.csv", index=False)
print(f"    events.csv — {len(df_events)} rows")

# 5. SUMMARY

print("\n Data generation complete!")
print(f"   customers  : {len(df_customers):>6,} rows  (incl. ~10 duplicates)")
print(f"   products   : {len(df_products):>6,} rows")
print(f"   orders     : {len(df_orders):>6,} rows")
print(f"   order_items: {len(df_order_items):>6,} rows")
print(f"   events     : {len(df_events):>6,} rows")
print(f"\n   Files saved to → {OUTPUT_DIR}/")