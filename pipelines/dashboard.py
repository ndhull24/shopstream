import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings("ignore")

# ── Resolve paths ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "shopstream.duckdb")
print(f"DB PATH: {DB_PATH}")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ShopStream Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    .stMetric { background-color: #1e2130; border-radius: 10px; padding: 10px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #a78bfa;
        margin: 20px 0 10px 0;
        border-bottom: 2px solid #a78bfa;
        padding-bottom: 5px;
    }
    [data-testid="stMetricDelta"] svg { display: none; }
    div[data-testid="stMetricDelta"] > div { font-weight: 600; }
    div[data-testid="stMetricDeltaPositive"] { color: #22c55e !important; }
    div[data-testid="stMetricDeltaNegative"] { color: #ef4444 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────
@st.cache_data
def load_data():
    duck = duckdb.connect(DB_PATH, read_only=True)

    orders = duck.execute("SELECT * FROM fct_orders").df()
    customers = duck.execute("SELECT * FROM dim_customers").df()
    products = duck.execute("SELECT * FROM dim_products").df()
    rfm = duck.execute("SELECT * FROM fct_customer_metrics").df()

    monthly = duck.execute("""
        SELECT
            order_year, order_month,
            COUNT(order_id)             AS num_orders,
            ROUND(SUM(order_total), 2)  AS revenue,
            COUNT(DISTINCT customer_id) AS unique_customers
        FROM fct_orders
        WHERE status = 'completed'
        GROUP BY order_year, order_month
        ORDER BY order_year, order_month
    """).df()

    cat_revenue = duck.execute("""
        SELECT p.category,
               COUNT(DISTINCT o.order_id)       AS orders,
               ROUND(SUM(oi.line_total), 2)     AS revenue,
               ROUND(AVG(oi.line_total), 2)     AS avg_order_value
        FROM raw_order_items oi
        JOIN raw_products p ON oi.product_id = p.product_id
        JOIN raw_orders   o ON oi.order_id   = o.order_id
        WHERE o.status = 'completed'
        GROUP BY p.category ORDER BY revenue DESC
    """).df()

    top_products = duck.execute("""
        SELECT p.product_name, p.category,
               CASE
                   WHEN p.price < 20  THEN 'budget'
                   WHEN p.price < 100 THEN 'mid'
                   WHEN p.price < 300 THEN 'premium'
                   ELSE 'luxury'
               END                              AS price_tier,
               COUNT(oi.item_id)                AS times_ordered,
               ROUND(SUM(oi.line_total), 2)     AS total_revenue
        FROM raw_order_items oi
        JOIN raw_products p ON oi.product_id = p.product_id
        JOIN raw_orders   o ON oi.order_id   = o.order_id
        WHERE o.status = 'completed'
        GROUP BY p.product_name, p.category, p.price
        ORDER BY total_revenue DESC
        LIMIT 20
    """).df()

    duck.close()
    return orders, customers, products, rfm, monthly, cat_revenue, top_products

@st.cache_data
def build_ml_models(rfm):
    features = ["recency_days", "frequency", "monetary",
                "r_score", "f_score", "m_score"]
    df = rfm[rfm["rfm_segment"].notna()].copy()
    X = df[features].fillna(0)

    # Churn model
    df["churned"] = (df["recency_days"] >= 180).astype(int)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                  random_state=42, n_jobs=-1)
    clf.fit(X, df["churned"])
    df["churn_probability"] = clf.predict_proba(X)[:, 1]
    df["churn_risk"] = pd.cut(df["churn_probability"],
                               bins=[0, 0.3, 0.6, 1.0],
                               labels=["Low", "Medium", "High"])

    # Segmentation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["recency_days", "frequency", "monetary"]])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    cluster_map = df.groupby("cluster")["monetary"].mean().sort_values(
        ascending=False).index
    label_map = {c: l for c, l in zip(
        cluster_map, ["Champions", "Loyal", "At Risk", "Dormant"])}
    df["ml_segment"] = df["cluster"].map(label_map)

    return df

# ── Load data ────────────────────────────────────────────────
with st.spinner("Loading ShopStream data..."):
    orders, customers, products, rfm, monthly, cat_revenue, top_products = load_data()
    ml_df = build_ml_models(rfm)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/6d28d9/ffffff?text=ShopStream")
    st.markdown("---")
    st.markdown("### Filters")

    years = sorted(orders["order_year"].dropna().unique().tolist())
    selected_years = st.multiselect("Year", years, default=years)

    categories = sorted(products["category"].unique().tolist())
    selected_cats = st.multiselect("Product category", categories, default=categories)

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Navigation", ["📊 Overview", "🛍️ Products",
                  "👥 Customers", "🤖 ML Insights",
                  "📈 Forecast"],
                label_visibility="collapsed")
    st.markdown("---")
    st.caption("ShopStream v1.0 | Built with Streamlit + DuckDB")

# ── Filter data ──────────────────────────────────────────────
filtered_orders = orders[orders["order_year"].isin(selected_years)]
completed = filtered_orders[filtered_orders["status"] == "completed"]

# ── KPI card template & helper ───────────────────────────────
kpi_style = """
<div style="
    background: {bg};
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    border-left: 5px solid {border};
">
    <div style="font-size:1.6rem; margin-bottom:4px;">{icon}</div>
    <div style="color:#94a3b8; font-size:0.75rem;
                text-transform:uppercase; letter-spacing:1px;
                margin-bottom:6px;">{label}</div>
    <div style="color:{value_color}; font-size:1.8rem;
                font-weight:800; line-height:1.1;">{value}</div>
    <div style="color:{delta_color}; font-size:0.8rem;
                margin-top:6px; font-weight:600;">{delta}</div>
</div>
"""

def kpi_card(col, bg, border, icon, label, value, value_color, delta, delta_color):
    col.markdown(kpi_style.format(
        bg=bg, border=border, icon=icon,
        label=label, value=value,
        value_color=value_color,
        delta=delta, delta_color=delta_color
    ), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 ShopStream — Business Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    total_rev    = completed["order_total"].sum()
    total_orders = len(completed)
    avg_order    = completed["order_total"].mean()
    unique_custs = completed["customer_id"].nunique()
    cancel_rate  = (filtered_orders["status"] == "cancelled").mean() * 100

    kpi_card(col1, "#052e16", "#22c55e", "💰",
             "Total revenue", f"${total_rev:,.0f}",
             "#22c55e", "All time", "#86efac")

    kpi_card(col2, "#172554", "#3b82f6", "📦",
             "Completed orders", f"{total_orders:,}",
             "#60a5fa",
             f"{total_orders/len(filtered_orders)*100:.0f}% of all orders",
             "#93c5fd")

    kpi_card(col3, "#1c1917", "#f59e0b", "🛒",
             "Avg order value", f"${avg_order:,.2f}",
             "#fbbf24", "Per completed order", "#fde68a")

    kpi_card(col4, "#0f172a", "#a78bfa", "👥",
             "Unique customers", f"{unique_custs:,}",
             "#c4b5fd", "With completed orders", "#ddd6fe")

    cancel_color = "#ef4444" if cancel_rate > 15 else \
                   "#f59e0b" if cancel_rate > 8  else "#22c55e"
    cancel_bg    = "#2d0a0a" if cancel_rate > 15 else \
                   "#1c1205" if cancel_rate > 8  else "#052e16"

    kpi_card(col5, cancel_bg, cancel_color, "❌",
             "Cancellation rate", f"{cancel_rate:.1f}%",
             cancel_color,
             "🔴 High"   if cancel_rate > 15 else
             "🟡 Watch"  if cancel_rate > 8  else "🟢 Healthy",
             cancel_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly revenue chart
    st.markdown('<p class="section-header">Monthly revenue trend</p>',
                unsafe_allow_html=True)

    monthly["period"] = monthly["order_year"].astype(str) + "-" + \
                        monthly["order_month"].astype(str).str.zfill(2)
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=monthly["period"], y=monthly["revenue"],
        mode="lines+markers", name="Revenue",
        line=dict(color="#a78bfa", width=3),
        marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.1)"
    ))
    fig_rev.add_trace(go.Bar(
        x=monthly["period"], y=monthly["num_orders"],
        name="Orders", yaxis="y2",
        marker_color="rgba(34,211,238,0.4)"
    ))
    fig_rev.update_layout(
        template="plotly_dark", height=350,
        yaxis=dict(title="Revenue ($)"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # Order status breakdown
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Order status breakdown</p>',
                    unsafe_allow_html=True)
        status_counts = filtered_orders["status"].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_dark"
        )
        fig_status.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_status, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Revenue by day of week</p>',
                    unsafe_allow_html=True)
        dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        completed_dow = completed.copy()
        completed_dow["order_date"] = pd.to_datetime(completed_dow["order_date"])
        completed_dow["dow"] = completed_dow["order_date"].dt.dayofweek
        dow_rev = completed_dow.groupby("dow")["order_total"].sum().reset_index()
        dow_rev["day"] = dow_rev["dow"].map(dow_map)
        fig_dow = px.bar(dow_rev, x="day", y="order_total",
                     color="order_total",
                     color_continuous_scale="Viridis",
                     template="plotly_dark",
                     labels={"order_total": "Revenue ($)", "day": "Day"})
        fig_dow.update_layout(height=320, margin=dict(t=10, b=10),
                               showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE: PRODUCTS
# ════════════════════════════════════════════════════════════
elif page == "🛍️ Products":
    st.title("🛍️ Product Analytics")

    col1, col2, col3 = st.columns(3)

    kpi_card(col1, "#172554", "#3b82f6", "📦",
             "Total products", f"{len(products):,}",
             "#60a5fa", "In catalogue", "#93c5fd")

    active_pct = products['is_active'].sum() / len(products) * 100
    kpi_card(col2, "#052e16", "#22c55e", "✅",
             "Active products", f"{int(products['is_active'].sum()):,}",
             "#22c55e", f"{active_pct:.0f}% of catalogue", "#86efac")

    kpi_card(col3, "#1c1917", "#f59e0b", "🏷️",
             "Categories", f"{products['category'].nunique()}",
             "#fbbf24", "Product categories", "#fde68a")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Revenue by category</p>',
                    unsafe_allow_html=True)
        fig_cat = px.bar(
            cat_revenue, x="revenue", y="category",
            orientation="h", color="revenue",
            color_continuous_scale="Plasma",
            template="plotly_dark",
            labels={"revenue": "Revenue ($)", "category": ""}
        )
        fig_cat.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Price tier distribution</p>',
                    unsafe_allow_html=True)
        tier_counts = products["price_tier"].value_counts()
        fig_tier = px.pie(
            values=tier_counts.values, names=tier_counts.index,
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_dark"
        )
        fig_tier.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown('<p class="section-header">Top 20 products by revenue</p>',
                unsafe_allow_html=True)
    fig_top = px.bar(
        top_products, x="total_revenue", y="product_name",
        orientation="h", color="category",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template="plotly_dark",
        labels={"total_revenue": "Revenue ($)", "product_name": ""}
    )
    fig_top.update_layout(height=550, margin=dict(t=10, b=10))
    st.plotly_chart(fig_top, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE: CUSTOMERS
# ════════════════════════════════════════════════════════════
elif page == "👥 Customers":
    st.title("👥 Customer Analytics")

    col1, col2, col3, col4 = st.columns(4)

    kpi_card(col1, "#172554", "#3b82f6", "👥",
             "Total customers", f"{len(customers):,}",
             "#60a5fa", "All time", "#93c5fd")

    kpi_card(col2, "#052e16", "#22c55e", "⭐",
             "VIP customers", f"{(customers['segment']=='VIP').sum():,}",
             "#22c55e", "🟢 Highest value", "#86efac")

    at_risk = (customers['segment']=='At-Risk').sum()
    at_risk_pct = at_risk / len(customers) * 100
    kpi_card(col3, "#1c1205", "#f59e0b", "⚠️",
             "At-Risk", f"{at_risk:,}",
             "#f59e0b", f"🟡 {at_risk_pct:.0f}% of base", "#fde68a")

    kpi_card(col4, "#0f172a", "#a78bfa", "🆕",
             "New customers", f"{(customers['segment']=='New').sum():,}",
             "#c4b5fd", "Recently acquired", "#ddd6fe")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Customer segments</p>',
                    unsafe_allow_html=True)
        seg_counts = customers["segment"].value_counts()
        fig_seg = px.pie(
            values=seg_counts.values, names=seg_counts.index,
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_dark"
        )
        fig_seg.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">RFM segment distribution</p>',
                    unsafe_allow_html=True)
        rfm_counts = rfm["rfm_segment"].value_counts()
        fig_rfm = px.bar(
            x=rfm_counts.index, y=rfm_counts.values,
            color=rfm_counts.values,
            color_continuous_scale="Turbo",
            template="plotly_dark",
            labels={"x": "Segment", "y": "Customers"}
        )
        fig_rfm.update_layout(height=320, margin=dict(t=10, b=10),
                               showlegend=False)
        st.plotly_chart(fig_rfm, use_container_width=True)

    st.markdown('<p class="section-header">Recency vs monetary spend</p>',
                unsafe_allow_html=True)
    fig_scatter = px.scatter(
        rfm[rfm["rfm_segment"].notna()],
        x="recency_days", y="monetary",
        color="rfm_segment", size="frequency",
        color_discrete_map={
            "Champions": "#22c55e",
            "Loyal":     "#86efac",
            "At Risk":   "#f59e0b",
            "Lost":      "#ef4444"
        },
        template="plotly_dark",
        labels={"recency_days": "Days since last order",
                "monetary": "Total spend ($)",
                "rfm_segment": "Segment"}
    )
    fig_scatter.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE: ML INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "🤖 ML Insights":
    st.title("🤖 Machine Learning Insights")

    col1, col2, col3 = st.columns(3)
    high_churn = (ml_df["churn_risk"] == "High").sum()
    champions  = (ml_df["ml_segment"] == "Champions").sum()
    avg_churn  = ml_df["churn_probability"].mean()

    high_churn_pct = high_churn / len(ml_df) * 100
    kpi_card(col1, "#2d0a0a", "#ef4444", "🔴",
             "High churn risk", f"{high_churn:,} customers",
             "#ef4444", f"🔴 {high_churn_pct:.0f}% at risk", "#fca5a5")

    champ_pct = champions / len(ml_df) * 100
    kpi_card(col2, "#052e16", "#22c55e", "🏆",
             "Champions", f"{champions:,} customers",
             "#22c55e", f"🟢 Top {champ_pct:.0f}% of base", "#86efac")

    risk_color = "#ef4444" if avg_churn > 0.5 else \
                 "#f59e0b" if avg_churn > 0.3 else "#22c55e"
    risk_bg    = "#2d0a0a" if avg_churn > 0.5 else \
                 "#1c1205" if avg_churn > 0.3 else "#052e16"
    risk_label = "🔴 High" if avg_churn > 0.5 else \
                 "🟡 Moderate" if avg_churn > 0.3 else "🟢 Low"
    kpi_card(col3, risk_bg, risk_color, "📉",
             "Avg churn probability", f"{avg_churn:.1%}",
             risk_color, risk_label, risk_color)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Churn risk distribution</p>',
                    unsafe_allow_html=True)
        risk_counts = ml_df["churn_risk"].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values, names=risk_counts.index,
            hole=0.45,
            color=risk_counts.index,
            color_discrete_map={
                "Low":    "#22c55e",
                "Medium": "#f59e0b",
                "High":   "#ef4444"
            },
            template="plotly_dark"
        )
        fig_risk.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">ML customer segments</p>',
                    unsafe_allow_html=True)
        ml_seg_counts = ml_df["ml_segment"].value_counts()
        fig_ml_seg = px.bar(
            x=ml_seg_counts.index, y=ml_seg_counts.values,
            color=ml_seg_counts.index,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_dark",
            labels={"x": "Segment", "y": "Customers"}
        )
        fig_ml_seg.update_layout(height=320, margin=dict(t=10, b=10),
                                  showlegend=False)
        st.plotly_chart(fig_ml_seg, use_container_width=True)

    st.markdown('<p class="section-header">Churn probability by RFM segment</p>',
                unsafe_allow_html=True)
    fig_churn_rfm = px.box(
        ml_df[ml_df["rfm_segment"].notna()],
        x="rfm_segment", y="churn_probability",
        color="rfm_segment",
        color_discrete_map={
            "Champions": "#22c55e",
            "Loyal":     "#86efac",
            "At Risk":   "#f59e0b",
            "Lost":      "#ef4444"
        },
        template="plotly_dark",
        labels={"rfm_segment": "RFM Segment",
                "churn_probability": "Churn probability"}
    )
    fig_churn_rfm.update_layout(height=380, margin=dict(t=10, b=10),
                                 showlegend=False)
    st.plotly_chart(fig_churn_rfm, use_container_width=True)

    st.markdown('<p class="section-header">Top 20 customers at highest churn risk</p>',
                unsafe_allow_html=True)
    top_churn = ml_df.nlargest(20, "churn_probability")[[
        "customer_id", "rfm_segment", "recency_days",
        "frequency", "monetary", "churn_probability"
    ]].copy()
    top_churn["churn_probability"] = top_churn["churn_probability"].map("{:.1%}".format)
    top_churn["monetary"] = top_churn["monetary"].map("${:,.2f}".format)
    st.dataframe(top_churn, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE: FORECAST
# ════════════════════════════════════════════════════════════
elif page == "📈 Forecast":
    st.title("📈 Sales Forecast")

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    ts = orders[orders["status"] == "completed"].copy()
    ts["order_date"] = pd.to_datetime(ts["order_date"])
    ts["period"] = ts["order_date"].dt.to_period("M")

    monthly_rev = ts.groupby("period")["order_total"].sum().reset_index()
    monthly_rev["period_idx"] = range(len(monthly_rev))
    monthly_rev["period_str"] = monthly_rev["period"].astype(str)
    monthly_rev["month_num"]  = monthly_rev["period"].dt.month
    monthly_rev["sin_month"]  = np.sin(2 * np.pi * monthly_rev["month_num"] / 12)
    monthly_rev["cos_month"]  = np.cos(2 * np.pi * monthly_rev["month_num"] / 12)

    train = monthly_rev[:-3]
    test  = monthly_rev[-3:]
    feat_cols = ["period_idx", "sin_month", "cos_month"]

    model = LinearRegression()
    model.fit(train[feat_cols], train["order_total"])

    test_pred  = model.predict(test[feat_cols])
    mae = mean_absolute_error(test["order_total"], test_pred)

    last_idx    = monthly_rev["period_idx"].max()
    last_period = monthly_rev["period"].max()

    future_rows = []
    for i in range(1, 4):
        fp = last_period + i
        future_rows.append({
            "period_idx": last_idx + i,
            "month_num":  fp.month,
            "sin_month":  np.sin(2 * np.pi * fp.month / 12),
            "cos_month":  np.cos(2 * np.pi * fp.month / 12),
            "period_str": str(fp),
        })
    future_df   = pd.DataFrame(future_rows)
    future_pred = model.predict(future_df[feat_cols])

    col1, col2, col3 = st.columns(3)

    kpi_card(col1, "#172554", "#3b82f6", "📅",
             "Next month forecast", f"${future_pred[0]:,.0f}",
             "#60a5fa", "Projected revenue", "#93c5fd")

    kpi_card(col2, "#0f172a", "#a78bfa", "📅",
             "+2 months", f"${future_pred[1]:,.0f}",
             "#c4b5fd", "Projected revenue", "#ddd6fe")

    mae_color = "#22c55e" if mae < 10000 else \
                "#f59e0b" if mae < 25000 else "#ef4444"
    mae_bg    = "#052e16" if mae < 10000 else \
                "#1c1205" if mae < 25000 else "#2d0a0a"
    kpi_card(col3, mae_bg, mae_color, "📊",
             "Model MAE", f"${mae:,.0f}",
             mae_color,
             "🟢 Accurate"    if mae < 10000 else
             "🟡 Moderate"    if mae < 25000 else "🔴 Review model",
             mae_color)

    st.markdown("---")
    st.markdown('<p class="section-header">Revenue forecast — next 3 months</p>',
                unsafe_allow_html=True)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=monthly_rev["period_str"], y=monthly_rev["order_total"],
        mode="lines+markers", name="Actual",
        line=dict(color="#a78bfa", width=3),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.1)"
    ))
    fig_fc.add_trace(go.Scatter(
        x=list(test["period_str"]) + future_df["period_str"].tolist(),
        y=list(test_pred) + list(future_pred),
        mode="lines+markers", name="Forecast",
        line=dict(color="#f472b6", width=3, dash="dot"),
        marker=dict(size=10, symbol="triangle-up")
    ))
    fig_fc.add_vrect(
        x0=test["period_str"].iloc[0],
        x1=future_df["period_str"].iloc[-1],
        fillcolor="rgba(244,114,182,0.05)",
        line_width=0,
        annotation_text="Forecast zone",
        annotation_position="top left"
    )
    fig_fc.update_layout(
        template="plotly_dark", height=420,
        yaxis_title="Revenue ($)",
        xaxis_title="Month",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=20, b=40)
    )
    fig_fc.update_xaxes(tickangle=45)
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown('<p class="section-header">Forecast breakdown</p>',
                unsafe_allow_html=True)
    forecast_table = pd.DataFrame({
        "Month": future_df["period_str"],
        "Forecasted revenue": [f"${p:,.2f}" for p in future_pred],
        "vs last actual": [
            f"{((p / monthly_rev['order_total'].iloc[-1]) - 1) * 100:+.1f}%"
            for p in future_pred
        ]
    })
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)