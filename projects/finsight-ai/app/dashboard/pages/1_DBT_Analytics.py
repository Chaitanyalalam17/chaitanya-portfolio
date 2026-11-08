
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.dbt_analytics import load_dbt_analytics

st.set_page_config(
    page_title="FinSight AI | DBT Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 FinSight AI")
st.header("DBT Analytics Dashboard")
st.caption(
    "Financial analytics powered by DuckDB, dbt, Python and Streamlit."
)

@st.cache_data(ttl=300)
def get_analytics():
    return load_dbt_analytics()

try:
    analytics = get_analytics()
except Exception as exc:
    st.error(f"Unable to load analytics: {exc}")
    st.info("Run 'dbt run' inside dbt_project first.")
    st.stop()

daily = analytics["daily"].copy()
channels = analytics["channels"].copy()
customers = analytics["customers"].copy()

# Financial KPIs
st.subheader("Financial Overview")

total_transactions = int(daily["total_transactions"].sum())
total_value = float(daily["total_transaction_value"].sum())
avg_value = (
    total_value / total_transactions
    if total_transactions else 0
)

c1, c2, c3 = st.columns(3)
c1.metric("Total Transactions", f"{total_transactions:,}")
c2.metric("Total Transaction Value", f"{total_value:,.2f}")
c3.metric("Average Transaction", f"{avg_value:,.2f}")

st.divider()

# Daily trends
st.subheader("Daily Financial Performance")

daily["transaction_day"] = pd.to_datetime(
    daily["transaction_day"]
)

fig_daily = px.line(
    daily,
    x="transaction_day",
    y="total_transaction_value",
    markers=True,
    title="Daily Transaction Value",
)

st.plotly_chart(fig_daily, width="stretch")

# Channel analytics
st.subheader("Channel Performance")

fig_channels = px.bar(
    channels,
    x="channel",
    y="total_transaction_value",
    color="channel",
    title="Transaction Value by Channel",
)

st.plotly_chart(fig_channels, width="stretch")

# Customer analytics
st.subheader("Top 10 Customers")

top_customers = customers.sort_values(
    "total_transaction_value",
    ascending=False,
).head(10).copy()

top_customers["customer_id"] = (
    top_customers["customer_id"].astype(str)
)

fig_customers = px.bar(
    top_customers,
    x="customer_id",
    y="total_transaction_value",
    color="total_transaction_value",
    title="Top 10 Customers by Transaction Value",
)

st.plotly_chart(fig_customers, width="stretch")

st.subheader("Analytics Data Tables")

with st.expander("Daily Analytics"):
    st.dataframe(daily, width="stretch")

with st.expander("Channel Analytics"):
    st.dataframe(channels, width="stretch")

with st.expander("Customer Analytics"):
    st.dataframe(customers, width="stretch")
