
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_FILE = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide"
)

st.title("FinSight AI")
st.subheader("Financial Intelligence Dashboard")

if not DB_FILE.exists():
    st.error("Database not found. Run the ingestion pipeline first.")
    st.stop()

with duckdb.connect(str(DB_FILE), read_only=True) as conn:
    df = conn.execute("SELECT * FROM transactions").df()

df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# Sidebar filters
st.sidebar.header("Filters")

selected_types = st.sidebar.multiselect(
    "Transaction Type",
    options=sorted(df["transaction_type"].unique()),
    default=sorted(df["transaction_type"].unique())
)

selected_channels = st.sidebar.multiselect(
    "Channel",
    options=sorted(df["channel"].unique()),
    default=sorted(df["channel"].unique())
)

filtered_df = df[
    df["transaction_type"].isin(selected_types)
    & df["channel"].isin(selected_channels)
].copy()

# KPIs
total_transactions = len(filtered_df)
total_amount = filtered_df["amount"].sum()
average_amount = filtered_df["amount"].mean()
unique_customers = filtered_df["customer_id"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions", f"{total_transactions:,}")
col2.metric("Total Value", f"{total_amount:,.2f}")
col3.metric(
    "Average Transaction",
    f"{average_amount:,.2f}" if pd.notna(average_amount) else "N/A"
)
col4.metric("Unique Customers", f"{unique_customers:,}")

st.divider()

# Charts
left, right = st.columns(2)

with left:
    st.subheader("Transaction Value by Type")
    type_summary = (
        filtered_df.groupby("transaction_type", as_index=False)["amount"]
        .sum()
    )

    fig_type = px.bar(
        type_summary,
        x="transaction_type",
        y="amount",
        color="transaction_type",
        labels={
            "transaction_type": "Transaction Type",
            "amount": "Total Value"
        }
    )
    st.plotly_chart(fig_type, use_container_width=True)

with right:
    st.subheader("Transactions by Channel")
    channel_summary = (
        filtered_df.groupby("channel", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    fig_channel = px.pie(
        channel_summary,
        names="channel",
        values="count",
        hole=0.45
    )
    st.plotly_chart(fig_channel, use_container_width=True)

# Daily trend
st.subheader("Daily Transaction Trend")

daily_summary = (
    filtered_df.assign(
        date=filtered_df["transaction_date"].dt.date
    )
    .groupby("date", as_index=False)["amount"]
    .sum()
)

fig_trend = px.line(
    daily_summary,
    x="date",
    y="amount",
    markers=True,
    labels={"date": "Date", "amount": "Transaction Value"}
)

st.plotly_chart(fig_trend, use_container_width=True)

# Transaction table
st.subheader("Transaction Records")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download Filtered CSV",
    data=filtered_df.to_csv(index=False),
    file_name="finsight_transactions.csv",
    mime="text/csv"
)
