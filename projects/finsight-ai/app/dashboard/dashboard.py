
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FinSight AI | Financial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_FILE = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"


# ==========================================
# 2. LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    if not DB_FILE.exists():
        raise FileNotFoundError(
            "Database not found. Run the ingestion pipeline first."
        )

    with duckdb.connect(str(DB_FILE), read_only=True) as conn:
        data = conn.execute(
            "SELECT * FROM transactions"
        ).df()

    data["transaction_date"] = pd.to_datetime(
        data["transaction_date"]
    )

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load data: {error}")
    st.stop()


# ==========================================
# 3. HEADER
# ==========================================

st.title("📊 FinSight AI")
st.markdown("### Financial Intelligence Dashboard")

st.caption(
    "An interactive financial analytics platform powered "
    "by Python, DuckDB, Streamlit and Plotly. "
    "All transaction data is synthetic."
)

st.divider()


# ==========================================
# 4. SIDEBAR FILTERS
# ==========================================

st.sidebar.title("FinSight AI")
st.sidebar.header("Dashboard Filters")

min_date = df["transaction_date"].min().date()
max_date = df["transaction_date"].max().date()

selected_dates = st.sidebar.date_input(
    "Transaction Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

transaction_types = sorted(
    df["transaction_type"].dropna().unique()
)

selected_types = st.sidebar.multiselect(
    "Transaction Type",
    options=transaction_types,
    default=transaction_types
)

channels = sorted(df["channel"].dropna().unique())

selected_channels = st.sidebar.multiselect(
    "Transaction Channel",
    options=channels,
    default=channels
)


# ==========================================
# 5. APPLY FILTERS
# ==========================================

filtered_df = df[
    df["transaction_type"].isin(selected_types)
    & df["channel"].isin(selected_channels)
].copy()

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_df = filtered_df[
        filtered_df["transaction_date"].dt.date.between(
            start_date, end_date
        )
    ]

if filtered_df.empty:
    st.warning(
        "No transactions match the selected filters. "
        "Please adjust your filters."
    )
    st.stop()


# ==========================================
# 6. KPI METRICS
# ==========================================

total_transactions = len(filtered_df)
total_value = filtered_df["amount"].sum()
average_transaction = filtered_df["amount"].mean()
unique_customers = filtered_df["customer_id"].nunique()

st.subheader("Financial Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Transactions",
        value=f"{total_transactions:,}"
    )

with col2:
    st.metric(
        label="Total Transaction Value",
        value=f"{total_value:,.2f}"
    )

with col3:
    st.metric(
        label="Average Transaction",
        value=f"{average_transaction:,.2f}"
    )

with col4:
    st.metric(
        label="Unique Customers",
        value=f"{unique_customers:,}"
    )

st.divider()


# ==========================================
# 7. TRANSACTION TYPE AND CHANNEL CHARTS
# ==========================================

left, right = st.columns(2)

with left:
    st.subheader("Transaction Value by Type")

    type_summary = (
        filtered_df.groupby(
            "transaction_type", as_index=False
        )["amount"].sum()
    )

    fig_type = px.bar(
        type_summary,
        x="transaction_type",
        y="amount",
        color="transaction_type",
        labels={
            "transaction_type": "Transaction Type",
            "amount": "Total Value"
        },
        template="plotly_dark"
    )

    fig_type.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(
        fig_type,
        use_container_width=True
    )


with right:
    st.subheader("Transactions by Channel")

    channel_summary = (
        filtered_df.groupby(
            "channel", as_index=False
        )
        .size()
        .rename(columns={"size": "count"})
    )

    fig_channel = px.pie(
        channel_summary,
        names="channel",
        values="count",
        hole=0.55,
        template="plotly_dark"
    )

    fig_channel.update_layout(
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(
        fig_channel,
        use_container_width=True
    )


# ==========================================
# 8. DAILY TRANSACTION TREND
# ==========================================

st.divider()
st.subheader("Daily Transaction Trend")

daily_summary = (
    filtered_df.assign(
        date=filtered_df["transaction_date"].dt.date
    )
    .groupby("date", as_index=False)["amount"]
    .sum()
)

fig_daily = px.line(
    daily_summary,
    x="date",
    y="amount",
    markers=True,
    labels={
        "date": "Date",
        "amount": "Transaction Value"
    },
    template="plotly_dark"
)

st.plotly_chart(
    fig_daily,
    use_container_width=True
)


# ==========================================
# 9. MONTHLY TRANSACTION TREND
# ==========================================

st.divider()
st.subheader("Monthly Transaction Trend")

monthly_df = filtered_df.copy()

monthly_df["month"] = (
    monthly_df["transaction_date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly_summary = (
    monthly_df.groupby(
        "month", as_index=False
    )["amount"].sum()
)

fig_monthly = px.bar(
    monthly_summary,
    x="month",
    y="amount",
    labels={
        "month": "Month",
        "amount": "Transaction Value"
    },
    template="plotly_dark"
)

st.plotly_chart(
    fig_monthly,
    use_container_width=True
)


# ==========================================
# 10. TOP CUSTOMERS
# ==========================================

st.divider()
st.subheader("Top 10 Customers by Transaction Value")

top_customers = (
    filtered_df.groupby(
        "customer_id", as_index=False
    )["amount"]
    .sum()
    .sort_values("amount", ascending=False)
    .head(10)
)

top_customers["customer_id"] = (
    top_customers["customer_id"].astype(str)
)

fig_customers = px.bar(
    top_customers,
    x="customer_id",
    y="amount",
    color="amount",
    labels={
        "customer_id": "Customer ID",
        "amount": "Transaction Value"
    },
    template="plotly_dark",
    color_continuous_scale="Blues"
)

fig_customers.update_layout(
    coloraxis_showscale=False
)

st.plotly_chart(
    fig_customers,
    use_container_width=True
)


# ==========================================
# 11. DATA QUALITY OVERVIEW
# ==========================================

st.divider()
st.subheader("Data Quality Overview")

# Quality checks cover the complete loaded dataset,
# not just the transactions selected by filters.

missing_values = int(df.isnull().sum().sum())

duplicate_ids = int(
    df["transaction_id"].duplicated().sum()
)

invalid_amounts = int(
    (df["amount"] <= 0).sum()
)

q1, q2, q3 = st.columns(3)

q1.metric("Missing Values", missing_values)
q2.metric("Duplicate Transaction IDs", duplicate_ids)
q3.metric("Invalid Amounts", invalid_amounts)

if (
    missing_values == 0
    and duplicate_ids == 0
    and invalid_amounts == 0
):
    st.success("All configured data quality checks passed.")
else:
    st.warning("Data quality issues detected.")


# ==========================================
# 12. TRANSACTION RECORDS
# ==========================================

st.divider()
st.subheader("Transaction Records")

st.dataframe(
    filtered_df.sort_values(
        "transaction_date", ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 13. DOWNLOAD DATA
# ==========================================

st.download_button(
    label="Download Filtered Transactions",
    data=filtered_df.to_csv(index=False),
    file_name="finsight_transactions.csv",
    mime="text/csv"
)


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "FinSight AI | Developed by Chaitanya Lalam | "
    "Python • DuckDB • Streamlit • Plotly"
)
