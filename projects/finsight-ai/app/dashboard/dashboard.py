
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
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_FILE = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"

PRIMARY = "#38BDF8"
SECONDARY = "#818CF8"
BACKGROUND = "#0B1220"
CARD = "#172338"
TEXT = "#E2E8F0"


# ==========================================
# 2. CUSTOM DASHBOARD STYLING
# ==========================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    div[data-testid="stMetric"] {
        background: #172338;
        border: 1px solid #29364D;
        border-radius: 14px;
        padding: 20px;
    }

    div[data-testid="stMetricLabel"] {
        color: #A7B5CC;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
    }

    div[data-testid="stPlotlyChart"] {
        border: 1px solid #29364D;
        border-radius: 14px;
        padding: 8px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 3. LOAD DATA
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

    required_columns = {
        "transaction_id",
        "transaction_date",
        "transaction_type",
        "channel",
        "customer_id",
        "amount",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing database columns: {sorted(missing)}"
        )

    data["transaction_date"] = pd.to_datetime(
        data["transaction_date"],
        errors="coerce",
    )

    return data


try:
    df = load_data()

except (FileNotFoundError, ValueError, duckdb.Error) as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.stop()


# ==========================================
# 4. HEADER
# ==========================================

st.title("📊 FinSight AI")
st.markdown(
    "### Financial Intelligence & Analytics Platform"
)

st.caption(
    "An end-to-end financial intelligence platform "
    "built with Python, DuckDB, dbt, machine learning, "
    "explainable AI and local generative AI."
)

st.info(
    "Demo environment: All financial transactions are "
    "synthetic. Transaction values have no specified currency."
)

st.divider()


# ==========================================
# 5. SIDEBAR FILTERS
# ==========================================

st.sidebar.title("FinSight AI")
st.sidebar.caption("Financial Intelligence Platform")
st.sidebar.divider()

st.sidebar.subheader("Dashboard Filters")

valid_dates = df["transaction_date"].dropna()

if valid_dates.empty:
    st.error("No valid transaction dates were found.")
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

selected_dates = st.sidebar.date_input(
    "Transaction Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

transaction_types = sorted(
    df["transaction_type"].dropna().unique()
)

selected_types = st.sidebar.multiselect(
    "Transaction Type",
    options=transaction_types,
    default=transaction_types,
)

channels = sorted(
    df["channel"].dropna().unique()
)

selected_channels = st.sidebar.multiselect(
    "Transaction Channel",
    options=channels,
    default=channels,
)

st.sidebar.divider()
st.sidebar.caption(
    "FinSight AI • Portfolio demonstration"
)


# ==========================================
# 6. APPLY FILTERS
# ==========================================

filtered_df = df[
    df["transaction_type"].isin(selected_types)
    & df["channel"].isin(selected_channels)
].copy()

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_df = filtered_df[
        filtered_df["transaction_date"].dt.date.between(
            start_date,
            end_date,
        )
    ]

if filtered_df.empty:
    st.warning(
        "No transactions match the selected filters. "
        "Please adjust your filters."
    )
    st.stop()


# ==========================================
# 7. KPI METRICS
# ==========================================

st.subheader("Financial Overview")

total_transactions = len(filtered_df)
total_value = filtered_df["amount"].sum()
average_transaction = filtered_df["amount"].mean()
unique_customers = filtered_df["customer_id"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}",
)

col2.metric(
    "Total Transaction Value",
    f"{total_value:,.2f}",
)

col3.metric(
    "Average Transaction",
    f"{average_transaction:,.2f}",
)

col4.metric(
    "Unique Customers",
    f"{unique_customers:,}",
)

st.divider()


# ==========================================
# 8. CHART HELPER
# ==========================================

def style_chart(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(
            family="Arial",
            color=TEXT,
            size=12,
        ),
        height=height,
        margin=dict(
            l=15,
            r=15,
            t=25,
            b=20,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
    )

    fig.update_yaxes(
        gridcolor="#263349",
        zeroline=False,
    )

    return fig


# ==========================================
# 9. TRANSACTION TYPE & CHANNEL
# ==========================================

left, right = st.columns(2)

with left:
    st.subheader("Transaction Value by Type")

    type_summary = (
        filtered_df.groupby(
            "transaction_type",
            as_index=False,
        )["amount"].sum()
        .sort_values("amount", ascending=False)
    )

    fig_type = px.bar(
        type_summary,
        x="transaction_type",
        y="amount",
        color="transaction_type",
        color_discrete_sequence=[
            PRIMARY,
            SECONDARY,
            "#2DD4BF",
            "#FBBF24",
            "#FB7185",
        ],
        labels={
            "transaction_type": "Transaction Type",
            "amount": "Total Transaction Value",
        },
    )

    fig_type.update_layout(
        showlegend=False,
    )

    st.plotly_chart(
        style_chart(fig_type),
        use_container_width=True,
    )


with right:
    st.subheader("Transactions by Channel")

    channel_summary = (
        filtered_df.groupby(
            "channel",
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )

    fig_channel = px.pie(
        channel_summary,
        names="channel",
        values="count",
        hole=0.60,
        color_discrete_sequence=[
            PRIMARY,
            SECONDARY,
            "#2DD4BF",
            "#FBBF24",
            "#FB7185",
        ],
    )

    fig_channel.update_traces(
        textinfo="percent+label",
    )

    st.plotly_chart(
        style_chart(fig_channel),
        use_container_width=True,
    )


# ==========================================
# 10. DAILY TRANSACTION TREND
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
        "amount": "Transaction Value",
    },
)

fig_daily.update_traces(
    line=dict(
        color=PRIMARY,
        width=3,
    ),
    marker=dict(size=6),
)

st.plotly_chart(
    style_chart(fig_daily, height=400),
    use_container_width=True,
)


# ==========================================
# 11. MONTHLY TRANSACTION TREND
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
        "month",
        as_index=False,
    )["amount"].sum()
)

fig_monthly = px.bar(
    monthly_summary,
    x="month",
    y="amount",
    color_discrete_sequence=[SECONDARY],
    labels={
        "month": "Month",
        "amount": "Transaction Value",
    },
)

st.plotly_chart(
    style_chart(fig_monthly),
    use_container_width=True,
)


# ==========================================
# 12. TOP CUSTOMERS
# ==========================================

st.divider()
st.subheader("Top 10 Customers by Transaction Value")

top_customers = (
    filtered_df.groupby(
        "customer_id",
        as_index=False,
    )["amount"]
    .sum()
    .sort_values(
        "amount",
        ascending=False,
    )
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
    color_continuous_scale=[
        "#1D4ED8",
        PRIMARY,
    ],
    labels={
        "customer_id": "Customer ID",
        "amount": "Transaction Value",
    },
)

fig_customers.update_layout(
    coloraxis_showscale=False,
)

st.plotly_chart(
    style_chart(fig_customers),
    use_container_width=True,
)


# ==========================================
# 13. DATA QUALITY OVERVIEW
# ==========================================

st.divider()
st.subheader("Data Quality Overview")

missing_values = int(
    df.isnull().sum().sum()
)

duplicate_ids = int(
    df["transaction_id"].duplicated().sum()
)

invalid_amounts = int(
    (df["amount"] <= 0).sum()
)

q1, q2, q3 = st.columns(3)

q1.metric(
    "Missing Values",
    missing_values,
)

q2.metric(
    "Duplicate Transaction IDs",
    duplicate_ids,
)

q3.metric(
    "Invalid Amounts",
    invalid_amounts,
)

if (
    missing_values == 0
    and duplicate_ids == 0
    and invalid_amounts == 0
):
    st.success(
        "All configured data quality checks passed."
    )
else:
    st.warning(
        "Data quality issues were detected."
    )


# ==========================================
# 14. TRANSACTION RECORDS
# ==========================================

st.divider()
st.subheader("Transaction Records")

st.dataframe(
    filtered_df.sort_values(
        "transaction_date",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)


# ==========================================
# 15. DOWNLOAD DATA
# ==========================================

st.download_button(
    label="📥 Download Filtered Transactions",
    data=filtered_df.to_csv(index=False),
    file_name="finsight_transactions.csv",
    mime="text/csv",
)


# ==========================================
# 16. FOOTER
# ==========================================

st.divider()

st.caption(
    "FinSight AI | Developed by Chaitanya Lalam | "
    "Python • DuckDB • dbt • Streamlit • Plotly • ML • SHAP"
)
