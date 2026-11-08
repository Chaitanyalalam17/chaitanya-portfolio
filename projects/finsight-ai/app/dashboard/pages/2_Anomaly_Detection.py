
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="FinSight AI | Anomaly Detection",
    page_icon="🔍",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "anomaly_results.csv"
)

st.title("🔍 FinSight AI")
st.header("AI-Powered Anomaly Detection")

st.caption(
    "Isolation Forest machine learning model for identifying "
    "unusual financial transactions in synthetic data."
)

if not DATA_PATH.exists():
    st.error(
        "Anomaly results not found. Run "
        "python src/models/train_anomaly_model.py first."
    )
    st.stop()


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data(DATA_PATH)

df["is_anomaly"] = (
    df["is_anomaly"].astype(str).str.lower() == "true"
)

df["status"] = df["is_anomaly"].map(
    {True: "Anomaly", False: "Normal"}
)

total = len(df)
anomalies = int(df["is_anomaly"].sum())
normal = total - anomalies
rate = anomalies / total * 100 if total else 0

st.divider()
st.subheader("Detection Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total:,}")
col2.metric("Flagged Anomalies", f"{anomalies:,}")
col3.metric("Normal Transactions", f"{normal:,}")
col4.metric("Anomaly Rate", f"{rate:.2f}%")

st.divider()

st.subheader("Transaction Anomaly Analysis")

left, right = st.columns(2)

with left:
    counts = (
        df["status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Count")
    )

    fig = px.pie(
        counts,
        names="Status",
        values="Count",
        hole=0.55,
        title="Normal vs Anomalous Transactions",
        color="Status",
        color_discrete_map={
            "Normal": "#6366F1",
            "Anomaly": "#EF4444",
        },
    )

    st.plotly_chart(fig, width="stretch")

with right:
    fig = px.scatter(
        df,
        x="amount",
        y="anomaly_score",
        color="status",
        title="Transaction Amount vs Anomaly Score",
        hover_data=["transaction_id", "channel"],
        color_discrete_map={
            "Normal": "#6366F1",
            "Anomaly": "#EF4444",
        },
    )

    st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("Anomalies by Transaction Type")

anomaly_df = df[df["is_anomaly"]]

if not anomaly_df.empty:
    fig = px.histogram(
        anomaly_df,
        x="transaction_type",
        color="transaction_type",
        title="Flagged Transactions by Type",
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No anomalies found.")

st.divider()

st.subheader("Explore Transactions")

selection = st.selectbox(
    "Filter transactions",
    ["All Transactions", "Anomalies Only", "Normal Only"],
)

if selection == "Anomalies Only":
    filtered = df[df["is_anomaly"]]
elif selection == "Normal Only":
    filtered = df[~df["is_anomaly"]]
else:
    filtered = df

st.dataframe(
    filtered.sort_values(
        "anomaly_score", ascending=False
    ),
    width="stretch",
    hide_index=True,
)

st.download_button(
    "Download Filtered Transactions (CSV)",
    data=filtered.to_csv(index=False),
    file_name="finsight_anomaly_results.csv",
    mime="text/csv",
)

st.caption(
    "Note: Flagged anomalies are statistical outliers, "
    "not verified fraudulent transactions."
)
