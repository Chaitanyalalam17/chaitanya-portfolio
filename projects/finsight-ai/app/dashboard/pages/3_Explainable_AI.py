
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="FinSight AI | Explainable AI",
    page_icon="🧠",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "processed" / "anomaly_results.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["is_anomaly"] = (
        df["is_anomaly"].astype(str).str.lower() == "true"
    )
    return df


st.title("🧠 FinSight AI")
st.header("Explainable AI - Anomaly Insights")

st.caption(
    "Explore unusual financial transactions using "
    "interpretable comparisons with historical data. "
    "All transactions are synthetic."
)

if not DATA_PATH.exists():
    st.error("Anomaly results not found. Train the model first.")
    st.stop()

df = load_data()

st.divider()
st.subheader("Anomaly Overview")

total = len(df)
anomalies = int(df["is_anomaly"].sum())
normal = total - anomalies
rate = anomalies / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Flagged Anomalies", f"{anomalies:,}")
c3.metric("Normal Transactions", f"{normal:,}")
c4.metric("Anomaly Rate", f"{rate:.2f}%")

st.divider()
st.subheader("Explore a Flagged Transaction")

flagged = df[df["is_anomaly"]].copy()

if flagged.empty:
    st.info("No anomalies found.")
    st.stop()

transaction_id = st.selectbox(
    "Select Transaction ID",
    flagged["transaction_id"].tolist(),
)

selected = flagged.loc[
    flagged["transaction_id"] == transaction_id
].iloc[0]

st.write("### Transaction Details")

col1, col2, col3 = st.columns(3)

col1.metric("Transaction Amount", f"{selected['amount']:,.2f}")
col2.metric("Anomaly Score", f"{selected['anomaly_score']:.4f}")
col3.metric("Transaction Type", selected["transaction_type"])

st.write(f"**Channel:** {selected['channel']}")
st.write("**Status:** Flagged as unusual")

st.divider()
st.subheader("Why Might This Transaction Be Unusual?")

median_amount = df["amount"].median()

type_median = df.loc[
    df["transaction_type"] == selected["transaction_type"],
    "amount"
].median()

channel_median = df.loc[
    df["channel"] == selected["channel"],
    "amount"
].median()

comparison = pd.DataFrame({
    "Category": [
        "Selected Transaction",
        "Overall Median",
        "Transaction Type Median",
        "Channel Median",
    ],
    "Amount": [
        selected["amount"],
        median_amount,
        type_median,
        channel_median,
    ],
})

fig = px.bar(
    comparison,
    x="Category",
    y="Amount",
    color="Category",
    title="Transaction Amount Comparison",
)

st.plotly_chart(fig, width="stretch")

if selected["amount"] > median_amount * 2:
    st.warning(
        "This transaction amount is more than twice "
        "the overall median transaction amount."
    )
elif selected["amount"] < median_amount * 0.5:
    st.info(
        "This transaction amount is less than half "
        "the overall median transaction amount."
    )
else:
    st.info(
        "The amount is relatively close to the typical "
        "range. Other transaction characteristics may "
        "contribute to the model's anomaly assessment."
    )

st.caption(
    "These comparisons provide descriptive context. "
    "They are not model-derived feature contributions "
    "or proof of fraud."
)

st.divider()
st.subheader("Anomaly Score Distribution")

score_fig = px.histogram(
    df,
    x="anomaly_score",
    color="is_anomaly",
    nbins=35,
    title="Distribution of Model Anomaly Scores",
    labels={
        "anomaly_score": "Anomaly Score",
        "is_anomaly": "Flagged Anomaly",
    },
)

st.plotly_chart(score_fig, width="stretch")

st.divider()
st.subheader("Flagged Transactions")

st.dataframe(
    flagged[
        [
            "transaction_id",
            "amount",
            "transaction_type",
            "channel",
            "anomaly_score",
        ]
    ].sort_values("anomaly_score"),
    width="stretch",
    hide_index=True,
)

st.caption(
    "Note: Isolation Forest scores depend on the "
    "scoring method used during model training. "
    "These visualisations do not establish causality."
)
