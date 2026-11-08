
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="SHAP Explanations | FinSight AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 FinSight AI")
st.header("SHAP Model Explanations")
st.caption(
    "Explainable anomaly detection using SHAP and Isolation Forest. "
    "All transaction data is synthetic."
)


@st.cache_data
def load_data():
    transactions = pd.read_csv(
        DATA_DIR / "anomaly_results.csv"
    )
    explanations = pd.read_csv(
        DATA_DIR / "shap_explanations.csv"
    )

    return transactions.merge(
        explanations,
        on="transaction_id",
        validate="one_to_one",
    )


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(
        MODEL_DIR / "isolation_forest.joblib"
    )
    scaler = joblib.load(
        MODEL_DIR / "anomaly_scaler.joblib"
    )
    feature_columns = joblib.load(
        MODEL_DIR / "feature_columns.joblib"
    )
    return model, scaler, feature_columns


try:
    df = load_data()
    model, scaler, feature_columns = load_model_artifacts()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Unable to load SHAP data: {exc}")
    st.stop()

shap_columns = [f"shap_{name}" for name in feature_columns]

st.divider()
st.subheader("Model Explanation Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions Explained", f"{len(df):,}")
col2.metric("Model Features", len(feature_columns))
col3.metric("Flagged Anomalies", int(df["is_anomaly"].sum()))
col4.metric("Explanation Method", "Tree SHAP")

st.divider()

st.subheader("Global Feature Importance")

importance = (
    df[shap_columns]
    .abs()
    .mean()
    .sort_values(ascending=True)
)

importance_df = pd.DataFrame({
    "Feature": [
        name.removeprefix("shap_")
        for name in importance.index
    ],
    "Mean Absolute SHAP": importance.values,
})

fig = px.bar(
    importance_df,
    x="Mean Absolute SHAP",
    y="Feature",
    orientation="h",
    title="Average Absolute Feature Contribution",
    color="Mean Absolute SHAP",
    color_continuous_scale="Blues",
)

fig.update_layout(
    height=480,
    coloraxis_showscale=False,
)

st.plotly_chart(fig, width="stretch")

st.info(
    "Global importance shows the average magnitude of each "
    "feature's contribution to the Isolation Forest model output."
)

st.divider()

st.subheader("Individual Transaction Explanation")

flagged_only = st.checkbox(
    "Show flagged anomalies only",
    value=True,
)

selection_df = (
    df[df["is_anomaly"]]
    if flagged_only
    else df
)

transaction_id = st.selectbox(
    "Select Transaction ID",
    selection_df["transaction_id"].tolist(),
)

selected = df.loc[
    df["transaction_id"] == transaction_id
].iloc[0]

c1, c2, c3 = st.columns(3)

c1.metric("Transaction Amount", f"{selected['amount']:,.2f}")
c2.metric("Anomaly Score", f"{selected['anomaly_score']:.4f}")
c3.metric(
    "Model Classification",
    "Anomaly" if selected["is_anomaly"] else "Normal",
)

st.write(
    f"**Transaction Type:** {selected['transaction_type']}  "
    f"| **Channel:** {selected['channel']}"
)

st.subheader("Feature Contributions")

contributions = pd.DataFrame({
    "Feature": feature_columns,
    "SHAP Value": [
        selected[f"shap_{name}"]
        for name in feature_columns
    ],
})

contributions = contributions.sort_values(
    "SHAP Value"
)

fig_local = px.bar(
    contributions,
    x="SHAP Value",
    y="Feature",
    orientation="h",
    color="SHAP Value",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    title=f"SHAP Contributions — Transaction {transaction_id}",
)

fig_local.update_layout(
    height=500,
    coloraxis_showscale=False,
)

fig_local.add_vline(
    x=0,
    line_dash="dash",
    line_color="gray",
)

st.plotly_chart(fig_local, width="stretch")

st.caption(
    "These contributions explain the model's internal scoring "
    "output. They are not fraud probabilities or proof of fraud."
)

st.divider()

st.subheader("SHAP Waterfall Explanation")

features = pd.get_dummies(
    pd.DataFrame([{
        "amount": selected["amount"],
        "transaction_type": selected["transaction_type"],
        "channel": selected["channel"],
    }]),
    columns=["transaction_type", "channel"],
    dtype=int,
)

features = features.reindex(
    columns=feature_columns,
    fill_value=0,
)

X = scaler.transform(features)

explainer = shap.TreeExplainer(model)
base_value = float(
    np.asarray(explainer.expected_value).reshape(-1)[0]
)

shap_values = np.array([
    selected[f"shap_{name}"]
    for name in feature_columns
])

ordered = np.argsort(np.abs(shap_values))[::-1]
ordered_features = [
    feature_columns[i] for i in ordered
]
ordered_values = shap_values[ordered]

fig_waterfall = go.Figure(
    go.Waterfall(
        orientation="h",
        measure=["relative"] * len(ordered_values)
        + ["total"],
        y=ordered_features + ["Model Output"],
        x=ordered_values.tolist() + [0],
        base=base_value,
        connector={"line": {"color": "gray"}},
    )
)

fig_waterfall.update_layout(
    title="SHAP Contribution Waterfall",
    height=550,
    xaxis_title="Isolation Forest Internal Model Output",
)

st.plotly_chart(fig_waterfall, width="stretch")

st.caption(
    "The baseline and SHAP values are expressed in the "
    "Isolation Forest's internal Tree SHAP output space. "
    "They do not represent the displayed anomaly score directly."
)

st.divider()

st.subheader("Download Explanations")

st.download_button(
    "Download SHAP Explanations CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="finsight_shap_explanations.csv",
    mime="text/csv",
)
