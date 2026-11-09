
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.genai.transaction_context import (
    get_anomaly_summary,
    get_transaction_context,
)


ANOMALY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "anomaly_results.csv"
)


st.set_page_config(
    page_title="Transaction Explorer | FinSight AI",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Transaction Explorer")
st.caption(
    "Explore synthetic transactions, anomaly detection "
    "results and SHAP feature contributions."
)


@st.cache_data
def load_transactions():
    df = pd.read_csv(ANOMALY_PATH)

    required_columns = {
        "transaction_id",
        "amount",
        "transaction_type",
        "channel",
        "is_anomaly",
        "anomaly_score",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["is_anomaly"] = (
        df["is_anomaly"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    if df["is_anomaly"].isna().any():
        raise ValueError("Invalid anomaly flag values.")

    return df


try:
    df = load_transactions()
    summary = get_anomaly_summary()

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Unable to load transaction data: {error}")
    st.stop()


st.subheader("Transaction Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Transactions Scored",
    f"{summary['total_transactions_scored']:,}",
)

col2.metric(
    "Anomalies Detected",
    f"{summary['anomalies_detected']:,}",
)

col3.metric(
    "Anomaly Rate",
    f"{summary['anomaly_percentage']:.2f}%",
)


st.divider()

st.subheader("Filter Transactions")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    selected_types = st.multiselect(
        "Transaction Type",
        options=sorted(
            df["transaction_type"].dropna().unique()
        ),
        default=[],
    )

with filter_col2:
    selected_channels = st.multiselect(
        "Channel",
        options=sorted(
            df["channel"].dropna().unique()
        ),
        default=[],
    )

with filter_col3:
    anomaly_filter = st.selectbox(
        "Anomaly Status",
        options=[
            "All Transactions",
            "Anomalous Only",
            "Normal Only",
        ],
    )


filtered_df = df.copy()

if selected_types:
    filtered_df = filtered_df[
        filtered_df["transaction_type"].isin(
            selected_types
        )
    ]

if selected_channels:
    filtered_df = filtered_df[
        filtered_df["channel"].isin(
            selected_channels
        )
    ]

if anomaly_filter == "Anomalous Only":
    filtered_df = filtered_df[
        filtered_df["is_anomaly"]
    ]

elif anomaly_filter == "Normal Only":
    filtered_df = filtered_df[
        ~filtered_df["is_anomaly"]
    ]


st.metric(
    "Matching Transactions",
    f"{len(filtered_df):,}",
)

st.dataframe(
    filtered_df.sort_values("transaction_id"),
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Transactions (CSV)",
    data=csv_data,
    file_name="finsight_filtered_transactions.csv",
    mime="text/csv",
)


st.divider()

st.subheader("Top 10 Anomalous Transactions")

top_anomalies = (
    df[df["is_anomaly"]]
    .sort_values(
        "anomaly_score",
        ascending=False,
    )
    .head(10)
)

st.caption(
    "Higher anomaly scores indicate more unusual "
    "transactions in this saved model output."
)

st.dataframe(
    top_anomalies[
        [
            "transaction_id",
            "amount",
            "transaction_type",
            "channel",
            "anomaly_score",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


st.divider()

st.subheader("Analyse an Individual Transaction")

available_ids = sorted(
    filtered_df["transaction_id"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

if not available_ids:
    st.info(
        "No transactions match the current filters. "
        "Change the filters to select a transaction."
    )

else:
    default_index = (
        available_ids.index(39)
        if 39 in available_ids
        else 0
    )

    selected_id = st.selectbox(
        "Select Transaction ID",
        options=available_ids,
        index=default_index,
    )

    if st.button(
        "Analyse Transaction",
        type="primary",
    ):
        try:
            transaction = get_transaction_context(
                int(selected_id)
            )

            if not transaction["found"]:
                st.warning(
                    f"Transaction {selected_id} was not found."
                )

            else:
                st.subheader(
                    f"Transaction {selected_id} — Analysis"
                )

                detail_col1, detail_col2, detail_col3 = (
                    st.columns(3)
                )

                detail_col1.metric(
                    "Amount",
                    f"{transaction['amount']:,.2f}",
                )

                detail_col2.metric(
                    "Transaction Type",
                    transaction["transaction_type"],
                )

                detail_col3.metric(
                    "Channel",
                    transaction["channel"],
                )

                if transaction["is_anomaly"]:
                    st.warning(
                        "This transaction was flagged "
                        "as anomalous by Isolation Forest."
                    )
                else:
                    st.success(
                        "This transaction was not flagged "
                        "as anomalous."
                    )

                st.metric(
                    "Anomaly Score",
                    f"{transaction['anomaly_score']:.6f}",
                )

                features = transaction.get(
                    "top_shap_features", []
                )

                if features:
                    st.subheader(
                        "Top SHAP Feature Contributions"
                    )

                    shap_df = pd.DataFrame(features)

                    st.bar_chart(
                        shap_df,
                        x="feature",
                        y="shap_value",
                        horizontal=True,
                    )

                    st.dataframe(
                        shap_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.info(
                        "SHAP values represent contributions "
                        "to the model output, not fraud "
                        "probabilities."
                    )

                else:
                    st.info(
                        "No SHAP explanation is available "
                        "for this transaction."
                    )

        except (FileNotFoundError, ValueError, KeyError) as error:
            st.error(
                f"Unable to analyse transaction: {error}"
            )


st.divider()

st.caption(
    "FinSight AI uses synthetic financial data. "
    "Anomaly detection does not establish fraud. "
    "Transaction amounts have no specified currency."
)
