
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


st.set_page_config(
    page_title="Transaction Explorer | FinSight AI",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Transaction Explorer")
st.caption(
    "Explore synthetic financial transactions, "
    "anomaly detection results and SHAP explanations."
)

try:
    summary = get_anomaly_summary()

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

except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"Unable to load anomaly summary: {error}")
    st.stop()


st.divider()

st.subheader("Search for a Transaction")

transaction_id = st.number_input(
    "Transaction ID",
    min_value=1,
    value=39,
    step=1,
)

if st.button(
    "Analyse Transaction",
    type="primary",
):
    try:
        transaction = get_transaction_context(
            int(transaction_id)
        )

        if not transaction["found"]:
            st.warning(
                f"Transaction {transaction_id} was not found."
            )

        else:
            st.subheader(
                f"Transaction {transaction_id} — Analysis"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Transaction Amount",
                f"{transaction['amount']:,.2f}",
            )

            col2.metric(
                "Transaction Type",
                transaction["transaction_type"],
            )

            col3.metric(
                "Channel",
                transaction["channel"],
            )

            if transaction["is_anomaly"]:
                st.warning(
                    "This transaction was flagged as anomalous "
                    "by the Isolation Forest model."
                )
            else:
                st.success(
                    "This transaction was not flagged as anomalous."
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
                    "SHAP values explain contributions to the "
                    "model output. They are not fraud probabilities."
                )

            else:
                st.info(
                    "No SHAP explanations are available "
                    "for this transaction."
                )

    except (FileNotFoundError, KeyError, ValueError) as error:
        st.error(
            f"Unable to analyse transaction: {error}"
        )


st.divider()

st.caption(
    "FinSight AI uses synthetic financial data. "
    "Anomaly flags do not establish fraudulent activity. "
    "No currency is assumed for transaction amounts."
)
