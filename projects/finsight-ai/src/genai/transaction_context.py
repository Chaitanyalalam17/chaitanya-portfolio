
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANOMALY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "anomaly_results.csv"
)

SHAP_PATH = (
    PROJECT_ROOT / "data" / "processed" / "shap_explanations.csv"
)


def get_anomaly_summary():
    if not ANOMALY_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly results not found: {ANOMALY_PATH}"
        )

    df = pd.read_csv(ANOMALY_PATH)

    total = len(df)
    anomalies = int(df["is_anomaly"].sum())

    return {
        "total_transactions_scored": total,
        "anomalies_detected": anomalies,
        "anomaly_percentage": round(
            (anomalies / total) * 100, 2
        ) if total else 0,
        "note": (
            "Anomalies are unusual transactions, "
            "not confirmed fraud."
        ),
    }


def get_transaction_context(transaction_id: int):
    if not ANOMALY_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly results not found: {ANOMALY_PATH}"
        )

    anomalies = pd.read_csv(ANOMALY_PATH)

    matching = anomalies[
        anomalies["transaction_id"] == transaction_id
    ]

    if matching.empty:
        return {
            "transaction_id": transaction_id,
            "found": False,
        }

    row = matching.iloc[0]

    result = {
        "transaction_id": transaction_id,
        "found": True,
        "amount": float(row["amount"]),
        "transaction_type": str(row["transaction_type"]),
        "channel": str(row["channel"]),
        "is_anomaly": bool(row["is_anomaly"]),
        "anomaly_score": float(row["anomaly_score"]),
    }

    if SHAP_PATH.exists():
        shap_df = pd.read_csv(SHAP_PATH)

        shap_matching = shap_df[
            shap_df["transaction_id"] == transaction_id
        ]

        if not shap_matching.empty:
            shap_row = shap_matching.iloc[0]

            feature_values = {}

            for column in shap_df.columns:
                if column.startswith("shap_"):
                    feature_values[
                        column.replace("shap_", "", 1)
                    ] = float(shap_row[column])

            top_features = sorted(
                feature_values.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:5]

            result["top_shap_features"] = [
                {
                    "feature": feature,
                    "shap_value": value,
                }
                for feature, value in top_features
            ]

    return result


if __name__ == "__main__":
    print("FinSight AI Transaction Intelligence")
    print("-" * 40)

    print("\nAnomaly summary:")
    print(get_anomaly_summary())

    print("\nTransaction 39:")
    print(get_transaction_context(39))
