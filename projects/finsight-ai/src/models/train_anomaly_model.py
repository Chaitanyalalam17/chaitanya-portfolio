
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"
MODEL_DIR = PROJECT_ROOT / "models"


def train_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        df = con.execute(
            """
            SELECT
                transaction_id,
                amount,
                transaction_type,
                channel
            FROM transactions
            """
        ).df()

    if df.empty:
        raise ValueError("No transactions found in the database.")

    features = pd.get_dummies(
        df[["amount", "transaction_type", "channel"]],
        columns=["transaction_type", "channel"],
        dtype=int,
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )

    predictions = model.fit_predict(X)

    df["anomaly_prediction"] = predictions
    df["is_anomaly"] = predictions == -1
    df["anomaly_score"] = -model.score_samples(X)

    joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, MODEL_DIR / "anomaly_scaler.joblib")
    joblib.dump(features.columns.tolist(), MODEL_DIR / "feature_columns.joblib")

    output_path = PROJECT_ROOT / "data" / "processed" / "anomaly_results.csv"
    df.to_csv(output_path, index=False)

    print("Anomaly detection model trained successfully!")
    print(f"Total transactions: {len(df)}")
    print(f"Flagged anomalies: {df['is_anomaly'].sum()}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    train_model()
