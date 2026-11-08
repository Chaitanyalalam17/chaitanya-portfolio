
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"
MODEL_DIR = PROJECT_ROOT / "models"


def load_data():
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        df = con.execute("""
            SELECT transaction_id, amount, transaction_type, channel
            FROM transactions
        """).df()

    return df


def prepare_features(df):
    feature_columns = joblib.load(
        MODEL_DIR / "feature_columns.joblib"
    )

    features = pd.get_dummies(
        df[["amount", "transaction_type", "channel"]],
        columns=["transaction_type", "channel"],
        dtype=int,
    )

    features = features.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    scaler = joblib.load(
        MODEL_DIR / "anomaly_scaler.joblib"
    )

    scaled_features = scaler.transform(features)

    return pd.DataFrame(
        scaled_features,
        columns=feature_columns,
        index=df.index,
    )


def generate_shap_explanations():
    model = joblib.load(
        MODEL_DIR / "isolation_forest.joblib"
    )

    df = load_data()

    if df.empty:
        raise ValueError("No transaction data found.")

    X = prepare_features(df)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_array = np.asarray(shap_values)

    if shap_array.shape != X.shape:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_array.shape}"
        )

    shap_df = pd.DataFrame(
        shap_array,
        columns=[
            f"shap_{column}" for column in X.columns
        ],
    )

    shap_df.insert(
        0,
        "transaction_id",
        df["transaction_id"].values,
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "shap_explanations.csv"
    )

    shap_df.to_csv(output_path, index=False)

    print("SHAP explanations generated successfully!")
    print(f"Transactions explained: {len(shap_df)}")
    print(f"Features explained: {len(X.columns)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    generate_shap_explanations()
