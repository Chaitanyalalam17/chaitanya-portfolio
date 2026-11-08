from pydantic import BaseModel
from src.genai.rag_assistant import ask_finsight

from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"

app = FastAPI(
    title="FinSight AI API",
    description="Financial Intelligence and Anomaly Detection API",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "application": "FinSight AI",
        "status": "running",
        "message": "Welcome to the FinSight AI API",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FinSight AI Backend",
    }


@app.get("/analytics/overview")
def financial_overview():
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Financial database is unavailable.",
        )

    try:
        with duckdb.connect(str(DB_PATH), read_only=True) as con:
            result = con.execute(
                """
                SELECT
                    COUNT(*) AS total_transactions,
                    COALESCE(SUM(amount), 0) AS total_value,
                    COALESCE(AVG(amount), 0) AS average_transaction,
                    COUNT(DISTINCT customer_id) AS unique_customers
                FROM transactions
                """
            ).fetchone()

        return {
            "total_transactions": result[0],
            "total_transaction_value": round(result[1], 2),
            "average_transaction": round(result[2], 2),
            "unique_customers": result[3],
        }

    except duckdb.Error:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve financial analytics.",
        )

import pandas as pd

ANOMALY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "anomaly_results.csv"
)


def load_anomaly_data():
    if not ANOMALY_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Anomaly results are unavailable. Train the model first.",
        )

    try:
        return pd.read_csv(ANOMALY_PATH)
    except (OSError, pd.errors.ParserError):
        raise HTTPException(
            status_code=500,
            detail="Unable to read anomaly results.",
        )


@app.get("/anomalies")
def get_anomalies(limit: int = 20):
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=422,
            detail="Limit must be between 1 and 100.",
        )

    df = load_anomaly_data()
    anomalies = df[df["is_anomaly"] == True]

    return {
        "total_anomalies": len(anomalies),
        "returned": min(limit, len(anomalies)),
        "transactions": anomalies.head(limit).to_dict(
            orient="records"
        ),
    }


@app.get("/anomalies/{transaction_id}")
def get_transaction_anomaly(transaction_id: int):
    df = load_anomaly_data()

    transaction = df[
        df["transaction_id"] == transaction_id
    ]

    if transaction.empty:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found.",
        )

    return transaction.iloc[0].to_dict()

SHAP_PATH = (
    PROJECT_ROOT / "data" / "processed" / "shap_explanations.csv"
)


@app.get("/explanations/{transaction_id}")
def get_transaction_explanation(transaction_id: int):
    if not SHAP_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="SHAP explanations are unavailable. Generate them first.",
        )

    try:
        shap_df = pd.read_csv(SHAP_PATH)
    except (OSError, pd.errors.ParserError):
        raise HTTPException(
            status_code=500,
            detail="Unable to read SHAP explanations.",
        )

    transaction = shap_df[
        shap_df["transaction_id"] == transaction_id
    ]

    if transaction.empty:
        raise HTTPException(
            status_code=404,
            detail="Transaction explanation not found.",
        )

    shap_values = transaction.iloc[0].drop(
        labels=["transaction_id"]
    )

    contributions = {
        feature.replace("shap_", "", 1): float(value)
        for feature, value in shap_values.items()
    }

    sorted_contributions = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    return {
        "transaction_id": transaction_id,
        "explanation_method": "Tree SHAP",
        "feature_contributions": contributions,
        "top_features": [
            {
                "feature": feature,
                "shap_value": value,
            }
            for feature, value in sorted_contributions[:5]
        ],
    }

class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat_with_finsight(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    try:
        result = ask_finsight(question)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "FinSight AI assistant is unavailable. "
                "Check Ollama and the knowledge base."
            ),
        ) from error
