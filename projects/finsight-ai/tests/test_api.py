
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_financial_overview():
    response = client.get("/analytics/overview")

    assert response.status_code == 200

    data = response.json()

    assert "total_transactions" in data
    assert "total_transaction_value" in data
    assert "average_transaction" in data
    assert "unique_customers" in data


def test_anomalies():
    response = client.get("/anomalies")

    assert response.status_code == 200

    data = response.json()

    assert "total_anomalies" in data
    assert "transactions" in data
    assert isinstance(data["transactions"], list)


def test_transaction_anomaly():
    response = client.get("/anomalies/39")

    assert response.status_code == 200
    assert response.json()["transaction_id"] == 39


def test_shap_explanation():
    response = client.get("/explanations/39")

    assert response.status_code == 200

    data = response.json()

    assert data["transaction_id"] == 39
    assert data["explanation_method"] == "Tree SHAP"
    assert len(data["feature_contributions"]) == 9
    assert len(data["top_features"]) == 5


def test_missing_explanation():
    response = client.get("/explanations/999999")

    assert response.status_code == 404

from unittest.mock import patch


def test_chat_success():
    mock_result = {
        "question": "Which model does FinSight AI use?",
        "answer": "FinSight AI uses Isolation Forest.",
        "sources": ["finsight_overview.md"],
    }

    with patch(
        "app.api.main.ask_finsight",
        return_value=mock_result,
    ):
        response = client.post(
            "/chat",
            json={
                "question": "Which model does FinSight AI use?"
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "FinSight AI uses Isolation Forest."
    )
    assert "finsight_overview.md" in response.json()["sources"]


def test_chat_empty_question():
    response = client.post(
        "/chat",
        json={"question": "   "},
    )

    assert response.status_code == 400


def test_chat_unavailable():
    with patch(
        "app.api.main.ask_finsight",
        side_effect=RuntimeError("Ollama unavailable"),
    ):
        response = client.post(
            "/chat",
            json={"question": "Explain anomaly detection"},
        )

    assert response.status_code == 503
