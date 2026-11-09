
import pytest

from src.genai.transaction_context import (
    get_anomaly_summary,
    get_transaction_context,
)
from src.genai.rag_assistant import extract_transaction_id


def test_anomaly_summary():
    result = get_anomaly_summary()

    assert result["total_transactions_scored"] == 1000
    assert result["anomalies_detected"] == 50
    assert result["anomaly_percentage"] == 5.0


def test_transaction_39_exists():
    result = get_transaction_context(39)

    assert result["found"] is True
    assert result["transaction_id"] == 39
    assert result["is_anomaly"] is True


def test_transaction_39_details():
    result = get_transaction_context(39)

    assert result["amount"] == pytest.approx(458.95)
    assert result["transaction_type"] == "Transfer"
    assert result["channel"] == "Branch"


def test_transaction_39_shap():
    result = get_transaction_context(39)

    features = result["top_shap_features"]

    assert len(features) == 5
    assert features[0]["feature"] == "amount"
    assert features[0]["shap_value"] == pytest.approx(
        -2.5477022471355393
    )


def test_missing_transaction():
    result = get_transaction_context(999999)

    assert result["found"] is False


def test_extract_transaction_id():
    assert extract_transaction_id(
        "Is transaction 39 anomalous?"
    ) == 39

    assert extract_transaction_id(
        "Explain transaction ID 39"
    ) == 39

    assert extract_transaction_id(
        "How many anomalies were detected?"
    ) is None
