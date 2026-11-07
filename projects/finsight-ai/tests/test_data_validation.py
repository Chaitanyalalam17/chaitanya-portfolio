
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.load_transactions import validate_transactions


@pytest.fixture
def valid_data():
    return pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "customer_id": [1001, 1002, 1003],
        "amount": [100.50, 250.00, 75.25],
        "transaction_type": ["Purchase", "Transfer", "Deposit"],
        "channel": ["Online", "Mobile", "Branch"],
        "transaction_date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ],
    })


def test_valid_transactions(valid_data):
    validate_transactions(valid_data)


def test_missing_values(valid_data):
    df = valid_data.copy()
    df.loc[0, "amount"] = None

    with pytest.raises(ValueError, match="Missing values"):
        validate_transactions(df)


def test_duplicate_transaction_ids(valid_data):
    df = valid_data.copy()
    df.loc[1, "transaction_id"] = 1

    with pytest.raises(ValueError, match="Duplicate"):
        validate_transactions(df)


def test_negative_transaction_amount(valid_data):
    df = valid_data.copy()
    df.loc[0, "amount"] = -100

    with pytest.raises(ValueError, match="Invalid transaction amounts"):
        validate_transactions(df)


def test_missing_required_column(valid_data):
    df = valid_data.drop(columns=["customer_id"])

    with pytest.raises(ValueError, match="Missing columns"):
        validate_transactions(df)
