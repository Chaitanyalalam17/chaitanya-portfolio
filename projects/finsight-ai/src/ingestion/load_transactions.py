
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
DB_FILE = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"


def validate_transactions(df):
    required_columns = [
        "transaction_id",
        "customer_id",
        "amount",
        "transaction_type",
        "channel",
        "transaction_date",
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values detected")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected")

    if (df["amount"] <= 0).any():
        raise ValueError("Invalid transaction amounts detected")

    print("Data validation passed!")


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE, parse_dates=["transaction_date"])

    validate_transactions(df)

    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DB_FILE)) as conn:
        conn.register("transactions_df", df)

        conn.execute("""
            CREATE OR REPLACE TABLE transactions AS
            SELECT * FROM transactions_df
        """)

        summary = conn.execute("""
            SELECT
                transaction_type,
                COUNT(*) AS total_transactions,
                ROUND(SUM(amount), 2) AS total_amount
            FROM transactions
            GROUP BY transaction_type
            ORDER BY total_amount DESC
        """).df()

    print(f"Loaded {len(df)} transactions into DuckDB")
    print(f"Database saved to: {DB_FILE}")
    print("\nTransaction summary:")
    print(summary)


if __name__ == "__main__":
    main()
