
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"
)


def get_financial_context():
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {DATABASE_PATH}"
        )

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        result = connection.execute(
            """
            SELECT
                COUNT(*) AS total_transactions,
                COALESCE(SUM(amount), 0) AS total_value,
                COALESCE(AVG(amount), 0) AS average_value,
                COUNT(DISTINCT customer_id) AS unique_customers
            FROM transactions
            """
        ).fetchone()

        return {
            "total_transactions": int(result[0]),
            "total_transaction_value": round(float(result[1]), 2),
            "average_transaction": round(float(result[2]), 2),
            "unique_customers": int(result[3]),
        }

    finally:
        connection.close()


if __name__ == "__main__":
    print("FinSight AI Live Financial Context")
    print("-" * 40)
    print(get_financial_context())
