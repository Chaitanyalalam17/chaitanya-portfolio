
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"


def load_dbt_analytics() -> dict[str, pd.DataFrame]:
    """Load dbt-generated analytics tables from DuckDB."""

    queries = {
        "daily": """
            SELECT *
            FROM analytics.daily_financial_kpis
            ORDER BY transaction_day
        """,
        "channels": """
            SELECT *
            FROM analytics.channel_analytics
            ORDER BY total_transaction_value DESC
        """,
        "customers": """
            SELECT *
            FROM analytics.customer_analytics
            ORDER BY total_transaction_value DESC
        """,
    }

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "DuckDB database not found. Run the ingestion pipeline."
        )

    results = {}

    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        for name, query in queries.items():
            results[name] = conn.execute(query).df()

    return results


if __name__ == "__main__":
    analytics = load_dbt_analytics()

    for name, dataframe in analytics.items():
        print(f"\n{name.upper()} ANALYTICS")
        print(dataframe.head())

