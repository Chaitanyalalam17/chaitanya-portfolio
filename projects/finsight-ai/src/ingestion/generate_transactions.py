
from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible synthetic financial data
rng = np.random.default_rng(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"

def generate_transactions(n=1000):
    transactions = pd.DataFrame({
        "transaction_id": range(1, n + 1),
        "customer_id": rng.integers(1001, 1201, size=n),
        "amount": np.round(
            rng.lognormal(mean=4.5, sigma=1.0, size=n), 2
        ),
        "transaction_type": rng.choice(
            ["Purchase", "Transfer", "Withdrawal", "Deposit"],
            size=n
        ),
        "channel": rng.choice(
            ["Online", "Mobile", "ATM", "Branch"],
            size=n
        ),
        "transaction_date": pd.date_range(
            start="2026-01-01", periods=n, freq="h"
        )
    })

    return transactions

if __name__ == "__main__":
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = generate_transactions()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(df)} transactions")
    print(f"Saved to: {OUTPUT_FILE}")
    print(df.head())
