
# FinSight AI Knowledge Base

## Project Overview
FinSight AI is a financial intelligence platform built
using Python, DuckDB, dbt, Streamlit, FastAPI,
machine learning, SHAP, and Ollama.

The platform uses synthetic financial transactions
for demonstration and educational purposes.

## Financial Analytics
The platform provides:
- Total transaction count
- Total transaction value
- Average transaction amount
- Unique customer count
- Daily financial KPIs
- Channel analytics
- Customer analytics

## Data Engineering
Financial transactions are processed using Python
and stored in DuckDB.

dbt transforms the transaction data into analytical
models for reporting and visualization.

## Anomaly Detection
The platform uses an Isolation Forest model to
identify unusual financial transactions.

The model uses transaction amount, transaction type,
and transaction channel as input features.

An anomaly is not necessarily fraud. Flagged
transactions require further investigation.

## Explainable AI
SHAP is used to explain the Isolation Forest model's
output by estimating feature contributions.

SHAP values do not represent fraud probabilities.

## API
FastAPI provides endpoints for financial analytics,
anomaly results, and transaction explanations.

## Important Limitations
- The dataset is synthetic.
- The model has not been validated against confirmed fraud labels.
- Anomaly scores should not be treated as fraud probabilities.
- The platform is a demonstration project, not a
  production financial decision-making system.
