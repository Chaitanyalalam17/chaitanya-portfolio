
SELECT
    CAST(transaction_date AS DATE) AS transaction_day,
    COUNT(*) AS total_transactions,
    ROUND(SUM(amount), 2) AS total_transaction_value,
    ROUND(AVG(amount), 2) AS average_transaction_value,
    COUNT(DISTINCT customer_id) AS unique_customers
FROM main.transactions
GROUP BY 1
