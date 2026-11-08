
SELECT
    customer_id,
    COUNT(*) AS total_transactions,
    ROUND(SUM(amount), 2) AS total_transaction_value,
    ROUND(AVG(amount), 2) AS average_transaction_value,
    MIN(transaction_date) AS first_transaction,
    MAX(transaction_date) AS latest_transaction
FROM main.transactions
GROUP BY customer_id
