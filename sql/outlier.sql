-- test_outliers_advanced.sql
-- Advanced tests for outlier detection as SQL filtering predicates

\echo '=========================================='
\echo '=== ADVANCED OUTLIER DETECTION TESTS ==='
\echo '=========================================='

\echo ''
\echo '=== Setup: Create Test Data ==='

-- Create a test table with known outliers
DROP TABLE IF EXISTS test_transactions CASCADE;
CREATE TABLE test_transactions (
    id SERIAL PRIMARY KEY,
    user_id INT,
    amount NUMERIC(10,2),
    frequency INT,  -- transactions per month
    account_age_days INT,
    is_fraud BOOLEAN DEFAULT FALSE
);

-- Insert normal transactions
INSERT INTO test_transactions (user_id, amount, frequency, account_age_days)
SELECT 
    generate_series(1, 100),
    random() * 500 + 50,  -- Normal: $50-$550
    (random() * 10 + 5)::int,  -- Normal: 5-15 transactions/month
    (random() * 365 + 30)::int  -- Normal: 30-395 days old
;

-- Insert known fraudulent outliers (extreme values)
INSERT INTO test_transactions (user_id, amount, frequency, account_age_days, is_fraud)
VALUES 
    (101, 5000.00, 50, 5, TRUE),   -- High amount, high frequency, new account
    (102, 4800.00, 45, 3, TRUE),   -- High amount, high frequency, new account
    (103, 0.01, 100, 1, TRUE),     -- Tiny amount, very high frequency, brand new
    (104, 3000.00, 35, 7, TRUE),   -- High amount, high frequency
    (105, 2500.00, 30, 2, TRUE);   -- High amount, moderate frequency, new

\echo 'Test data created: 100 normal + 5 fraudulent transactions'

\echo ''
\echo '=== Test 1: Isolation Forest Detection ==='
SELECT db4ml.detect_outliers(
    'SELECT amount, frequency, account_age_days FROM test_transactions',
    'isolation_forest'
) AS if_job_id \gset

SELECT pg_sleep(8);

\echo 'Isolation Forest Results:'
SELECT 
    COUNT(*) AS total_outliers,
    COUNT(*) FILTER (WHERE (row_data->>'is_fraud')::boolean = TRUE) AS true_positives,
    COUNT(*) FILTER (WHERE (row_data->>'is_fraud')::boolean = FALSE) AS false_positives
FROM db4ml.outliers o
JOIN test_transactions t ON (o.row_data->>'id')::int = t.id
WHERE o.job_id = :if_job_id;

\echo ''
\echo 'Top 10 Detected Outliers (Isolation Forest):'
SELECT 
    (row_data->>'id')::int AS transaction_id,
    (row_data->>'amount')::numeric AS amount,
    (row_data->>'frequency')::int AS frequency,
    (row_data->>'account_age_days')::int AS account_age,
    ROUND(outlier_score::numeric, 4) AS score,
    t.is_fraud AS actual_fraud
FROM db4ml.outliers o
JOIN test_transactions t ON (o.row_data->>'id')::int = t.id
WHERE o.job_id = :if_job_id
ORDER BY ABS(outlier_score) DESC
LIMIT 10;

\echo ''
\echo '=== Test 2: Local Outlier Factor (LOF) ==='
SELECT db4ml.detect_outliers(
    'SELECT amount, frequency, account_age_days FROM test_transactions',
    'lof'
) AS lof_job_id \gset

SELECT pg_sleep(8);

\echo 'LOF Results:'
SELECT 
    COUNT(*) AS total_outliers,
    COUNT(*) FILTER (WHERE (row_data->>'is_fraud')::boolean = TRUE) AS true_positives,
    COUNT(*) FILTER (WHERE (row_data->>'is_fraud')::boolean = FALSE) AS false_positives
FROM db4ml.outliers o
JOIN test_transactions t ON (o.row_data->>'id')::int = t.id
WHERE o.job_id = :lof_job_id;

\echo ''
\echo '=== Test 3: Z-Score Method ==='
SELECT db4ml.detect_outliers(
    'SELECT amount, frequency, account_age_days FROM test_transactions',
    'zscore'
) AS zscore_job_id \gset

SELECT pg_sleep(8);

\echo 'Z-Score Results:'
SELECT 
    COUNT(*) AS total_outliers,
    COUNT(*) FILTER (WHERE (row_data->>'is_fraud')::boolean = TRUE) AS true_positives,
    COUNT(*) FILTER (WHERE (row_data->>'is_fraud')::boolean = FALSE) AS false_positives
FROM db4ml.outliers o
JOIN test_transactions t ON (o.row_data->>'id')::int = t.id
WHERE o.job_id = :zscore_job_id;

\echo ''
\echo '=== Test 4: Method Comparison ==='
SELECT 
    detection_method,
    COUNT(*) AS outliers_detected,
    AVG(CASE WHEN t.is_fraud THEN 1.0 ELSE 0.0 END) AS precision,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraud_caught
FROM db4ml.outliers o
JOIN test_transactions t ON (o.row_data->>'id')::int = t.id
GROUP BY detection_method
ORDER BY fraud_caught DESC;

\echo ''
\echo '=== Test 5: SQL WHERE Clause Integration ==='

-- Create helper function for cleaner queries
CREATE OR REPLACE FUNCTION is_transaction_outlier(
    trans_id INT,
    method TEXT
) RETURNS BOOLEAN AS $$
    SELECT EXISTS (
        SELECT 1 
        FROM db4ml.outliers o
        JOIN db4ml.jobs j ON o.job_id = j.id
        WHERE (o.row_data->>'id')::int = trans_id
        AND j.algo_name = 'outlier_' || method
        AND j.status = 'completed'
        ORDER BY j.finished_at DESC
        LIMIT 1
    );
$$ LANGUAGE SQL STABLE;

\echo 'Query 1: Get only non-outlier transactions'
SELECT 
    id, 
    amount, 
    frequency,
    'Normal' AS status
FROM test_transactions
WHERE NOT is_transaction_outlier(id, 'isolation_forest')
LIMIT 5;

\echo ''
\echo 'Query 2: Get only outlier transactions'
SELECT 
    id, 
    amount, 
    frequency,
    is_fraud,
    'Outlier' AS status
FROM test_transactions
WHERE is_transaction_outlier(id, 'isolation_forest')
ORDER BY amount DESC;

\echo ''
\echo 'Query 3: Aggregate stats excluding outliers'
SELECT 
    'With Outliers' AS dataset,
    COUNT(*) AS count,
    ROUND(AVG(amount)::numeric, 2) AS avg_amount,
    ROUND(STDDEV(amount)::numeric, 2) AS stddev_amount,
    MAX(amount) AS max_amount
FROM test_transactions
UNION ALL
SELECT 
    'Without Outliers' AS dataset,
    COUNT(*) AS count,
    ROUND(AVG(amount)::numeric, 2) AS avg_amount,
    ROUND(STDDEV(amount)::numeric, 2) AS stddev_amount,
    MAX(amount) AS max_amount
FROM test_transactions
WHERE NOT is_transaction_outlier(id, 'isolation_forest');

\echo ''
\echo '=== Test 6: Dynamic Outlier View ==='

CREATE OR REPLACE VIEW suspicious_transactions AS
SELECT 
    t.*,
    o.detection_method,
    o.outlier_score,
    o.detected_at
FROM test_transactions t
JOIN db4ml.outliers o ON (o.row_data->>'id')::int = t.id
WHERE o.job_id IN (
    SELECT j.id 
    FROM db4ml.jobs j 
    WHERE j.algo_name LIKE 'outlier_%' 
    AND j.status = 'completed'
    ORDER BY j.finished_at DESC 
    LIMIT 3
);

\echo 'Suspicious Transactions View:'
SELECT 
    id,
    amount,
    frequency,
    is_fraud,
    detection_method,
    ROUND(outlier_score::numeric, 4) AS score
FROM suspicious_transactions
ORDER BY is_fraud DESC, ABS(outlier_score) DESC;

\echo ''
\echo '=== Test 7: Real-time Outlier Scoring ==='

-- Create function to get outlier score for new data
CREATE OR REPLACE FUNCTION get_outlier_score(
    p_amount NUMERIC,
    p_frequency INT,
    p_account_age INT,
    method TEXT DEFAULT 'isolation_forest'
) RETURNS TABLE(is_outlier BOOLEAN, score NUMERIC, similar_outliers INT) AS $$
DECLARE
    latest_job_id BIGINT;
    avg_score NUMERIC;
    score_threshold NUMERIC;
BEGIN
    -- Get latest job for this method
    SELECT j.id INTO latest_job_id
    FROM db4ml.jobs j
    WHERE j.algo_name = 'outlier_' || method
    AND j.status = 'completed'
    ORDER BY j.finished_at DESC
    LIMIT 1;
    
    IF latest_job_id IS NULL THEN
        RETURN QUERY SELECT FALSE, 0::NUMERIC, 0;
        RETURN;
    END IF;
    
    -- Calculate average outlier score threshold
    SELECT AVG(ABS(o.outlier_score)) INTO avg_score
    FROM db4ml.outliers o
    WHERE o.job_id = latest_job_id;
    
    score_threshold := avg_score * 1.2;  -- 20% above average
    
    -- Find similar transactions in outlier set
    RETURN QUERY
    SELECT 
        TRUE as is_outlier,
        avg_score as score,
        COUNT(*)::INT as similar_outliers
    FROM db4ml.outliers o
    WHERE o.job_id = latest_job_id
    AND ABS((o.row_data->>'amount')::numeric - p_amount) < 500
    AND ABS((o.row_data->>'frequency')::int - p_frequency) < 10;
END;
$$ LANGUAGE plpgsql;

\echo 'Test new transaction scoring:'
SELECT * FROM get_outlier_score(5000.00, 50, 5, 'isolation_forest');
SELECT * FROM get_outlier_score(300.00, 8, 180, 'isolation_forest');

\echo ''
\echo '=== Test 8: Outlier-Based Alerts ==='

CREATE OR REPLACE VIEW high_risk_transactions AS
SELECT 
    t.id,
    t.user_id,
    t.amount,
    t.frequency,
    t.account_age_days,
    COUNT(DISTINCT o.detection_method) AS methods_flagged,
    AVG(ABS(o.outlier_score)) AS avg_outlier_score,
    array_agg(DISTINCT o.detection_method) AS flagged_by,
    t.is_fraud AS actual_status
FROM test_transactions t
JOIN db4ml.outliers o ON (o.row_data->>'id')::int = t.id
GROUP BY t.id, t.user_id, t.amount, t.frequency, t.account_age_days, t.is_fraud
HAVING COUNT(DISTINCT o.detection_method) >= 2  -- Flagged by 2+ methods
ORDER BY COUNT(DISTINCT o.detection_method) DESC, AVG(ABS(o.outlier_score)) DESC;

\echo 'High Risk Transactions (flagged by multiple methods):'
SELECT * FROM high_risk_transactions;

\echo ''
\echo '=== Test 9: Performance Metrics ==='

\echo 'Detection Performance by Method:'
WITH fraud_stats AS (
    SELECT 
        o.detection_method,
        COUNT(*) AS total_flagged,
        SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS true_positives,
        SUM(CASE WHEN NOT t.is_fraud THEN 1 ELSE 0 END) AS false_positives,
        (SELECT COUNT(*) FROM test_transactions WHERE is_fraud) AS total_fraud
    FROM db4ml.outliers o
    JOIN test_transactions t ON (o.row_data->>'id')::int = t.id
    GROUP BY o.detection_method
)
SELECT 
    detection_method,
    total_flagged,
    true_positives,
    false_positives,
    ROUND((true_positives::NUMERIC / NULLIF(total_flagged, 0) * 100), 2) AS precision_pct,
    ROUND((true_positives::NUMERIC / total_fraud * 100), 2) AS recall_pct,
    ROUND((2.0 * (true_positives::NUMERIC / NULLIF(total_flagged, 0)) * 
           (true_positives::NUMERIC / total_fraud) / 
           NULLIF((true_positives::NUMERIC / NULLIF(total_flagged, 0)) + 
                  (true_positives::NUMERIC / total_fraud), 0) * 100), 2) AS f1_score_pct
FROM fraud_stats
ORDER BY f1_score_pct DESC NULLS LAST;

\echo ''
\echo '=== Test 10: Cleanup Function ==='

CREATE OR REPLACE FUNCTION cleanup_old_outliers(days_old INT DEFAULT 30) 
RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM db4ml.outliers
    WHERE detected_at < NOW() - (days_old || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

\echo 'Cleanup function created. Test (dry run):'
SELECT COUNT(*) AS would_delete
FROM db4ml.outliers
WHERE detected_at < NOW() - INTERVAL '30 days';

\echo ''
\echo '=========================================='
\echo '=== ADVANCED TESTS COMPLETE ==='
\echo '=========================================='

\echo ''
\echo 'Summary:'
SELECT 
    'Total Test Transactions' AS metric,
    COUNT(*)::TEXT AS value
FROM test_transactions
UNION ALL
SELECT 
    'Actual Fraud Cases' AS metric,
    COUNT(*)::TEXT AS value
FROM test_transactions
WHERE is_fraud
UNION ALL
SELECT 
    'Outliers Detected (All Methods)' AS metric,
    COUNT(DISTINCT (row_data->>'id'))::TEXT AS value
FROM db4ml.outliers o
JOIN db4ml.jobs j ON o.job_id = j.id
WHERE j.algo_name LIKE 'outlier_%'
UNION ALL
SELECT 
    'Detection Methods Tested' AS metric,
    COUNT(DISTINCT detection_method)::TEXT AS value
FROM db4ml.outliers;

\echo ''
\echo '✅ All advanced outlier detection tests completed!'