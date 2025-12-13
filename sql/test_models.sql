\echo '=== DB4ML COMPREHENSIVE BENCHMARK (RESTORED & FIXED) ==='

-- 1. CLEANUP
DELETE FROM db4ml.outliers;
DELETE FROM db4ml.jobs;
DELETE FROM db4ml.models;
DROP TABLE IF EXISTS housing_sim;
DROP TABLE IF EXISTS mobile_sim;
DROP TABLE IF EXISTS cluster_sim;
DROP TABLE IF EXISTS outlier_sim;

-- 2. DATA GENERATION
\echo '-> Generating 4 Datasets...'

-- Dataset 1: Housing (Regression)
CREATE TABLE housing_sim (crim float, rm float, age float, medv float);
INSERT INTO housing_sim 
SELECT random(), random()*10, random()*100, random()*500 
FROM generate_series(1, 5000);

-- Dataset 2: Mobile (Classification)
CREATE TABLE mobile_sim (battery_power float, ram float, price_range int);
INSERT INTO mobile_sim 
SELECT random()*4000, random()*4000, (CASE WHEN random() > 0.5 THEN 1 ELSE 0 END)
FROM generate_series(1, 5000);

-- Dataset 3: Customer (Clustering)
CREATE TABLE cluster_sim (spending_score float, income float);
INSERT INTO cluster_sim 
SELECT random()*100, random()*100000 
FROM generate_series(1, 5000);

-- Dataset 4: Security Logs (Outlier Detection)
CREATE TABLE outlier_sim (val1 float, val2 float);
INSERT INTO outlier_sim SELECT random()*10, random()*10 FROM generate_series(1, 1000);
-- Inject outliers
INSERT INTO outlier_sim VALUES (1000, 1000), (500, 500), (200, 200), (999, 999), (888, 888);


-- 3. LAUNCH JOBS (PAIRS for Comparison)
\echo '-> Launching Training Jobs...'

-- A. Housing (Regression) - COMPARE FIT vs PARTIAL
SELECT db4ml.train_async('SELECT * FROM housing_sim', 'medv', 'sgd_regressor:fit') AS h_fit \gset
SELECT db4ml.train_async('SELECT * FROM housing_sim', 'medv', 'sgd_regressor:partial_fit') AS h_part \gset

-- B. Mobile (Classification) - COMPARE FIT vs PARTIAL
SELECT db4ml.train_async('SELECT * FROM mobile_sim', 'price_range', 'sgd_classifier:fit') AS m_fit \gset
SELECT db4ml.train_async('SELECT * FROM mobile_sim', 'price_range', 'sgd_classifier:partial_fit') AS m_part \gset

-- C. Clustering (KMeans) - COMPARE FIT vs PARTIAL
SELECT db4ml.train_async('SELECT * FROM cluster_sim', '3', 'minibatch_kmeans:fit') AS c_fit \gset
SELECT db4ml.train_async('SELECT * FROM cluster_sim', '3', 'minibatch_kmeans:partial_fit') AS c_part \gset

-- D. Outlier Detection (Single Job)
SELECT db4ml.detect_outliers('SELECT * FROM outlier_sim', 'isolation_forest') AS o_job \gset


-- 4. WAIT LOOP
\echo '-> Waiting for workers (max 30s)...'
DO $$
DECLARE
    unfinished INT;
    i INT := 0;
BEGIN
    LOOP
        SELECT COUNT(*) INTO unfinished FROM db4ml.jobs WHERE status IN ('pending', 'processing');
        EXIT WHEN unfinished = 0;
        IF i >= 30 THEN RAISE EXCEPTION 'Timeout waiting for jobs'; END IF;
        PERFORM pg_sleep(1);
        i := i + 1;
    END LOOP;
END $$;


-- 5. REPORTING
\echo ''
\echo '=================================================='
\echo 'RESULTS: DATASET 1 - REGRESSION (Fit vs Partial)'
\echo '=================================================='
SELECT 
    name,
    ROUND((metrics->>'training_time_sec')::numeric, 5) AS time_sec,
    ROUND((metrics->>'mse')::numeric, 2) AS mse,
    ROUND((metrics->>'r2_score')::numeric, 4) AS r2
FROM db4ml.models
WHERE training_data_sql LIKE '%housing_sim%'
ORDER BY name;

\echo ''
\echo '=================================================='
\echo 'RESULTS: DATASET 2 - CLASSIFICATION (Fit vs Partial)'
\echo '=================================================='
SELECT 
    name,
    ROUND((metrics->>'training_time_sec')::numeric, 5) AS time_sec,
    ROUND((metrics->>'accuracy')::numeric, 4) AS accuracy
FROM db4ml.models
WHERE training_data_sql LIKE '%mobile_sim%'
ORDER BY name;

\echo ''
\echo '=================================================='
\echo 'RESULTS: DATASET 3 - CLUSTERING (Fit vs Partial)'
\echo '=================================================='
SELECT 
    name,
    ROUND((metrics->>'training_time_sec')::numeric, 5) AS time_sec,
    (metrics->>'n_clusters') AS clusters
FROM db4ml.models
WHERE training_data_sql LIKE '%cluster_sim%'
ORDER BY name;

\echo ''
\echo '=================================================='
\echo 'RESULTS: DATASET 4 - OUTLIER DETECTION'
\echo '=================================================='
SELECT 
    detection_method, 
    COUNT(*) as outliers_found
FROM db4ml.outliers
GROUP BY detection_method;

SELECT row_data, outlier_score 
FROM db4ml.outliers 
WHERE outlier_score < -0.1 -- Lower score = more anomalous
ORDER BY outlier_score ASC 
LIMIT 5;