-- \echo '=== DB4ML DEBUG BENCHMARK ==='

-- -- 1. Setup Test Data
-- DROP TABLE IF EXISTS benchmark_data;
-- CREATE TABLE benchmark_data (val1 float, val2 float, target float);
-- INSERT INTO benchmark_data 
-- SELECT random(), random(), random()*100 
-- FROM generate_series(1, 10000);

-- -- 2. Clear old logs
-- DELETE FROM db4ml.jobs;
-- DELETE FROM db4ml.models;

-- -- 3. Launch Jobs (Using colon syntax)
-- \echo '-> Launching Jobs...'

-- -- Job A: FIT
-- SELECT db4ml.train_async(
--     'SELECT * FROM benchmark_data', 'target', 'sgd_regressor:fit'
-- ) AS job_fit \gset

-- -- Job B: PARTIAL_FIT
-- SELECT db4ml.train_async(
--     'SELECT * FROM benchmark_data', 'target', 'sgd_regressor:partial_fit'
-- ) AS job_part \gset

-- -- 4. Wait Loop (Force wait for 10s)
-- \echo '-> Waiting 10s for workers...'
-- SELECT pg_sleep(10);

-- -- 5. DIAGNOSTICS (If this shows 'failed', we see the error)
-- \echo '-> JOB STATUS REPORT:'
-- SELECT id, algo_name, status, substring(error_msg, 1, 80) as error_short 
-- FROM db4ml.jobs;

-- -- 6. FINAL COMPARISON
-- \echo '-> PERFORMANCE COMPARISON:'
-- SELECT 
--     algo_name,
--     ROUND((metrics->>'training_time_sec')::numeric, 5) as time_sec,
--     ROUND((metrics->>'mse')::numeric, 5) as mse
-- FROM db4ml.models
-- ORDER BY algo_name;



\echo '=== DB4ML COMPREHENSIVE 3-DATASET BENCHMARK ==='

-- 1. CLEANUP
DELETE FROM db4ml.jobs;
DELETE FROM db4ml.models;
DROP TABLE IF EXISTS housing_sim;
DROP TABLE IF EXISTS mobile_sim;
DROP TABLE IF EXISTS cluster_sim;

-- 2. DATA GENERATION
\echo '-> Generating 3 Datasets (10k rows each)...'

-- Dataset 1: Housing Simulation (Regression)
-- Target: medv (Price)
CREATE TABLE housing_sim (crim float, rm float, age float, medv float);
INSERT INTO housing_sim 
SELECT random(), random()*10, random()*100, random()*500 
FROM generate_series(1, 10000);

-- Dataset 2: Mobile Price Simulation (Classification)
-- Target: price_range (0 or 1)
CREATE TABLE mobile_sim (battery_power float, ram float, price_range int);
INSERT INTO mobile_sim 
SELECT random()*4000, random()*4000, (CASE WHEN random() > 0.5 THEN 1 ELSE 0 END)
FROM generate_series(1, 10000);

-- Dataset 3: Customer Simulation (Clustering)
-- Target: None (Unsupervised)
CREATE TABLE cluster_sim (spending_score float, income float);
INSERT INTO cluster_sim 
SELECT random()*100, random()*100000 
FROM generate_series(1, 10000);


-- 3. LAUNCH JOBS
\echo '-> Launching 6 Training Jobs...'

-- A. Housing (SGD Regressor)
SELECT db4ml.train_async('SELECT * FROM housing_sim', 'medv', 'sgd_regressor:fit') AS h_fit \gset
SELECT db4ml.train_async('SELECT * FROM housing_sim', 'medv', 'sgd_regressor:partial_fit') AS h_part \gset

-- B. Mobile (SGD Classifier)
SELECT db4ml.train_async('SELECT * FROM mobile_sim', 'price_range', 'sgd_classifier:fit') AS m_fit \gset
SELECT db4ml.train_async('SELECT * FROM mobile_sim', 'price_range', 'sgd_classifier:partial_fit') AS m_part \gset

-- C. Clustering (MiniBatch KMeans)
-- Note: '3' is the cluster count passed as target
SELECT db4ml.train_async('SELECT * FROM cluster_sim', '3', 'minibatch_kmeans:fit') AS c_fit \gset
SELECT db4ml.train_async('SELECT * FROM cluster_sim', '3', 'minibatch_kmeans:partial_fit') AS c_part \gset


-- 4. ROBUST WAIT LOOP
\echo '-> Waiting for all workers to finish...'

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
\echo 'RESULTS: DATASET 1 - HOUSING (REGRESSION)'
\echo '=================================================='

SELECT 
    algo_name,
    ROUND((metrics->>'training_time_sec')::numeric, 5) AS time_sec,
    ROUND((metrics->>'mse')::numeric, 2) AS mse,
    ROUND((metrics->>'r2_score')::numeric, 4) AS r2_score
FROM db4ml.models
WHERE training_data_sql LIKE '%housing_sim%'
ORDER BY algo_name;

\echo ''
\echo '=================================================='
\echo 'RESULTS: DATASET 2 - MOBILE (CLASSIFICATION)'
\echo '=================================================='

SELECT 
    algo_name,
    ROUND((metrics->>'training_time_sec')::numeric, 5) AS time_sec,
    -- Accuracy isn't calculated for Regressors, so we ensure we pull valid metrics
    -- If accuracy is missing (it shouldn't be for classifier), it returns null
    (metrics->>'mse') AS mse_loss
FROM db4ml.models
WHERE training_data_sql LIKE '%mobile_sim%'
ORDER BY algo_name;

-- Note: In the python script, SGDClassifier treated as 'classifier' saves 'accuracy' 
-- but I see in previous output you might have only saved 'mse'/'r2' if the type logic wasn't fully separated.
-- If you used my exact previous script, SGDClassifier might have fallen into "Supervised" generic block.
-- Let's check the JSON keys available:
SELECT algo_name, metrics FROM db4ml.models WHERE training_data_sql LIKE '%mobile_sim%';


\echo ''
\echo '=================================================='
\echo 'RESULTS: DATASET 3 - CUSTOMER (CLUSTERING)'
\echo '=================================================='

SELECT 
    algo_name,
    ROUND((metrics->>'training_time_sec')::numeric, 5) AS time_sec,
    (metrics->>'n_clusters') AS clusters_found
FROM db4ml.models
WHERE training_data_sql LIKE '%cluster_sim%'
ORDER BY algo_name;

\echo ''
\echo 'End of Benchmark.'