-- test_models.sql
-- Enhanced test script for db4ml extension with metadata tracking and corrected ID fetching

\echo '=== DB4ML COMPREHENSIVE TEST SUITE ==='

\echo '=== A. TRAINING ALL MODELS (ASYNC) ==='
\echo '--------------------------------------'

-- 1. Housing Regression (Linear Regression)
\echo '-> 1. Training Housing Model (Linear Regression)'
SELECT db4ml.train_async(
    'SELECT crim, rm, age, dis, tax, ptratio, lstat, medv FROM housing_data'::TEXT,
    'medv'::TEXT,
    'linear_regression'::TEXT
) AS housing_job_id \gset

-- 2. Salary Regression (Random Forest Regressor)
\echo '-> 2. Training Salary Model (Random Forest Regressor)'
SELECT db4ml.train_async(
    'SELECT years_experience, salary FROM salary_data'::TEXT,
    'salary'::TEXT,
    'random_forest_regressor'::TEXT
) AS salary_job_id \gset

-- 3. Salary Regression (Gradient Boosting - for comparison)
\echo '-> 3. Training Salary Model v2 (Gradient Boosting Regressor)'
SELECT db4ml.train_async(
    'SELECT years_experience, salary FROM salary_data'::TEXT,
    'salary'::TEXT,
    'gradient_boosting_regressor'::TEXT
) AS salary_gb_job_id \gset

-- 4. Mobile Price Classification (Logistic Regression)
\echo '-> 4. Training Mobile Price Model (Logistic Regression)'
SELECT db4ml.train_async(
    'SELECT battery_power, ram, px_height, px_width, mobile_wt, four_g, touch_screen, price_range FROM mobile_price_data'::TEXT,
    'price_range'::TEXT,
    'logistic_regression'::TEXT
) AS mobile_job_id \gset

-- 5. Iris Classification (K-Nearest Neighbors)
\echo '-> 5. Training Iris Model (K-Nearest Neighbors)'
SELECT db4ml.train_async(
    'SELECT sepal_length, sepal_width, petal_length, petal_width, 
             CASE species WHEN ''Iris-setosa'' THEN 0 WHEN ''Iris-versicolor'' THEN 1 ELSE 2 END AS species_num
      FROM iris_data'::TEXT,
    'species_num'::TEXT,
    'k_nearest_neighbors'::TEXT
) AS iris_job_id \gset

-- 6. Customer Clustering (K-Means - Unsupervised)
\echo '-> 6. Training Customer Segmentation (K-Means Clustering, N=4)'
SELECT db4ml.train_async(
    'SELECT years_experience, salary FROM salary_data'::TEXT,
    '4'::TEXT, -- N_CLUSTERS passed as target_col argument
    'k_means_clustering'::TEXT
) AS cluster_job_id \gset

\echo '=== B. OUTLIER DETECTION JOBS ==='
\echo '---------------------------------'

-- 7. Housing Outliers (Isolation Forest)
\echo '-> 7. Detecting Housing Price Outliers (Isolation Forest)'
SELECT db4ml.detect_outliers(
    'SELECT crim, rm, age, dis, tax, ptratio, lstat, medv FROM housing_data'::TEXT,
    'isolation_forest'::TEXT
) AS housing_outlier_job_id \gset

-- 8. Salary Outliers (Local Outlier Factor)
\echo '-> 8. Detecting Salary Outliers (LOF)'
SELECT db4ml.detect_outliers(
    'SELECT years_experience, salary FROM salary_data'::TEXT,
    'lof'::TEXT
) AS salary_outlier_job_id \gset

-- 9. Mobile Price Outliers (Z-Score)
\echo '-> 9. Detecting Mobile Spec Outliers (Z-Score)'
SELECT db4ml.detect_outliers(
    'SELECT battery_power, ram, px_height, px_width, mobile_wt FROM mobile_price_data'::TEXT,
    'zscore'::TEXT
) AS mobile_outlier_job_id \gset

\echo '=== C. WAITING FOR JOBS TO COMPLETE ==='
\echo '---------------------------------------'
\echo 'Current job status:'
SELECT 
    id, 
    algo_name, 
    status, 
    created_at,
    CASE 
        WHEN status = 'pending' THEN '⏳ Pending'
        WHEN status = 'processing' THEN '⚙️  Processing'
        WHEN status = 'completed' THEN '✅ Completed'
        WHEN status = 'failed' THEN '❌ Failed'
    END AS status_icon
FROM db4ml.jobs 
ORDER BY created_at DESC 
LIMIT 9;

\echo ''
\echo 'Waiting 15 seconds for background workers to complete...'
SELECT pg_sleep(15);

\echo ''
\echo 'Final job status:'
SELECT 
    id, 
    algo_name, 
    status,
    finished_at - created_at AS duration,
    CASE 
        WHEN status = 'completed' THEN '✅'
        WHEN status = 'failed' THEN '❌'
        ELSE '⚠️'
    END AS result,
    result_model_id -- Display the crucial Model ID
FROM db4ml.jobs 
ORDER BY created_at DESC 
LIMIT 9;

\echo '=== C.5 FETCHING MODEL IDs FOR PREDICTION (CRITICAL STEP) ==='
\echo '-----------------------------------------------------------'
-- Must run these queries to map Job IDs to Model IDs for prediction.

SELECT result_model_id AS housing_model_id FROM db4ml.jobs WHERE id = :housing_job_id \gset
SELECT result_model_id AS salary_model_id FROM db4ml.jobs WHERE id = :salary_job_id \gset
SELECT result_model_id AS salary_gb_model_id FROM db4ml.jobs WHERE id = :salary_gb_job_id \gset
SELECT result_model_id AS mobile_model_id FROM db4ml.jobs WHERE id = :mobile_job_id \gset
SELECT result_model_id AS iris_model_id FROM db4ml.jobs WHERE id = :iris_job_id \gset
SELECT result_model_id AS cluster_model_id FROM db4ml.jobs WHERE id = :cluster_job_id \gset

\echo 'Model IDs are now set for prediction.'

\echo '=== D. MODEL METADATA & TRACKING ==='
-- ... (This section remains unchanged as it uses the models table) ...

\echo ''
\echo 'Automatic Metadata Captured:'
SELECT 
    id AS model_id,
    name,
    algo_type,
    algo_name,
    training_row_count AS rows_trained,
    array_length(feature_columns, 1) AS num_features,
    target_column,
    substring(data_hash, 1, 8) || '...' AS data_hash_preview,
    created_at
FROM db4ml.models
ORDER BY created_at DESC;

\echo ''
\echo 'Training Data SQL Tracking:'
SELECT 
    id,
    name,
    substring(training_data_sql, 1, 60) || '...' AS sql_query_preview
FROM db4ml.models
ORDER BY created_at DESC;

\echo ''
\echo 'Feature Columns Used:'
SELECT 
    id,
    name,
    feature_columns
FROM db4ml.models
ORDER BY created_at DESC;

\echo '=== E. MODEL PERFORMANCE METRICS ==='
-- ... (This section remains unchanged as it uses the models table) ...

\echo ''
\echo 'Performance Summary:'
SELECT 
    id AS model_id,
    name,
    algo_type,
    algo_name,
    CASE algo_type
        WHEN 'regressor' THEN 
            'R²=' || ROUND((metrics->>'r2_score')::numeric, 4)::text || 
            ' MSE=' || ROUND((metrics->>'mse')::numeric, 2)::text ||
            ' RMSE=' || ROUND((metrics->>'rmse')::numeric, 2)::text
        WHEN 'classifier' THEN 
            'Accuracy=' || ROUND((metrics->>'accuracy')::numeric, 4)::text ||
            ' (Train=' || ROUND((metrics->>'train_accuracy')::numeric, 4)::text || ')'
        WHEN 'clusterer' THEN 
            'Clusters=' || (metrics->>'n_clusters')::text ||
            ' DB_Score=' || COALESCE(ROUND((metrics->>'davies_bouldin_score')::numeric, 4)::text, 'N/A')
        ELSE 'N/A'
    END AS performance_metrics,
    (metrics->>'training_rows')::int AS train_rows,
    (metrics->>'test_rows')::int AS test_rows
FROM db4ml.models
ORDER BY created_at DESC;

\echo ''
\echo 'Using Model Performance View:'
SELECT * FROM db4ml.model_performance ORDER BY created_at DESC;

\echo '=== F. OUTLIER DETECTION RESULTS ==='
-- ... (This section remains unchanged as it uses job_id and outlier_job_id vars) ...

\echo ''
\echo 'Outlier Summary by Job:'
SELECT 
    j.id AS job_id,
    j.algo_name,
    COUNT(o.id) AS outliers_found,
    ROUND(AVG(o.outlier_score)::numeric, 4) AS avg_score,
    MIN(o.outlier_score) AS min_score,
    MAX(o.outlier_score) AS max_score
FROM db4ml.jobs j
LEFT JOIN db4ml.outliers o ON j.id = o.job_id
WHERE j.algo_name LIKE 'outlier_%'
GROUP BY j.id, j.algo_name
ORDER BY j.created_at DESC;

\echo ''
\echo 'Top 5 Housing Price Outliers:'
SELECT 
    (row_data->>'medv')::numeric AS median_value,
    (row_data->>'crim')::numeric AS crime_rate,
    (row_data->>'rm')::numeric AS avg_rooms,
    ROUND(outlier_score::numeric, 4) AS outlier_score,
    detected_at
FROM db4ml.outliers
WHERE job_id = :housing_outlier_job_id
ORDER BY ABS(outlier_score) DESC
LIMIT 5;

\echo ''
\echo 'Salary Outliers (High Earners):'
SELECT 
    (row_data->>'salary')::numeric AS salary,
    (row_data->>'years_experience')::numeric AS years_exp,
    ROUND(outlier_score::numeric, 4) AS outlier_score
FROM db4ml.outliers
WHERE job_id = :salary_outlier_job_id
ORDER BY (row_data->>'salary')::numeric DESC
LIMIT 5;

\echo ''
\echo 'Using Recent Outliers View:'
SELECT 
    detection_method,
    row_data->>'id' AS record_id,
    ROUND(outlier_score::numeric, 4) AS score,
    detected_at
FROM db4ml.recent_outliers
LIMIT 10;

\echo '=== G. MAKING PREDICTIONS ==='
\echo '-----------------------------'
-- All :job_id variables replaced with :model_id variables.

\echo ''
\echo 'Single Predictions:'
SELECT 
    'Housing (MedV)' AS model,
    ROUND(db4ml.predict(
        :housing_model_id, -- Model ID
        ARRAY[0.05, 6.5, 70.0, 5.0, 300.0, 16.0, 8.0]::FLOAT8[]
    )::numeric, 2) AS prediction
UNION ALL
SELECT 
    'Salary (RF)' AS model,
    ROUND(db4ml.predict(
        :salary_model_id, -- Model ID
        ARRAY[7.5]::FLOAT8[]
    )::numeric, 2) AS prediction
UNION ALL
SELECT 
    'Salary (GB)' AS model,
    ROUND(db4ml.predict(
        :salary_gb_model_id, -- Model ID
        ARRAY[7.5]::FLOAT8[]
    )::numeric, 2) AS prediction
UNION ALL
SELECT 
    'Mobile (Price Class)' AS model,
    ROUND(db4ml.predict(
        :mobile_model_id, -- Model ID
        ARRAY[4000.0, 3500.0, 1800.0, 1800.0, 150.0, 1.0, 1.0]::FLOAT8[]
    )::numeric, 0) AS prediction
UNION ALL
SELECT 
    'Iris (Species)' AS model,
    ROUND(db4ml.predict(
        :iris_model_id, -- Model ID
        ARRAY[6.3, 3.3, 6.0, 2.5]::FLOAT8[]
    )::numeric, 0) AS prediction
UNION ALL
SELECT 
    'Customer Cluster' AS model,
    ROUND(db4ml.predict(
        :cluster_model_id, -- Model ID
        ARRAY[5.0, 60000.0]::FLOAT8[]
    )::numeric, 0) AS prediction;

\echo ''
\echo 'Batch Predictions (First 5 Iris flowers):'
SELECT * FROM db4ml.predict_batch(
    :iris_model_id, -- Model ID
    'SELECT sepal_length, sepal_width, petal_length, petal_width 
      FROM iris_data LIMIT 5'
);

\echo '=== H. MODEL USAGE STATISTICS ==='
-- ... (This section remains unchanged as it uses the models table) ...

\echo ''
\echo 'Model Usage Tracking:'
SELECT 
    id,
    name,
    use_count,
    last_used_at,
    CASE 
        WHEN last_used_at IS NOT NULL THEN 
            ROUND(EXTRACT(EPOCH FROM (last_used_at - created_at))::numeric, 2) || ' seconds'
        ELSE 'Never used'
    END AS time_to_first_use
FROM db4ml.models
ORDER BY use_count DESC;

\echo '=== I. MODEL COMPARISON ==='
-- ... (This section remains unchanged as it uses the models table) ...

\echo ''
\echo 'Comparing Salary Models (RF vs GB):'
SELECT 
    name,
    algo_name,
    ROUND((metrics->>'r2_score')::numeric, 4) AS r2_score,
    ROUND((metrics->>'mse')::numeric, 2) AS mse,
    ROUND((metrics->>'rmse')::numeric, 2) AS rmse,
    training_row_count
FROM db4ml.models
WHERE training_data_sql LIKE '%salary_data%'
    AND algo_type = 'regressor'
ORDER BY (metrics->>'r2_score')::numeric DESC;

\echo '=== J. MODEL VERSIONING ==='
-- ... (This section remains unchanged as it uses the models table) ...

\echo ''
\echo 'Models by Data Hash (Versioning):'
SELECT 
    data_hash,
    COUNT(*) AS model_count,
    string_agg(name, ', ') AS models,
    MAX(training_row_count) AS rows
FROM db4ml.models
GROUP BY data_hash
HAVING COUNT(*) > 1
ORDER BY model_count DESC;

\echo '=== K. ADVANCED: OUTLIERS AS FILTER ==='
-- ... (This section remains unchanged as it uses job_id and outlier_job_id vars) ...

\echo ''
\echo 'Example: Filter out housing outliers from query:'
SELECT COUNT(*) AS total_houses FROM housing_data;

SELECT COUNT(*) AS houses_without_outliers
FROM housing_data h
WHERE NOT EXISTS (
    SELECT 1 FROM db4ml.outliers o
    WHERE o.job_id = :housing_outlier_job_id
    AND (o.row_data->>'medv')::numeric = h.medv
    AND (o.row_data->>'crim')::numeric = h.crim
);

\echo '=== L. MODEL ARTIFACTS VERIFICATION ==='
\echo '---------------------------------------'
-- Removed old placeholder L section, assuming this is where you verify data in db4ml.models

\echo '=== M. ERROR HANDLING TEST ==='
-- ... (This section remains unchanged as it uses the jobs table) ...

\echo ''
\echo 'Checking for failed jobs:'
SELECT 
    id,
    algo_name,
    status,
    substring(error_msg, 1, 100) AS error_preview
FROM db4ml.jobs
WHERE status = 'failed'
ORDER BY created_at DESC;

\echo '=== TEST SUITE COMPLETE ==='
\echo '---------------------------'

\echo ''
\echo 'Summary Statistics:'
SELECT 
    'Total Models' AS metric,
    COUNT(*)::text AS value
FROM db4ml.models
UNION ALL
SELECT 
    'Total Jobs' AS metric,
    COUNT(*)::text AS value
FROM db4ml.jobs
UNION ALL
SELECT 
    'Completed Jobs' AS metric,
    COUNT(*)::text AS value
FROM db4ml.jobs
WHERE status = 'completed'
UNION ALL
SELECT 
    'Failed Jobs' AS metric,
    COUNT(*)::text AS value
FROM db4ml.jobs
WHERE status = 'failed'
UNION ALL
SELECT 
    'Total Outliers Detected' AS metric,
    COUNT(*)::text AS value
FROM db4ml.outliers
UNION ALL
SELECT 
    'Total Model Uses' AS metric,
    SUM(use_count)::text AS value
FROM db4ml.models;

\echo ''