-- test_models.sql
-- Script to train all models and test predictions using the unified db4ml interface.

\echo '===================================='
\echo '=== A. TRAINING ALL MODELS (ASYNC) ==='
\echo '===================================='

-- 1. Housing Regression (Linear Regression)
\echo '-> 1. Training Housing Model (Linear Regression)'
SELECT db4ml.train_async(
    'SELECT crim, rm, age, dis, tax, ptratio, lstat, medv FROM housing_data'::TEXT,
    'medv'::TEXT, -- Target Column
    'linear_regression'::TEXT -- Algorithm Name
) AS housing_job_id \gset

-- 2. Salary Regression (Random Forest Regressor)
\echo '-> 2. Training Salary Model (Random Forest Regressor)'
SELECT db4ml.train_async(
    'SELECT years_experience, salary FROM salary_data'::TEXT,
    'salary'::TEXT,
    'random_forest_regressor'::TEXT
) AS salary_job_id \gset

-- 3. Mobile Price Classification (Logistic Regression)
\echo '-> 3. Training Mobile Price Model (Logistic Regression)'
-- We use a subset of numerical/binary features and the 'price_range' target.
SELECT db4ml.train_async(
    'SELECT battery_power, ram, px_height, px_width, mobile_wt, four_g, touch_screen, price_range FROM mobile_price_data'::TEXT,
    'price_range'::TEXT,
    'logistic_regression'::TEXT
) AS mobile_job_id \gset

-- 4. Iris Classification (K-Nearest Neighbors)
-- Note: Converts the 'species' text column to a numerical target (0, 1, 2)
\echo '-> 4. Training Iris Model (K-Nearest Neighbors)'
SELECT db4ml.train_async(
    'SELECT sepal_length, sepal_width, petal_length, petal_width, 
            CASE species WHEN ''Iris-setosa'' THEN 0 WHEN ''Iris-versicolor'' THEN 1 ELSE 2 END AS species_num
     FROM iris_data'::TEXT,
    'species_num'::TEXT,
    'k_nearest_neighbors'::TEXT
) AS iris_job_id \gset

\echo '===================================='
\echo '=== B. WAITING FOR JOBS TO COMPLETE ==='
\echo '===================================='
\echo 'Polling the jobs table. Please wait until status is not "pending" or "processing"...'
SELECT id, algo_name, status, created_at FROM db4ml.jobs ORDER BY created_at DESC LIMIT 4;

-- In a live environment, you would check this output repeatedly until all jobs are 'completed' or 'failed'.

\echo 'Pausing script for 10 seconds to allow background workers to complete training...'
SELECT pg_sleep(10); 
\echo 'Resuming script and verifying models...'

\echo '===================================='
\echo '=== C. MODEL PERFORMANCE METRICS ==='
\echo '===================================='
-- Shows metrics (R2 for Regressors, Accuracy for Classifiers) from the JSONB column.
SELECT 
    id AS model_id,
    algo_name,
    algo_type,
    -- Extract and format metrics based on model type
    CASE algo_type
        WHEN 'regressor' THEN 'R2: ' || ROUND((metrics->>'r2_score')::numeric, 4)::text
        WHEN 'classifier' THEN 'Accuracy: ' || ROUND((metrics->>'accuracy')::numeric, 4)::text
        ELSE 'N/A'
    END AS primary_metric_value,
    metrics
FROM db4ml.models
ORDER BY created_at DESC;

\echo '===================================='
\echo '=== D. MAKING PREDICTIONS ==='
\echo '===================================='

-- We assume the most recently created model ID matches the Job ID for simplicity in testing.
-- The predicted output is cast to numeric (float8 in Postgres).

-- 1. Housing Regression Prediction
SELECT 
    'Housing (Prediction: MedV)' AS scenario,
    -- Features: crim, rm, age, dis, tax, ptratio, lstat
    ROUND(db4ml.predict(:housing_job_id, ARRAY[0.05, 6.5, 70.0, 5.0, 300.0, 16.0, 8.0]::REAL[])::numeric, 2) AS prediction_value
UNION ALL
-- 2. Salary Regression Prediction
SELECT 
    'Salary (Prediction: $)' AS scenario,
    -- Features: years_experience
    ROUND(db4ml.predict(:salary_job_id, ARRAY[7.5]::REAL[])::numeric, 2) AS prediction_value
UNION ALL
-- 3. Mobile Classification Prediction
SELECT 
    'Mobile (Prediction: Price Class 0-3)' AS scenario,
    -- Features: battery_power, ram, px_height, px_width, mobile_wt, four_g, touch_screen
    -- Test case: High specs (should predict class 3 or 2)
    ROUND(db4ml.predict(:mobile_job_id, ARRAY[4000.0, 3500.0, 1800.0, 1800.0, 150.0, 1.0, 1.0]::REAL[])::numeric, 0) AS prediction_value
UNION ALL
-- 4. Iris Classification Prediction
SELECT 
    'Iris (Prediction: Species 0, 1, or 2)' AS scenario,
    -- Features: SepalL, SepalW, PetalL, PetalW (Test case for Iris-virginica, class 2)
    ROUND(db4ml.predict(:iris_job_id, ARRAY[6.3, 3.3, 6.0, 2.5]::REAL[])::numeric, 0) AS prediction_value;

\echo '===================================='
\echo '=== E. MODEL ARTIFACTS (Verification) ==='
\echo '===================================='
-- Show the serialized model size (should be > 0) and the algo name
SELECT 
    id,
    algo_name,
    pg_column_size(model_data) AS model_size_bytes
FROM db4ml.models
ORDER BY created_at DESC LIMIT 4;