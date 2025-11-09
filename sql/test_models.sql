\echo '=== Training Housing Model ==='
SELECT db4ml.train_linear(
    'SELECT crim, rm, age, dis, tax, ptratio, lstat, medv FROM housing_data',
    0.2
) AS housing_model_id \gset

\echo '\n=== Training Salary Model ==='
SELECT db4ml.train_linear(
    'SELECT years_experience, salary FROM salary_data',
    0.25
) AS salary_model_id \gset

\echo '\n=== Model Performance Metrics ==='
SELECT 
    m.id AS model_id,
    m.kind,
    m.n_features,
    mt.train_rows,
    mt.test_rows,
    ROUND(mt.train_r2::numeric, 4) AS train_r2,
    ROUND(mt.test_r2::numeric, 4) AS test_r2
FROM db4ml.models m
JOIN db4ml.metrics mt ON m.id = mt.model_id
ORDER BY m.created_at DESC;

\echo '\n=== Making Predictions ==='
SELECT 
    'Housing (crim=0.05, rm=6.5, age=70, dis=5, tax=300, ptratio=16, lstat=8)' AS scenario,
    ROUND(db4ml.predict_linear(:housing_model_id, ARRAY[0.05, 6.5, 70.0, 5.0, 300.0, 16.0, 8.0])::numeric, 2) AS prediction
UNION ALL
SELECT 
    'Salary (7.5 years experience)' AS scenario,
    ROUND(db4ml.predict_linear(:salary_model_id, ARRAY[7.5])::numeric, 2) AS prediction;

\echo '\n=== Model Artifacts ==='
SELECT 
    id,
    ROUND((artifact->>'intercept')::numeric, 4) AS intercept,
    jsonb_pretty(artifact->'weights') AS weights
FROM db4ml.models
ORDER BY created_at DESC;