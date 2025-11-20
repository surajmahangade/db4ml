-- Complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION db4ml" to load this file. \quit

-- NOTE: "CREATE SCHEMA" is removed. The .control file handles this automatically.

-- 1. Table to store trained models (pickled sklearn objects)
CREATE TABLE db4ml.models (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    algo_type TEXT,
    model_data BYTEA, -- The pickled model
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Job Queue for background training
CREATE TABLE db4ml.jobs (
    id BIGSERIAL PRIMARY KEY,
    sql_query TEXT NOT NULL,
    target_column TEXT NOT NULL,
    status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    error_msg TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

-- 3. The User-Facing Function (C)
-- Registers the background worker
CREATE FUNCTION db4ml.train_async(sql_query text, target_col text)
RETURNS bigint
AS 'MODULE_PATHNAME', 'db4ml_launch_training'
LANGUAGE C STRICT;

-- 4. The Worker Payload (PL/Python)
-- This is called by the C background worker. It runs inside the DB process.
CREATE OR REPLACE FUNCTION db4ml.internal_run_job(job_id bigint)
RETURNS void
LANGUAGE plpython3u
AS $$
import plpy
import traceback
import json  # <--- ADDED THIS

# Update status to processing immediately
try:
    plan_start = plpy.prepare("UPDATE db4ml.jobs SET status='processing' WHERE id=$1", ["bigint"])
    plpy.execute(plan_start, [job_id])
except Exception as e:
    plpy.error(f"Failed to set job status to processing: {e}")

try:
    # IMPORTS MOVED INSIDE TRY BLOCK
    import pickle
    import pandas as pd
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, accuracy_score
    import io

    # 1. Fetch Job Details
    plan = plpy.prepare("SELECT sql_query, target_column FROM db4ml.jobs WHERE id = $1", ["bigint"])
    res = plpy.execute(plan, [job_id])
    if not res:
        plpy.error(f"Job {job_id} not found")

    sql_query = res[0]["sql_query"]
    target_col = res[0]["target_column"]

    # 2. Fetch Data using pandas
    # We assume the query is valid.
    data_res = plpy.execute(sql_query)
    if not data_res:
        raise Exception("Query returned no data")
    
    # Convert result object to list of dicts for DataFrame
    # plpy result objects support list-like access
    df = pd.DataFrame.from_records(data_res)
    
    # 3. Preprocessing
    if target_col not in df.columns:
        raise Exception(f"Target column '{target_col}' not found in result set. Columns are: {list(df.columns)}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Simple logic: if target is float/int -> Linear, else Logistic
    is_classifier = False
    # Check if dtype is object (string) or if unique values are few (categorical)
    if y.dtype == 'object' or len(y.unique()) < 20: 
        model = LogisticRegression(max_iter=1000) # Increased max_iter for safety
        is_classifier = True
    else:
        model = LinearRegression()
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 4. Train
    model.fit(X_train, y_train)
    
    # 5. Metrics
    y_pred = model.predict(X_test)
    metrics = {}
    if is_classifier:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
    else:
        metrics['r2'] = r2_score(y_test, y_pred)
        
    # 6. Pickle the model
    # Use protocol 4 or higher for better compatibility, but default is usually fine
    model_bytes = pickle.dumps(model)
    
    # 7. Save to models table
    plan_ins = plpy.prepare(
        "INSERT INTO db4ml.models (name, algo_type, model_data, metrics) VALUES ($1, $2, $3, $4)",
        ["text", "text", "bytea", "jsonb"]
    )
    # FIX: Use json.dumps instead of plpy.dumps
    plpy.execute(plan_ins, [f"Job_{job_id}", "classifier" if is_classifier else "regressor", model_bytes, json.dumps(metrics)])
    
    # 8. Mark Job Complete
    plpy.execute(plpy.prepare("UPDATE db4ml.jobs SET status='completed', finished_at=NOW() WHERE id=$1", ["bigint"]), [job_id])

except Exception as e:
    # Catch ALL errors (Imports, SQL, Logic) and log to table
    tb = traceback.format_exc()
    # We create a new plan here to ensure we can write the error even if previous queries failed
    try:
        plan_err = plpy.prepare("UPDATE db4ml.jobs SET status='failed', error_msg=$1, finished_at=NOW() WHERE id=$2", ["text", "bigint"])
        plpy.execute(plan_err, [tb, job_id])
    except Exception as write_err:
        plpy.error(f"CRITICAL: Could not write error to table. Original error: {e}")

$$;

-- 5. Prediction Function (PL/Python)
-- Runs in foreground, unpickles model, predicts
CREATE OR REPLACE FUNCTION db4ml.predict(model_id bigint, features float8[])
RETURNS float8
LANGUAGE plpython3u
AS $$
import plpy
import pickle
import numpy as np

# 1. Fetch Model
plan = plpy.prepare("SELECT model_data FROM db4ml.models WHERE id = $1", ["bigint"])
res = plpy.execute(plan, [model_id])
if not res:
    plpy.error("Model not found")

model_data = res[0]["model_data"]

# 2. Unpickle
model = pickle.loads(model_data)

# 3. Predict
# Reshape input because sklearn expects 2D array
f_arr = np.array(features).reshape(1, -1)
prediction = model.predict(f_arr)

return float(prediction[0])
$$;