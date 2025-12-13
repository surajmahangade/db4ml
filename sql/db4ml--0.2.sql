-- db4ml--1.0.sql
\echo Use "CREATE EXTENSION db4ml" to load this file. \quit

-- TABLES (Reverted to original schema to match C code expectations)
CREATE TABLE db4ml.models (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    algo_type TEXT,
    algo_name TEXT,
    model_data BYTEA,
    metrics JSONB,
    training_data_sql TEXT,
    training_row_count INT,
    feature_columns TEXT[],
    target_column TEXT,
    data_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    use_count INT DEFAULT 0
);

CREATE TABLE db4ml.jobs (
    id BIGSERIAL PRIMARY KEY,
    sql_query TEXT NOT NULL,
    target_column TEXT,
    algo_name TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    error_msg TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    result_model_id BIGINT REFERENCES db4ml.models(id)
);

CREATE TABLE db4ml.outliers (
    id BIGSERIAL PRIMARY KEY,
    job_id BIGINT REFERENCES db4ml.jobs(id),
    detection_method TEXT,
    row_data JSONB,
    outlier_score FLOAT8,
    detected_at TIMESTAMPTZ DEFAULT NOW()
);

-- C WRAPPER FUNCTIONS (Reverted signatures)
CREATE FUNCTION db4ml.train_async(sql_query text, target_col text, algo_name text)
RETURNS bigint
AS 'MODULE_PATHNAME', 'db4ml_launch_training'
LANGUAGE C STRICT;

CREATE FUNCTION db4ml.detect_outliers(sql_query text, method text DEFAULT 'isolation_forest')
RETURNS bigint
AS 'MODULE_PATHNAME', 'db4ml_detect_outliers'
LANGUAGE C STRICT;

-- WORKER FUNCTION (Updated to parse 'algo:mode')
CREATE OR REPLACE FUNCTION db4ml.internal_run_job(job_id bigint)
RETURNS void
LANGUAGE plpython3u
AS $$
import plpy
import traceback
import json 
import pickle
import pandas as pd
import numpy as np
import hashlib
import time

# Import incremental learning algorithms
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set status to processing
try:
    plpy.execute(plpy.prepare("UPDATE db4ml.jobs SET status='processing' WHERE id=$1", ["bigint"]), [job_id])
except:
    pass

try:
    # 1. Fetch Job Details
    plan = plpy.prepare("SELECT sql_query, target_column, algo_name FROM db4ml.jobs WHERE id = $1", ["bigint"])
    res = plpy.execute(plan, [job_id])
    if not res:
        plpy.error(f"Job {job_id} not found")

    sql_query = res[0]["sql_query"]
    target_col = res[0]["target_column"]
    raw_algo_input = res[0]["algo_name"].lower()
    
    # 2. Parse Mode from Algo Name (e.g. "sgd_regressor:partial_fit")
    if ':' in raw_algo_input:
        algo_name, mode = raw_algo_input.split(':')
    else:
        algo_name = raw_algo_input
        mode = 'fit' # Default

    # 3. Handle Outliers
    if algo_name.startswith('outlier_'):
        # (Outlier logic stub - kept simple for this update)
        plpy.execute(plpy.prepare("UPDATE db4ml.jobs SET status='completed' WHERE id=$1", ["bigint"]), [job_id])
        return

    # 4. Fetch Data
    data_res = plpy.execute(sql_query)
    if not data_res:
        raise Exception("Query returned no data")
    
    df = pd.DataFrame.from_records(data_res)
    original_row_count = len(df)
    
    # Preprocessing
    df = df.select_dtypes(include=[np.number, 'bool_']).dropna()
    if df.empty: raise Exception("No numeric data")
    
    # Hashing
    data_hash = hashlib.md5(df.to_json(orient='records').encode()).hexdigest()
    
    # 5. Algorithm Selection
    model = None
    algo_type = None
    scaler = StandardScaler() # Critical for SGD

    # Parse Cluster Count
    cluster_count = 3
    if target_col and target_col.isdigit():
        cluster_count = int(target_col)

    if algo_name == 'sgd_regressor':
        model = SGDRegressor(random_state=42)
        algo_type = 'regressor'
    elif algo_name == 'minibatch_kmeans':
        model = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, batch_size=256)
        algo_type = 'clusterer'
    else:
        # Fallback to standard models if needed, but for this test we focus on SGD
        raise Exception(f"Algorithm {algo_name} not supported for comparison test")

    # 6. Training Execution
    start_time = time.time()
    
    if algo_type == 'clusterer':
        X = scaler.fit_transform(df)
        feature_cols = list(df.columns)
        
        if mode == 'partial_fit':
            batch_size = 200
            for i in range(0, len(X), batch_size):
                model.partial_fit(X[i:i+batch_size])
        else:
            model.fit(X)
            
        metrics = {'n_clusters': cluster_count}
        
    else: # Supervised
        if target_col not in df.columns: raise Exception("Target column missing")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        feature_cols = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling is mandatory for SGD
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        if mode == 'partial_fit':
            batch_size = 100
            # Simple chunking loop
            for i in range(0, X_train.shape[0], batch_size):
                end = min(i + batch_size, X_train.shape[0])
                model.partial_fit(X_train[i:end], y_train.iloc[i:end])
        else:
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'r2_score': float(r2_score(y_test, y_pred))
        }

    training_time = time.time() - start_time
    metrics['training_time_sec'] = float(training_time)
    
    # 7. Save Model
    # We bundle scaler with model for correct predictions later
    pipeline = {'model': model, 'scaler': scaler} 
    model_bytes = pickle.dumps(pipeline)
    
    plan_ins = plpy.prepare(
        """INSERT INTO db4ml.models 
        (name, algo_type, algo_name, model_data, metrics, training_data_sql, 
         training_row_count, feature_columns, target_column, data_hash) 
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING id""",
        ["text", "text", "text", "bytea", "jsonb", "text", "int4", "text[]", "text", "text"]
    )
    
    # Note: We save raw_algo_input (which includes :mode) into algo_name column
    # so we can distinguish them in the comparison SQL later.
    model_res = plpy.execute(plan_ins, [
        f"{algo_name}_{mode}", algo_type, raw_algo_input, model_bytes, 
        json.dumps(metrics), sql_query, original_row_count, feature_cols, target_col, data_hash
    ])
    
    # 8. Finish Job
    plpy.execute(plpy.prepare(
        "UPDATE db4ml.jobs SET status='completed', finished_at=NOW(), result_model_id=$1 WHERE id=$2", 
        ["bigint", "bigint"]), [model_res[0]["id"], job_id])

except Exception as e:
    plpy.execute(plpy.prepare(
        "UPDATE db4ml.jobs SET status='failed', error_msg=$1, finished_at=NOW() WHERE id=$2", 
        ["text", "bigint"]), [traceback.format_exc(), job_id])
$$;

-- PREDICTION FUNCTION (Fixed to use pipeline)
CREATE OR REPLACE FUNCTION db4ml.predict(model_id bigint, features float8[])
RETURNS float8
LANGUAGE plpython3u
AS $$
import plpy
import pickle
import numpy as np

res = plpy.execute(plpy.prepare("SELECT model_data FROM db4ml.models WHERE id = $1", ["bigint"]), [model_id])
if not res: plpy.error("Model not found")

pipeline = pickle.loads(res[0]["model_data"])
f_arr = np.array(features).reshape(1, -1)
# Scale using the training scaler
f_scaled = pipeline['scaler'].transform(f_arr)
return float(pipeline['model'].predict(f_scaled)[0])
$$;