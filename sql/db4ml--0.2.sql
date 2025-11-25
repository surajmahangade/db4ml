-- db4ml--1.0.sql
-- This file defines the database objects for the db4ml extension.

-- Complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION db4ml" to load this file. \quit

-- NOTE: The CREATE SCHEMA db4ml; command is typically handled automatically
-- by the extension's control file when the 'schema' parameter is set.

---
-- 1. Table to store trained models and metrics
---
CREATE TABLE db4ml.models (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    algo_type TEXT,        -- classifier, regressor, clusterer
    algo_name TEXT,        -- e.g., 'linear_regression', 'k_means_clustering'
    model_data BYTEA,      -- The pickled model
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

---
-- 2. Job Queue for background training
---
CREATE TABLE db4ml.jobs (
    id BIGSERIAL PRIMARY KEY,
    sql_query TEXT NOT NULL,
    target_column TEXT,    -- target_column is NULL for unsupervised (clustering)
    algo_name TEXT NOT NULL, -- The algorithm to run: e.g., 'linear_regression'
    status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    error_msg TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

---
-- 3. The User-Facing Function (C)
-- Registers the background worker and inserts a job into the queue.
---
CREATE FUNCTION db4ml.train_async(sql_query text, target_col text, algo_name text)
RETURNS bigint
AS 'MODULE_PATHNAME', 'db4ml_launch_training'
LANGUAGE C STRICT;

---
-- 4. The Worker Payload (PL/Python)
-- This function is called by the C background worker to run the training job.
---
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

# Sklearn Imports
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, davies_bouldin_score

# --- Start Job Processing ---

# Update status to processing immediately
try:
    plan_start = plpy.prepare("UPDATE db4ml.jobs SET status='processing' WHERE id=$1", ["bigint"])
    plpy.execute(plan_start, [job_id])
except Exception as e:
    plpy.warning(f"Failed to set job status to processing: {e}")

try:
    # 1. Fetch Job Details
    plan = plpy.prepare("SELECT sql_query, target_column, algo_name FROM db4ml.jobs WHERE id = $1", ["bigint"])
    res = plpy.execute(plan, [job_id])
    if not res:
        plpy.error(f"Job {job_id} not found")

    sql_query = res[0]["sql_query"]
    target_col = res[0]["target_column"]
    algo_name = res[0]["algo_name"].lower() # Ensure consistency
    
    # --- Define model and type based on algo_name ---
    
    model = None
    algo_type = None
    
    # REGRESSION MODELS
    if algo_name == 'linear_regression':
        model = LinearRegression()
        algo_type = 'regressor'
    elif algo_name == 'random_forest_regressor':
        model = RandomForestRegressor(random_state=42)
        algo_type = 'regressor'
    elif algo_name == 'gradient_boosting_regressor':
        model = GradientBoostingRegressor(random_state=42)
        algo_type = 'regressor'
        
    # CLASSIFICATION MODELS
    elif algo_name == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
        algo_type = 'classifier'
    elif algo_name == 'k_nearest_neighbors':
        model = KNeighborsClassifier(n_neighbors=5)
        algo_type = 'classifier'
        
    # UNSUPERVISED MODELS (Clustering)
    elif algo_name == 'k_means_clustering':
        model = KMeans(n_clusters=3, n_init='auto', random_state=42) # Default 3 clusters
        algo_type = 'clusterer'
    else:
        raise Exception(f"Unsupported algorithm name: {algo_name}")
    
    # --- Data Fetch and Preprocessing ---
    
    data_res = plpy.execute(sql_query)
    if not data_res:
        raise Exception("Query returned no data")
    
    df = pd.DataFrame.from_records(data_res)
    # Drop non-numeric columns and missing values for simple demonstration
    df = df.select_dtypes(include=[np.number, 'bool_']).dropna() 
    
    # Convert boolean to int if present, for compatibility
    for col in df.select_dtypes(include=['bool_']).columns:
        df[col] = df[col].astype(int)
        
    # --- Splitting Data (Skipped for Unsupervised) ---
    if algo_type == 'clusterer':
        X_train = df # Use all data as features
        
        # 4. Train (Clusterer)
        model.fit(X_train)
        
        # 5. Metrics (Clustering)
        cluster_labels = model.predict(X_train)
        metrics = {
            'n_clusters': model.n_clusters,
            'davies_bouldin_score': davies_bouldin_score(X_train, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 'N/A'
        }
        
    else: # Regressor or Classifier
        if target_col is None or target_col not in df.columns:
            raise Exception(f"Target column '{target_col}' not found or is missing.")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Train (Regressor/Classifier)
        model.fit(X_train, y_train)
        
        # 5. Metrics (Regressor/Classifier)
        y_pred = model.predict(X_test)
        metrics = {}
        
        if algo_type == 'classifier':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        elif algo_type == 'regressor':
            metrics['r2_score'] = r2_score(y_test, y_pred)
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            
    # --- Saving Results ---
    
    # 6. Pickle the model
    model_bytes = pickle.dumps(model)
    
    # 7. Save to models table
    plan_ins = plpy.prepare(
        "INSERT INTO db4ml.models (name, algo_type, algo_name, model_data, metrics) VALUES ($1, $2, $3, $4, $5)",
        ["text", "text", "text", "bytea", "jsonb"]
    )
    plpy.execute(plan_ins, [
        f"{algo_name}_Job_{job_id}", 
        algo_type, 
        algo_name, 
        model_bytes, 
        json.dumps(metrics)
    ])
    
    # 8. Mark Job Complete
    plpy.execute(plpy.prepare("UPDATE db4ml.jobs SET status='completed', finished_at=NOW() WHERE id=$1", ["bigint"]), [job_id])

except Exception as e:
    # Error handling to log failure and traceback
    tb = traceback.format_exc()
    try:
        plan_err = plpy.prepare("UPDATE db4ml.jobs SET status='failed', error_msg=$1, finished_at=NOW() WHERE id=$2", ["text", "bigint"])
        plpy.execute(plan_err, [tb, job_id])
    except Exception as write_err:
        plpy.error(f"CRITICAL: Could not write error to table. Original error: {e}")

$$;

---
-- 5. Prediction Function (PL/Python)
-- Runs in foreground, unpickles model, predicts
---
CREATE OR REPLACE FUNCTION db4ml.predict(model_id bigint, features float8[])
RETURNS float8
LANGUAGE plpython3u
AS $$
import plpy
import pickle
import numpy as np
from sklearn.base import is_classifier

# 1. Fetch Model
plan = plpy.prepare("SELECT model_data, algo_type FROM db4ml.models WHERE id = $1", ["bigint"])
res = plpy.execute(plan, [model_id])
if not res:
    plpy.error("Model not found")

model_data = res[0]["model_data"]
algo_type = res[0]["algo_type"]

# 2. Unpickle
model = pickle.loads(model_data)

# 3. Predict
# Reshape input because sklearn expects 2D array
f_arr = np.array(features).reshape(1, -1)

if algo_type == 'classifier':
    # For classification, we usually want the class (0 or 1)
    prediction = model.predict(f_arr)
elif algo_type == 'clusterer':
    # For clustering, we predict the cluster ID (integer)
    prediction = model.predict(f_arr)
else:
    # For regression, we predict the float value
    prediction = model.predict(f_arr)

# Return the first (and only) prediction as a float
# Note: Prediction results for classification/clustering will be cast to float8 in SQL, 
# which is generally fine for single predictions (0.0, 1.0, 2.0, etc.)
return float(prediction[0])
$$;