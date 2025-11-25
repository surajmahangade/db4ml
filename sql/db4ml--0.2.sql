-- db4ml--1.0.sql
\echo Use "CREATE EXTENSION db4ml" to load this file. \quit

--------------------------------------------------------------------------------
-- TABLES
--------------------------------------------------------------------------------

-- Table to store trained models with comprehensive metadata
CREATE TABLE db4ml.models (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    algo_type TEXT,
    algo_name TEXT,
    model_data BYTEA,
    metrics JSONB,
    training_data_sql TEXT,    -- Track which SQL was used for training
    training_row_count INT,    -- Number of rows used
    feature_columns TEXT[],    -- Columns used as features
    target_column TEXT,        -- Target column (NULL or N_CLUSTERS for unsupervised)
    data_hash TEXT,            -- Hash of training data for tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    use_count INT DEFAULT 0
);

-- Job Queue for background training
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

-- Outlier Detection Results Table
CREATE TABLE db4ml.outliers (
    id BIGSERIAL PRIMARY KEY,
    job_id BIGINT REFERENCES db4ml.jobs(id),
    detection_method TEXT,
    row_data JSONB,
    outlier_score FLOAT8,
    detected_at TIMESTAMPTZ DEFAULT NOW()
);

--------------------------------------------------------------------------------
-- C WRAPPER FUNCTIONS
--------------------------------------------------------------------------------

-- User-Facing Training Function
CREATE FUNCTION db4ml.train_async(sql_query text, target_col text, algo_name text)
RETURNS bigint
AS 'MODULE_PATHNAME', 'db4ml_launch_training'
LANGUAGE C STRICT;

-- User-Facing Outlier Detection Function
CREATE FUNCTION db4ml.detect_outliers(sql_query text, method text DEFAULT 'isolation_forest')
RETURNS bigint
AS 'MODULE_PATHNAME', 'db4ml_detect_outliers'
LANGUAGE C STRICT;

--------------------------------------------------------------------------------
-- PL/PYTHON WORKER (db4ml.internal_run_job)
--------------------------------------------------------------------------------

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

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Update status to processing
try:
    plan_start = plpy.prepare("UPDATE db4ml.jobs SET status='processing' WHERE id=$1", ["bigint"])
    plpy.execute(plan_start, [job_id])
except Exception as e:
    plpy.warning(f"Failed to set job status to processing: {e}")

try:
    # Fetch Job Details
    plan = plpy.prepare("SELECT sql_query, target_column, algo_name FROM db4ml.jobs WHERE id = $1", ["bigint"])
    res = plpy.execute(plan, [job_id])
    if not res:
        plpy.error(f"Job {job_id} not found")

    sql_query = res[0]["sql_query"]
    target_col = res[0]["target_column"]
    algo_name = res[0]["algo_name"].lower()
    
    # Fetch data
    data_res = plpy.execute(sql_query)
    if not data_res:
        raise Exception("Query returned no data")
    
    df = pd.DataFrame.from_records(data_res)
    original_row_count = len(df)
    
    # Handle Outlier Detection Separately
    if algo_name.startswith('outlier_'):
        method = algo_name.replace('outlier_', '')
        
        # Prepare numeric data
        df_numeric = df.select_dtypes(include=[np.number, 'bool_']).dropna()
        for col in df_numeric.select_dtypes(include=['bool_']).columns:
            df_numeric[col] = df_numeric[col].astype(int)
        
        if df_numeric.empty:
            raise Exception("No numeric columns found for outlier detection")
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)
        
        outliers = []
        scores = []
        
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=0.1, random_state=42)
            predictions = detector.fit_predict(X_scaled)
            scores = detector.score_samples(X_scaled)
            outliers = predictions == -1
            
        elif method == 'lof':
            detector = LocalOutlierFactor(contamination=0.1)
            predictions = detector.fit_predict(X_scaled)
            scores = detector.negative_outlier_factor_
            outliers = predictions == -1
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(X_scaled, axis=0))
            outliers = (z_scores > 3).any(axis=1)
            scores = z_scores.max(axis=1)
        else:
            raise Exception(f"Unsupported outlier detection method: {method}")
        
        # Store outliers in database
        outlier_indices = np.where(outliers)[0]
        plan_outlier = plpy.prepare(
            "INSERT INTO db4ml.outliers (job_id, detection_method, row_data, outlier_score) VALUES ($1, $2, $3, $4)",
            ["bigint", "text", "jsonb", "float8"]
        )
        
        for idx in outlier_indices:
            row_dict = df.iloc[idx].to_dict()
            # Convert numpy types to Python native types
            for k, v in row_dict.items():
                if isinstance(v, (np.integer, np.floating)):
                    row_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
                
            plpy.execute(plan_outlier, [
                job_id,
                method,
                json.dumps(row_dict),
                float(scores[idx]) if idx < len(scores) else 0.0
            ])
        
        # Mark job complete
        metrics = {
            'method': method,
            'total_rows': len(df),
            'outliers_found': int(outliers.sum()),
            'outlier_percentage': float(outliers.sum() / len(df) * 100)
        }
        
        plpy.execute(plpy.prepare(
            "UPDATE db4ml.jobs SET status='completed', finished_at=NOW() WHERE id=$1", 
            ["bigint"]), [job_id])
        
        return  # Exit early for outlier detection
    
    # Regular ML Model Training
    df = df.select_dtypes(include=[np.number, 'bool_']).dropna()
    for col in df.select_dtypes(include=['bool_']).columns:
        df[col] = df[col].astype(int)
    
    if df.empty:
        raise Exception("No numeric data available after preprocessing")
    
    # Calculate data hash for tracking
    data_str = df.to_json(orient='records')
    data_hash = hashlib.md5(data_str.encode()).hexdigest()
    
    # Define model
    model = None
    algo_type = None

    # --- NEW: Safe parsing of the cluster count from the target_col argument ---
    cluster_count = 3 # Default if target_col is NULL or invalid
    if target_col is not None:
        if target_col.isdigit():
            try:
                cluster_count = int(target_col)
            except ValueError:
                plpy.warning(f"Target column argument '{target_col}' not a valid cluster count. Using default: 3.")
        else:
             # If it's not a digit, assume it's a valid column name for supervised learning
             pass 
    # --- END NEW LOGIC ---

    if algo_name == 'linear_regression':
        model = LinearRegression()
        algo_type = 'regressor'
    elif algo_name == 'random_forest_regressor':
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        algo_type = 'regressor'
    elif algo_name == 'gradient_boosting_regressor':
        model = GradientBoostingRegressor(random_state=42)
        algo_type = 'regressor'
    elif algo_name == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
        algo_type = 'classifier'
    elif algo_name == 'k_nearest_neighbors':
        model = KNeighborsClassifier(n_neighbors=5)
        algo_type = 'classifier'
    elif algo_name == 'k_means_clustering':
        # --- MODIFIED: Use the parsed cluster_count ---
        model = KMeans(n_clusters=cluster_count, n_init='auto', random_state=42)
        algo_type = 'clusterer'
    elif algo_name == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
        algo_type = 'clusterer'
    else:
        raise Exception(f"Unsupported algorithm: {algo_name}")
    
    # Training logic
    if algo_type == 'clusterer':
        X_train = df
        feature_cols = list(df.columns)
        model.fit(X_train)
        
        cluster_labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        
        metrics = {
            'n_clusters': int(n_clusters),
            'davies_bouldin_score': davies_bouldin_score(X_train, cluster_labels) if n_clusters > 1 else 'N/A',
            'training_rows': len(X_train)
        }
        
    else:  # Supervised
        # target_col is now expected to be a column name or the number of clusters (which we filtered above).
        # Re-using the variable name `target_col` here for the column name is fine, as long as it's not a cluster count.
        if target_col is None or target_col not in df.columns:
            raise Exception(f"Target column '{target_col}' required for {algo_type}. If this was a cluster count, it failed validation.")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        feature_cols = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {'training_rows': len(X_train), 'test_rows': len(X_test)}
        
        if algo_type == 'classifier':
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            metrics['train_accuracy'] = float(accuracy_score(y_train, model.predict(X_train)))
        elif algo_type == 'regressor':
            metrics['r2_score'] = float(r2_score(y_test, y_pred))
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
    
    # Save model with metadata
    model_bytes = pickle.dumps(model)
    
    plan_ins = plpy.prepare(
        """INSERT INTO db4ml.models 
        (name, algo_type, algo_name, model_data, metrics, training_data_sql, 
         training_row_count, feature_columns, target_column, data_hash) 
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING id""",
        ["text", "text", "text", "bytea", "jsonb", "text", "int4", "text[]", "text", "text"]
    )
    
    model_res = plpy.execute(plan_ins, [
        f"{algo_name}_Job_{job_id}",
        algo_type,
        algo_name,
        model_bytes,
        json.dumps(metrics),
        sql_query,
        original_row_count,
        feature_cols,
        target_col,
        data_hash
    ])
    
    model_id = model_res[0]["id"]
    
    # Update job with model reference
    plpy.execute(plpy.prepare(
        "UPDATE db4ml.jobs SET status='completed', finished_at=NOW(), result_model_id=$1 WHERE id=$2", 
        ["bigint", "bigint"]), [model_id, job_id])

except Exception as e:
    tb = traceback.format_exc()
    try:
        plan_err = plpy.prepare(
            "UPDATE db4ml.jobs SET status='failed', error_msg=$1, finished_at=NOW() WHERE id=$2", 
            ["text", "bigint"])
        plpy.execute(plan_err, [tb, job_id])
    except Exception as write_err:
        plpy.error(f"CRITICAL: Could not write error. Original: {e}")
$$;

--------------------------------------------------------------------------------
-- PREDICTION FUNCTIONS
--------------------------------------------------------------------------------

-- Enhanced Prediction Function
CREATE OR REPLACE FUNCTION db4ml.predict(model_id bigint, features float8[])
RETURNS float8
LANGUAGE plpython3u
AS $$
import plpy
import pickle
import numpy as np

plan = plpy.prepare("SELECT model_data, algo_type FROM db4ml.models WHERE id = $1", ["bigint"])
res = plpy.execute(plan, [model_id])
if not res:
    plpy.error("Model not found")

model_data = res[0]["model_data"]
algo_type = res[0]["algo_type"]

model = pickle.loads(model_data)
f_arr = np.array(features).reshape(1, -1)
prediction = model.predict(f_arr)

# Update usage statistics
plpy.execute(plpy.prepare(
    "UPDATE db4ml.models SET last_used_at=NOW(), use_count=use_count+1 WHERE id=$1",
    ["bigint"]), [model_id])

return float(prediction[0])
$$;

-- Batch Prediction Function
CREATE OR REPLACE FUNCTION db4ml.predict_batch(model_id bigint, sql_query text)
RETURNS TABLE(row_num int, prediction float8)
LANGUAGE plpython3u
AS $$
import plpy
import pickle
import numpy as np
import pandas as pd

plan = plpy.prepare("SELECT model_data, feature_columns FROM db4ml.models WHERE id = $1", ["bigint"])
res = plpy.execute(plan, [model_id])
if not res:
    plpy.error("Model not found")

model = pickle.loads(res[0]["model_data"])
expected_features = res[0]["feature_columns"]

data_res = plpy.execute(sql_query)
df = pd.DataFrame.from_records(data_res)

# Ensure columns match training features
if expected_features:
    missing_cols = set(expected_features) - set(df.columns)
    if missing_cols:
        plpy.error(f"Missing columns: {missing_cols}")
    df = df[expected_features]

df_numeric = df.select_dtypes(include=[np.number, 'bool_']).fillna(0)
predictions = model.predict(df_numeric)

return [(i+1, float(pred)) for i, pred in enumerate(predictions)]
$$;

--------------------------------------------------------------------------------
-- VIEWS
--------------------------------------------------------------------------------

-- View outliers detected in the most recent job
CREATE OR REPLACE VIEW db4ml.recent_outliers AS
SELECT o.*, j.sql_query, j.created_at as job_created
FROM db4ml.outliers o
JOIN db4ml.jobs j ON o.job_id = j.id
ORDER BY o.detected_at DESC;

-- View model performance metrics
CREATE OR REPLACE VIEW db4ml.model_performance AS
SELECT 
    id, name, algo_type, algo_name,
    metrics->>'accuracy' as accuracy,
    metrics->>'r2_score' as r2_score,
    metrics->>'mse' as mse,
    training_row_count,
    created_at,
    use_count,
    last_used_at
FROM db4ml.models
ORDER BY created_at DESC;