-- sql/db4ml--0.2.sql
CREATE SCHEMA IF NOT EXISTS db4ml;

-- model registry
CREATE TABLE IF NOT EXISTS db4ml.models(
  id           bigserial PRIMARY KEY,
  created_at   timestamptz NOT NULL DEFAULT now(),
  kind         text        NOT NULL,              -- 'linear'
  query_text   text        NOT NULL,              -- training SELECT
  n_features   int         NOT NULL,
  artifact     jsonb       NOT NULL,              -- {"weights":[...], "intercept":b0}
  params       jsonb       NOT NULL DEFAULT '{}'::jsonb
);

-- train/test metrics
CREATE TABLE IF NOT EXISTS db4ml.metrics(
  id           bigserial PRIMARY KEY,
  model_id     bigint      NOT NULL REFERENCES db4ml.models(id) ON DELETE CASCADE,
  created_at   timestamptz NOT NULL DEFAULT now(),
  train_rows   int         NOT NULL,
  test_rows    int         NOT NULL,
  train_r2     double precision NOT NULL,
  test_r2      double precision NOT NULL
);

-- Train: user supplies a SELECT whose last column is y; preceding columns are X
-- Example call:
--   SELECT db4ml.train_linear($$SELECT x1,x2,y FROM t$$, 0.2, '{}');
CREATE OR REPLACE FUNCTION db4ml.train_linear(train_sql text,
                                              test_ratio double precision DEFAULT 0.2,
                                              params jsonb DEFAULT '{}'::jsonb)
RETURNS bigint
AS 'db4ml', 'db4ml_train_linear'
LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;

-- Predict for a single row of features; features length must match n_features
-- Example call:
--   SELECT db4ml.predict_linear(42, ARRAY[1.2, 3.4]);
CREATE OR REPLACE FUNCTION db4ml.predict_linear(model_id bigint,
                                                features double precision[])
RETURNS double precision
AS 'db4ml', 'db4ml_predict_linear'
LANGUAGE C STABLE STRICT PARALLEL SAFE;
