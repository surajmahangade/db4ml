-- sql/db4ml--0.1.sql
CREATE SCHEMA IF NOT EXISTS db4ml;
CREATE TABLE IF NOT EXISTS db4ml.models(
  id bigserial PRIMARY KEY,
  created_at timestamptz NOT NULL DEFAULT now(),
  table_name text NOT NULL,
  target_col text NOT NULL,
  params jsonb NOT NULL DEFAULT '{}'::jsonb,
  kind text NOT NULL DEFAULT 'mean',
  artifact jsonb NOT NULL,
  metrics jsonb NOT NULL
);

CREATE FUNCTION db4ml.train_model(src regclass, target text, params jsonb DEFAULT '{}'::jsonb)
RETURNS bigint
AS 'db4ml', 'db4ml_train_model'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION db4ml.predict(model_id bigint, x double precision)
RETURNS double precision
AS 'db4ml', 'db4ml_predict_scalar'
LANGUAGE C STABLE STRICT;
