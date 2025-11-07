# db4ml: PostgreSQL extension (PGXS) for SQL-driven Linear Regression (v0.2)

This repo shows how to build and use a PostgreSQL C extension (`db4ml`) using PGXS inside Docker.
**v0.2** adds: training from an arbitrary `SELECT` (last column = target `y`, others = features `X`),
a persistent model registry, and a `metrics` table with R².

It provides:
- `db4ml.train_linear(train_sql text, test_ratio double precision DEFAULT 0.2, params jsonb DEFAULT '{}'::jsonb) RETURNS bigint`
- `db4ml.predict_linear(model_id bigint, features double precision[]) RETURNS double precision`

> Contract for training query: the `SELECT` **must return only `double precision` columns**. Columns 1..M-1 are features, column M is the target.

---

## Prerequisites
- Docker (for building and running Postgres + headers)
- Project layout:
```
db4ml/
├─ Dockerfile
├─ Makefile
├─ db4ml.c
├─ db4ml.control          # default_version = '0.2'
└─ sql/
   └─ db4ml--0.2.sql
```

**Dockerfile** must install:
```
apt-get update && apt-get install -y --no-install-recommends \
  build-essential clang make git postgresql-server-dev-16
```

---

## 1) Build the Postgres dev image (PG16 + headers)
```bash
docker build -t pgdev:16 .
```

---

## 2) Run the container
Creates a named volume `pgdata` for the data directory and mounts your source repo.
```bash
docker run -d --name pg16dev \
  -e POSTGRES_USER=dev -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=dev \
  -v pgdata:/var/lib/postgresql/data \
  -v "$(pwd)":/src/db4ml \
  -p 5432:5432 pgdev:16

# wait for readiness
docker logs -f pg16dev
```

---

## 3) Build the extension inside the container
```bash
# build as postgres user
docker exec -it pg16dev bash -lc "cd /src/db4ml && make clean && make"

# install to server directories as root
docker exec -it -u 0 pg16dev bash -lc "cd /src/db4ml && make install"
```

**File locations in the container**:
```bash
pg_config --pkglibdir   # /usr/lib/postgresql/16/lib/
pg_config --sharedir    # /usr/share/postgresql/16/
# Expect:
#   $(pkglibdir)/db4ml.so
#   $(sharedir)/extension/db4ml.control
#   $(sharedir)/extension/db4ml--0.2.sql
```

---

## 4) Create or upgrade the extension in the DB
Default version must be `'0.2'` in `db4ml.control`.

```bash
# fresh load
docker exec -it pg16dev psql -U dev -d dev -c "CREATE EXTENSION db4ml;"

# or if upgrading from older version
docker exec -it pg16dev psql -U dev -d dev -c "ALTER EXTENSION db4ml UPDATE TO '0.2';"

# or reset cleanly
docker exec -it pg16dev psql -U dev -d dev -c "DROP EXTENSION IF EXISTS db4ml CASCADE; CREATE EXTENSION db4ml;"
```

Verify functions:
```bash
docker exec -it pg16dev psql -U dev -d dev -c "\df+ db4ml.*"
```

---

## 5) Usage: train and predict

### 5.1 Sample data
```sql
CREATE TABLE demo (
  x1 double precision,
  x2 double precision,
  y  double precision
);

INSERT INTO demo
SELECT g::float8,
       (g*2.0 + 3.0 + (random()-0.5))::float8,
       (g*0.5 + 1.0 + (random()-0.5))::float8
FROM generate_series(1,1000) AS g;
```

### 5.2 Train
- Last column must be target `y`
- All columns in the `SELECT` must be `double precision`

```sql
SELECT db4ml.train_linear(
  'SELECT x1, x2, y FROM demo'::text,    -- training SELECT
  0.2::double precision,                 -- 20% test split (tail rows)
  '{}'::jsonb                            -- params placeholder
) AS model_id;
```

### 5.3 Inspect
```sql
-- latest model and its JSON artifact
SELECT id, kind, n_features, artifact
FROM db4ml.models
ORDER BY id DESC LIMIT 1;

-- metrics (R²) for that model
SELECT *
FROM db4ml.metrics
ORDER BY id DESC LIMIT 1;
```

### 5.4 Predict
Features array length must equal `n_features`.
```sql
-- replace <id> with your model id returned above
SELECT db4ml.predict_linear(<id>, ARRAY[10.0, 5.0]::float8[]) AS y_hat;
```

---

## 6) Iteration cycle (when editing C or SQL)

When you **edit C code** (`db4ml.c`):
```bash
# rebuild
docker exec -it pg16dev bash -lc "cd /src/db4ml && make clean && make"
# reinstall as root
docker exec -it -u 0 pg16dev bash -lc "cd /src/db4ml && make install"
# restart to ensure a fresh backend loads the new .so
docker restart pg16dev
```

When you **edit only SQL objects** and not C symbols, a full restart is not required:
```bash
docker exec -it pg16dev psql -U dev -d dev -c "DROP EXTENSION IF EXISTS db4ml CASCADE; CREATE EXTENSION db4ml;"
```

---

## 7) Troubleshooting

- **`function db4ml.train_linear(unknown, numeric, unknown) does not exist`**  
  Cast arguments: `(text, double precision, jsonb)`  
  ```sql
  SELECT db4ml.train_linear('SELECT x1,x2,y FROM demo'::text, 0.2::double precision, '{}'::jsonb);
  ```

- **`\df+ db4ml.*` returns zero rows**  
  The extension isn’t loaded or `db4ml--0.2.sql` not installed.  
  - Ensure `db4ml.control` has `default_version = '0.2'`.  
  - Ensure `db4ml--0.2.sql` exists in `sql/` and was installed to `$(sharedir)/extension/`.  
  - Reinstall and restart backends.

- **`INSERT is not allowed in a non-volatile function`**  
  - `db4ml.train_linear` must be `LANGUAGE C VOLATILE PARALLEL UNSAFE`.  
  - In C, your `SPI_execute_with_args(..., /*read_only=*/false, ...)` for INSERT/UPDATE.  
  - Restart container to reload `.so`.

- **Permission denied on `make install`**  
  Install as root inside the container:
  ```bash
  docker exec -it -u 0 pg16dev bash -lc "cd /src/db4ml && make install"
  ```

- **Headers missing**  
  Your image must include `postgresql-server-dev-16`.

---

## 8) Clean up
```bash
docker rm -f pg16dev
docker volume rm pgdata
```

---

## Appendix: expected SQL objects (db4ml--0.2.sql)

```sql
CREATE SCHEMA IF NOT EXISTS db4ml;

CREATE TABLE IF NOT EXISTS db4ml.models(
  id bigserial PRIMARY KEY,
  created_at timestamptz NOT NULL DEFAULT now(),
  kind text NOT NULL,                    -- 'linear'
  query_text text NOT NULL,
  n_features int NOT NULL,
  artifact jsonb NOT NULL,               -- {"intercept":b0,"weights":[...],"n_features":p}
  params jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS db4ml.metrics(
  id bigserial PRIMARY KEY,
  model_id bigint NOT NULL REFERENCES db4ml.models(id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  train_rows int NOT NULL,
  test_rows int NOT NULL,
  train_r2 double precision NOT NULL,
  test_r2 double precision NOT NULL
);

CREATE OR REPLACE FUNCTION db4ml.train_linear(train_sql text,
                                              test_ratio double precision DEFAULT 0.2,
                                              params jsonb DEFAULT '{}'::jsonb)
RETURNS bigint
AS 'db4ml', 'db4ml_train_linear'
LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;

CREATE OR REPLACE FUNCTION db4ml.predict_linear(model_id bigint,
                                                features double precision[])
RETURNS double precision
AS 'db4ml', 'db4ml_predict_linear'
LANGUAGE C STABLE STRICT PARALLEL SAFE;
```
