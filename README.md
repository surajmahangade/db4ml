# db4ml: Minimal Postgres extension (PGXS) for train/predict

This repo shows how to build and use a PostgreSQL C extension (`db4ml`) using PGXS inside Docker. It provides:
- `db4ml.train_model(src regclass, target text, params jsonb)`
- `db4ml.predict(model_id bigint, x double precision)`

## Prerequisites
- Docker
- Project layout:
  ```text
  db4ml/
  ├─ Dockerfile
  ├─ Makefile
  ├─ db4ml.c
  ├─ db4ml.control
  └─ sql/
     └─ db4ml--0.1.sql
  ```

## 1) Build the Postgres dev image (PG16 + headers)
```bash
docker build -t pgdev:16 .
```

**Dockerfile** should install `postgresql-server-dev-16` and toolchain.

## 2) Run the container
Creates a named volume `pgdata` for the data directory and mounts your source.
```bash
docker run -d --name pg16dev \
  -e POSTGRES_USER=dev -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=dev \
  -v pgdata:/var/lib/postgresql/data \
  -v "$(pwd)":/src/db4ml \
  -p 5432:5432 pgdev:16

# wait for readiness
docker logs -f pg16dev
```

## 3) Build the extension inside the container
```bash
# build as postgres user
docker exec -it pg16dev bash -lc "cd /src/db4ml && make clean && make"

# install to server directories as root
docker exec -it -u 0 pg16dev bash -lc "cd /src/db4ml && make install"
```

## 4) Create the extension in the DB
```bash
docker exec -it pg16dev psql -U dev -d dev -c "CREATE EXTENSION db4ml;"
# verify
docker exec -it pg16dev psql -U dev -d dev -c "\dx db4ml"
docker exec -it pg16dev psql -U dev -d dev -c "\df+ db4ml.*"
```

## 5) Smoke test
```bash
docker exec -it pg16dev psql -U dev -d dev -c "CREATE TABLE IF NOT EXISTS public.t(y double precision);
INSERT INTO public.t SELECT generate_series(1,1000);
SELECT db4ml.train_model('public.t'::regclass, 'y', '{}'::jsonb) AS model_id;
SELECT db4ml.predict(1, 123.0) AS y_hat;"
```

Expected: `y_hat` equals the mean of `public.t.y` (toy model).

## 6) Iteration cycle
When you edit `db4ml.c` or SQL:
```bash
# rebuild
docker exec -it pg16dev bash -lc "cd /src/db4ml && make clean && make"
# reinstall as root
docker exec -it -u 0 pg16dev bash -lc "cd /src/db4ml && make install"
# restart to reload the .so if symbols changed
docker restart pg16dev
```

If you change only SQL objects and not C symbols, you can drop/re-create:
```bash
docker exec -it pg16dev psql -U dev -d dev -c "DROP EXTENSION IF EXISTS db4ml CASCADE; CREATE EXTENSION db4ml;"
```

## 7) Git setup
```bash
git init
printf "*.bs\n*.c\n*.so\n" >> .gitignore
git add .
git commit -m "Initial commit: db4ml extension"
git branch -M main
git remote add origin <YOUR_REMOTE_URL>
git push -u origin main
```

## 8) Troubleshooting
- **Permission denied on `make install`**: run install as root inside the container:
  ```bash
  docker exec -it -u 0 pg16dev bash -lc "cd /src/db4ml && make install"
  ```
- **`version to install must be specified`**: ensure `db4ml.control` contains `default_version = '0.1'` and `sql/db4ml--0.1.sql` exists.
- **`INSERT is not allowed in a non-volatile function`**:
  - SQL binding must be `LANGUAGE C VOLATILE PARALLEL UNSAFE`.
  - In C, call `SPI_execute_with_args(..., /*read_only=*/false, ...)` for writes.
  - Restart the container to reload the updated `.so`: `docker restart pg16dev`.
- **Headers missing**: image must have `postgresql-server-dev-16`. Verify:
  ```bash
  docker exec -it pg16dev bash -lc "dpkg -l | grep postgresql-server-dev-16"
  ```

## 9) Clean up
```bash
docker rm -f pg16dev
docker volume rm pgdata
```

## File locations (inside container)
```bash
pg_config --pkglibdir   # → /usr/lib/postgresql/16/lib/
pg_config --sharedir    # → /usr/share/postgresql/16/
# extension files:
#   $(sharedir)/extension/db4ml.control
#   $(sharedir)/extension/db4ml--0.1.sql
#   $(pkglibdir)/db4ml.so
```
