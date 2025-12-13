# db4ml: PostgreSQL extension (PGXS) for SQL-driven Linear Regression (v0.2)

This repo shows how to build and use a PostgreSQL C extension (`db4ml`) using PGXS inside Docker.
**v0.2** adds: training from an arbitrary `SELECT` (last column = target `y`, others = features `X`),
a persistent model registry, and a `metrics` table with R².

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

## 3) Iteration cycle (when editing C or SQL)

When you **edit C code** (`db4ml.c`):
```bash
docker exec -it pg16dev bash -lc "cd /src/db4ml && make clean && make"
docker exec -it -u 0 pg16dev bash -lc "cd /src/db4ml && make install"
docker restart pg16dev
docker exec -it -u 0 pg16dev bash -c "apt-get update && apt-get install -y postgresql-plpython3-16 python3-pip"
docker exec -it -u 0 pg16dev bash -c "pip3 install pandas scikit-learn --break-system-packages"

```

When you **edit only SQL objects** and not C symbols, a full restart is not required:
```bash
docker exec -it pg16dev psql -U dev -d dev -c "CREATE EXTENSION IF NOT EXISTS plpython3u;"
docker exec -it pg16dev psql -U dev -d dev -c "DROP EXTENSION IF EXISTS db4ml CASCADE; CREATE EXTENSION db4ml;"
docker exec -it pg16dev psql -U dev -d dev -f /src/db4ml/sql/load_data.sql
docker exec -it pg16dev psql -U dev -d dev -f /src/db4ml/sql/test_models.sql
docker exec -it pg16dev psql -U dev -d dev -f /src/db4ml/sql/outlier.sql

```