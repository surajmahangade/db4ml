# Dockerfile
FROM postgres:16

# Install compiler, make, and server headers that provide PGXS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential clang make git \
      postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*

# Optional: create a build user
RUN useradd -m builder && usermod -aG postgres builder
USER postgres

# Default env: database bootstrap
ENV POSTGRES_USER=dev POSTGRES_PASSWORD=dev POSTGRES_DB=dev
