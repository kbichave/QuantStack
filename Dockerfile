# QuantStack — single-stage production image
# Single-user, local deployment (no multi-tenancy, no auth)

FROM python:3.11-slim-bookworm

LABEL maintainer="QuantStack"
LABEL description="QuantStack trading platform"

# Optional: inject a corporate CA bundle (e.g. Zscaler) for local builds only.
# Usage: docker build --build-arg EXTRA_CA_CERT="$(cat ~/.zscaler_cert.pem)" .
# In CI this arg is empty — no extra certs are added to the production image.
ARG EXTRA_CA_CERT=""
RUN if [ -n "$EXTRA_CA_CERT" ]; then \
        printf '%s\n' "$EXTRA_CA_CERT" >> /usr/local/share/ca-certificates/corporate.crt \
        && update-ca-certificates; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir "uv>=0.5.0"

WORKDIR /app

# Copy package manifests first for layer caching
COPY pyproject.toml ./

# Copy all package source
COPY packages/ ./packages/

# Install all dependencies
RUN uv pip install --system --no-cache -e ".[all]"

# Create data directories
RUN mkdir -p /data/quant_pod

# Set data dir to /data so it can be volume-mounted
ENV PORTFOLIO_DB_PATH=/data/quant_pod/portfolio.duckdb
ENV PAPER_BROKER_DB_PATH=/data/quant_pod/paper_broker.duckdb
ENV AUDIT_LOG_DB_PATH=/data/quant_pod/audit_log.duckdb
ENV CALIBRATION_DB_PATH=/data/quant_pod/calibration.duckdb
ENV DUCKDB_PATH=/data/quant_pod/knowledge.duckdb
ENV KILL_SWITCH_SENTINEL=/data/quant_pod/KILL_SWITCH_ACTIVE

# Copy entrypoint
COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8420/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
