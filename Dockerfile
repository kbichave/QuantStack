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

# ── Layer 1: dependency install (cached unless pyproject.toml/uv.lock change) ──
COPY pyproject.toml uv.lock ./

# ── Layer 2: source code ──
COPY src/ ./src/

# Install all dependencies + editable package in one step.
# BuildKit cache reuses downloaded wheels across rebuilds.
# Install langgraph extras only (no torch/RL for graph services)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e ".[langgraph]"

# Copy scripts (entrypoint, migrations, etc.)
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /data/quantstack

# Set data dir to /data so it can be volume-mounted
ENV KILL_SWITCH_SENTINEL=/data/quantstack/KILL_SWITCH_ACTIVE

# Copy entrypoint
COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
