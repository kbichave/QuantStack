#!/usr/bin/env bash
# QuantStack Docker entrypoint

set -euo pipefail

MODE="${1:-api}"

echo "QuantStack starting in mode: ${MODE}"

case "${MODE}" in
  api)
    echo "Starting QuantStack API on :8420"
    exec uvicorn quantstack.api.server:app \
        --host 0.0.0.0 \
        --port 8420 \
        --workers 1 \
        --log-level info
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: docker run ... [api|shell]"
    exit 1
    ;;
esac
