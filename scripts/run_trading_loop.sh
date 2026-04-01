#!/usr/bin/env bash
# Minimal wrapper for trading loop
set -euo pipefail

cd "$(dirname "$0")/.."

# Load env vars
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

SLEEP="${TRADING_LOOP_INTERVAL:-300}"

echo "Starting trading loop"
echo "Interval: ${SLEEP}s"
echo ""

while :; do
    echo "[$(date)] Trading iteration starting..."
    cat prompts/trading_loop.md | claude --continue 2>&1 | tail -20
    echo "[$(date)] Sleeping ${SLEEP}s..."
    sleep $SLEEP
done
