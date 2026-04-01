#!/usr/bin/env bash
# Minimal wrapper for research loop (since Ralph doesn't support file-based prompts)
set -euo pipefail

cd "$(dirname "$0")/.."

# Load env vars
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

SLEEP="${RESEARCH_LOOP_INTERVAL:-120}"

echo "Starting research loop (RESEARCH_SYMBOL_OVERRIDE=${RESEARCH_SYMBOL_OVERRIDE:-auto})"
echo "Interval: ${SLEEP}s"
echo ""

while :; do
    echo "[$(date)] Research iteration starting..."
    cat prompts/research_loop.md | claude --continue 2>&1 | tail -20
    echo "[$(date)] Sleeping ${SLEEP}s..."
    sleep $SLEEP
done
