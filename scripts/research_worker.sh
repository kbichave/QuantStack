#!/usr/bin/env bash
# Research worker loop — run as many parallel instances as needed.
# Usage: bash scripts/research_worker.sh <label> [start_delay_seconds]
#
# label:         "a", "b", etc. — appended to log file name.
# start_delay:   seconds to sleep before first iteration (stagger parallel workers).
#
# Adaptive sleep (bootstrap vs steady-state):
#   Bootstrap (< 5 forward_testing strategies): 5 min fast iterations.
#   Steady-state (>= 5 forward_testing):        30 min maintenance cadence.
#
# Market-aware model routing:
#   Market hours (09:30–16:00 ET): haiku — lightweight refresh, 5 min.
#   After hours:                   sonnet — full research cycle, adaptive sleep.

set -euo pipefail

LABEL="${1:-a}"
START_DELAY="${2:-0}"
LOG_FILE="data/logs/research_loop_${LABEL}.log"

if [[ "$START_DELAY" -gt 0 ]]; then
    echo "[research-${LABEL}] Starting in ${START_DELAY}s..."
    sleep "$START_DELAY"
fi

while :; do
    # Heartbeat — record iteration start
    export HEARTBEAT_ITERATION=$(bash scripts/heartbeat.sh research_loop running 2>/dev/null \
        | grep '^HEARTBEAT_ITERATION=' | cut -d= -f2) || true

    # Market-aware model + sleep selection
    HOUR=$(TZ='America/New_York' date +%H)
    if [[ "$HOUR" -ge 9 && "$HOUR" -lt 16 ]]; then
        MODEL='haiku'
        SLEEP=300
    else
        MODEL='sonnet'
        # Adaptive sleep: fast bootstrap until 5 forward_testing strategies exist
        FT_COUNT=$(python3 -c "
from quantstack.db import open_db
conn = open_db()
n = conn.execute(\"SELECT COUNT(*) FROM strategies WHERE status='forward_testing'\").fetchone()[0]
conn.close()
print(n)
" 2>/dev/null || echo 0)
        if [[ "$FT_COUNT" -lt 5 ]]; then
            SLEEP=300    # bootstrap: 5 min
        else
            SLEEP=1800   # steady-state: 30 min
        fi
    fi

    echo "[research-${LABEL}] $(date '+%H:%M:%S') model=${MODEL} sleep=${SLEEP}s ft=${FT_COUNT:-?}"

    # Run research session
    cat prompts/research_loop.md | claude --model "$MODEL" 2>&1 | tee -a "$LOG_FILE"

    # Heartbeat — record iteration end
    bash scripts/heartbeat.sh research_loop completed || true

    sleep "$SLEEP"
done
