#!/usr/bin/env bash
# QuantStack — single entry point for the autonomous trading system.
# Usage: ./start.sh
# Starts 4 tmux windows: trading, research, supervisor, scheduler.
# Leave running for up to a month. See ./report.sh for performance summary.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Load .env
# ---------------------------------------------------------------------------
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.example and fill in credentials." >&2
    exit 1
fi
set -a; source .env; set +a

# ---------------------------------------------------------------------------
# 2. Check quantstack importable
# ---------------------------------------------------------------------------
if ! python3 -c "import quantstack" 2>/dev/null; then
    echo "ERROR: quantstack is not importable." >&2
    echo "  Fix: pip install -e ." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Check PostgreSQL reachable
# ---------------------------------------------------------------------------
if [[ -z "${TRADER_PG_URL:-}" ]]; then
    echo "ERROR: TRADER_PG_URL is not set in .env" >&2
    exit 1
fi
if ! python3 -c "
import psycopg2, os
conn = psycopg2.connect(os.environ['TRADER_PG_URL'])
conn.close()
" 2>/dev/null; then
    echo "ERROR: Cannot connect to PostgreSQL at TRADER_PG_URL=$TRADER_PG_URL" >&2
    exit 1
fi
echo "[start.sh] PostgreSQL OK"

# ---------------------------------------------------------------------------
# 4. Check tmux installed
# ---------------------------------------------------------------------------
if ! command -v tmux &>/dev/null; then
    echo "ERROR: tmux is not installed." >&2
    echo "  Fix: brew install tmux" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Check claude CLI installed
# ---------------------------------------------------------------------------
if ! command -v claude &>/dev/null; then
    echo "ERROR: claude CLI is not installed." >&2
    echo "  Fix: npm install -g @anthropic-ai/claude-code" >&2
    exit 1
fi


# ---------------------------------------------------------------------------
# 6. Run migrations (idempotent)
# ---------------------------------------------------------------------------
echo "[start.sh] Running DB migrations..."
python3 -c "
from quantstack.db import run_migrations, open_db
conn = open_db()
run_migrations(conn)
conn.close()
print('Migrations complete')
"

# ---------------------------------------------------------------------------
# 7. First-run detection: bootstrap universe if empty
# ---------------------------------------------------------------------------
UNIVERSE_COUNT=$(python3 -c "
from quantstack.db import open_db
conn = open_db()
row = conn.execute('SELECT COUNT(*) FROM universe WHERE is_active = TRUE').fetchone()
conn.close()
print(row[0])
" 2>/dev/null || echo "0")

if [[ "$UNIVERSE_COUNT" -eq 0 ]]; then
    echo "[start.sh] Empty universe — running quantstack-bootstrap..."
    quantstack-bootstrap
    echo "[start.sh] Bootstrap complete"
fi

# ---------------------------------------------------------------------------
# 8. Preflight check
# ---------------------------------------------------------------------------
echo "[start.sh] Running preflight checks..."
python3 -c "
from quantstack.coordination.preflight import PreflightCheck
from quantstack.db import open_db
conn = open_db()
report = PreflightCheck(conn, target_wallet=65000).run()
print(report.summary())
if not report.ready:
    import sys
    sys.exit(1)
conn.close()
" || { echo "ERROR: Preflight failed. Fix blockers above before starting." >&2; exit 1; }

# ---------------------------------------------------------------------------
# 8a. Data freshness check — run sync if OHLCV is stale (>1 trading day)
# ---------------------------------------------------------------------------
echo "[start.sh] Checking data freshness..."
STALE_DAYS=$(python3 -c "
from quantstack.db import open_db
from datetime import datetime, timedelta
conn = open_db()
row = conn.execute(
    \"SELECT MAX(timestamp) FROM ohlcv WHERE symbol = 'SPY' AND timeframe IN ('1D', '1d', 'daily')\"
).fetchone()
conn.close()
if row and row[0]:
    from datetime import timezone
    latest = row[0] if hasattr(row[0], 'date') else datetime.fromisoformat(str(row[0]))
    if latest.tzinfo:
        latest = latest.replace(tzinfo=None)
    age = (datetime.now() - latest).days
    print(age)
else:
    print(999)
" 2>/dev/null || echo "999")

if [[ "$STALE_DAYS" -gt 1 ]]; then
    echo "[start.sh] OHLCV data is ${STALE_DAYS} days stale — running incremental sync (background)..."
    nohup python3 scripts/acquire_historical_data.py --phases ohlcv_daily ohlcv_5min ohlcv_1h fundamentals news \
        >> data/logs/data_refresh.log 2>&1 &
    DATA_SYNC_PID=$!
    echo "[start.sh] Data sync started (PID ${DATA_SYNC_PID}). Trading loop will use available data meanwhile."
else
    echo "[start.sh] OHLCV data is fresh (${STALE_DAYS} days old)"
fi

# ---------------------------------------------------------------------------
# 8b. Credit regime display (informational — does NOT block startup)
# ---------------------------------------------------------------------------
echo "[start.sh] Checking credit regime..."
python3 -c "
import asyncio
from quantstack.mcp.tools.macro_signals import get_credit_market_signals
try:
    result = asyncio.run(get_credit_market_signals()) if asyncio.iscoroutinefunction(get_credit_market_signals) else get_credit_market_signals()
    regime = result.get('credit_regime', 'unknown') if isinstance(result, dict) else 'unknown'
    print(f'  Credit regime: {regime}')
    if regime == 'widening':
        print('  WARNING: Credit regime is WIDENING — long equity entries will be gated.')
        print('  System will trade when regime shifts. Options income strategies unaffected.')
    else:
        print('  Credit regime is stable — long entries are open.')
except Exception as e:
    print(f'  Could not check credit regime: {e} (non-fatal)')
" || true

# ---------------------------------------------------------------------------
# 8c. Memory compaction — trim oversized files before loops start
# ---------------------------------------------------------------------------
echo "[start.sh] Compacting memory files..."
python3 - <<'COMPACT_EOF'
import sys
from pathlib import Path

LIMITS = {
    "workshop_lessons.md": 100,
    "ml_experiment_log.md": 120,
    "trade_journal.md": 150,
    "ml_research_program.md": 80,
}
memory_dir = Path(".claude/memory")

for filename, max_lines in LIMITS.items():
    filepath = memory_dir / filename
    if not filepath.exists():
        continue
    lines = filepath.read_text().splitlines()
    if len(lines) <= max_lines:
        print(f"  {filename}: {len(lines)} lines (OK)")
        continue
    archive_path = memory_dir / f"{filename}.archive.md"
    excess = lines[:-max_lines]
    keep   = lines[-max_lines:]
    with open(archive_path, "a") as f:
        from datetime import datetime
        f.write(f"\n\n## Archived {datetime.now().isoformat()}\n\n")
        f.write("\n".join(excess))
    filepath.write_text("\n".join(keep))
    print(f"  {filename}: compacted {len(lines)} → {max_lines} lines ({len(excess)} archived)")
COMPACT_EOF

# ---------------------------------------------------------------------------
# 8d. Initial community intelligence scan (background, non-blocking)
# Seeds research_queue before the first research loop iteration runs.
# ---------------------------------------------------------------------------
if [[ -f ".claude/agents/community-intel.md" ]]; then
    echo "[start.sh] Seeding research_queue with community intelligence (background)..."
    nohup bash -c 'cat .claude/agents/community-intel.md | claude --model haiku 2>&1 | tee -a data/logs/community_intel.log' &
    COMMUNITY_INTEL_PID=$!
    echo "[start.sh] Community intel scan started in background (PID ${COMMUNITY_INTEL_PID})"
else
    echo "[start.sh] community-intel.md not found — skipping initial scan"
fi

# ---------------------------------------------------------------------------
# 9. Create directories
# ---------------------------------------------------------------------------
mkdir -p data/logs reports

# ---------------------------------------------------------------------------
# 10. Register SIGTERM handler — writes kill_switch=active to DB
# ---------------------------------------------------------------------------
_sigterm_handler() {
    echo "[start.sh] SIGTERM received — activating kill switch..."
    python3 -c "
from quantstack.db import open_db
conn = open_db()
conn.execute(\"UPDATE system_state SET value='active', updated_at=NOW() WHERE key='kill_switch'\")
conn.close()
print('Kill switch activated')
" || true
    tmux kill-session -t quantstack-loops 2>/dev/null || true
    exit 0
}
trap _sigterm_handler SIGTERM SIGINT

# ---------------------------------------------------------------------------
# 11. Kill any existing quantstack-loops session (idempotent restart)
# ---------------------------------------------------------------------------
tmux kill-session -t quantstack-loops 2>/dev/null || true

# ---------------------------------------------------------------------------
# 12. Create tmux session with 4 windows
# ---------------------------------------------------------------------------
echo "[start.sh] Starting tmux session quantstack-loops..."

# Pre-create session so we can inject env vars before any windows run.
# tmux windows inherit the session environment, not the parent shell's env.
ENV_PREFIX="set -a; source $(pwd)/.env; set +a;"

# Trading window — sonnet, market-aware sleep.
# 60s poll during market hours (09:30–16:00 ET); 30 min outside to avoid wasteful idle sessions.
# Shell wrapper calls scripts/heartbeat.sh before/after each Claude session,
# guaranteeing heartbeats even if the model doesn't execute code blocks.
tmux new-session -d -s quantstack-loops -n trading \
    "$ENV_PREFIX while :; do
       export HEARTBEAT_ITERATION=\$(bash scripts/heartbeat.sh trading_loop running 2>/dev/null | grep '^HEARTBEAT_ITERATION=' | cut -d= -f2)
       cat prompts/trading_loop.md | claude --model sonnet 2>&1 | tee -a data/logs/trading_loop.log
       bash scripts/heartbeat.sh trading_loop completed
       HOUR=\$(TZ='America/New_York' date +%H)
       if [[ \"\$HOUR\" -ge 9 && \"\$HOUR\" -lt 16 ]]; then sleep 60; else sleep 1800; fi
     done"

# Research window — market-aware model routing + adaptive interval.
# Market hours (09:30–16:00 ET): haiku — task is short (data refresh, signal check, watchlist).
#   Context is lightweight; no deep work needed while market is open.
# After hours: sonnet — full research cycles (evidence gathering, strategy design, backtest).
#   Subagents spawned by research loop inherit this model via .claude/agents/ frontmatter.
# Sleep: 5 min during market hours, 30 min outside.
tmux new-window -t quantstack-loops -n research \
    "$ENV_PREFIX while :; do
       export HEARTBEAT_ITERATION=\$(bash scripts/heartbeat.sh research_loop running 2>/dev/null | grep '^HEARTBEAT_ITERATION=' | cut -d= -f2)
       HOUR=\$(TZ='America/New_York' date +%H)
       if [[ \"\$HOUR\" -ge 9 && \"\$HOUR\" -lt 16 ]]; then
         MODEL='haiku'
         SLEEP=300
       else
         MODEL='sonnet'
         SLEEP=1800
       fi
       cat prompts/research_loop.md | claude --model \"\$MODEL\" 2>&1 | tee -a data/logs/research_loop.log
       bash scripts/heartbeat.sh research_loop completed
       sleep \$SLEEP
     done"

# Add supervisor window
tmux new-window -t quantstack-loops -n supervisor \
    "$ENV_PREFIX python3 -m quantstack.coordination.supervisor_main 2>&1 | tee -a data/logs/supervisor.log"

# Add scheduler window
tmux new-window -t quantstack-loops -n scheduler \
    "$ENV_PREFIX python scripts/scheduler.py 2>&1 | tee -a data/logs/scheduler.log"

# Add community-intel window — runs weekly on Sunday 19:00 ET, idles otherwise.
# Can be manually re-triggered by killing and restarting within this window.
tmux new-window -t quantstack-loops -n community-intel \
    "$ENV_PREFIX while :; do
       day=\$(date +%u)
       hour=\$(date +%H)
       if [[ \"\$day\" == 7 && \"\$hour\" == 19 ]]; then
         echo '[community-intel] Weekly scan starting...'
         cat .claude/agents/community-intel.md | claude --model haiku 2>&1 | tee -a data/logs/community_intel.log
         echo '[community-intel] Scan complete. Sleeping until next hourly check.'
       fi
       sleep 3600
     done"

# ---------------------------------------------------------------------------
# 13. Done
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  QuantStack is running.                                    ║"
echo "║                                                          ║"
echo "║  Attach:   tmux attach -t quantstack-loops                 ║"
echo "║  Windows:  trading | research | supervisor | scheduler | community-intel  ║"
echo "║  Stop:     tmux kill-session -t quantstack-loops           ║"
echo "║  Report:   ./report.sh                                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
