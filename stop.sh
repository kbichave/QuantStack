#!/usr/bin/env bash
# QuantStack — graceful shutdown.
# Usage: ./stop.sh
#
# 1. Activates kill switch in both DB and sentinel file (loops stop cleanly on
#    their next iteration — no open orders abandoned mid-flight).
# 2. Waits up to 30s for in-flight iterations to finish.
# 3. Kills the tmux session.
#
# Idempotent: safe to run when the system is already stopped.
# To restart:  ./start.sh
# To reset kill switch after restart:
#   python3 -c "from quantstack.execution.kill_switch import get_kill_switch; get_kill_switch().reset('manual')"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  QuantStack — Graceful Shutdown                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Activate kill switch — both layers
#   Layer A: DB system_state key='kill_switch'
#   Layer B: Sentinel file (KillSwitch.SENTINEL_FILE = ~/.quantstack/KILL_SWITCH_ACTIVE)
#            Format must match KillSwitch._write_sentinel() exactly.
# ---------------------------------------------------------------------------
echo "[stop.sh] Activating kill switch..."

python3 - <<'PYEOF'
import os, sys
from datetime import datetime
from pathlib import Path

now = datetime.now()

# Layer A: DB
try:
    import psycopg2
    pg_url = os.environ.get("TRADER_PG_URL")
    if not pg_url:
        raise RuntimeError("TRADER_PG_URL not set")
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO system_state (key, value, updated_at)
        VALUES ('kill_switch', 'active', NOW())
        ON CONFLICT (key) DO UPDATE
            SET value = 'active', updated_at = NOW()
    """)
    conn.close()
    print("[stop.sh]   Kill switch written to DB")
except Exception as e:
    print(f"[stop.sh]   WARNING: DB write failed ({e}) — sentinel file will still stop loops on next restart", file=sys.stderr)

# Layer B: Sentinel file — format matches KillSwitch._write_sentinel() exactly
sentinel_path = Path(os.environ.get("KILL_SWITCH_SENTINEL", "~/.quantstack/KILL_SWITCH_ACTIVE")).expanduser()
sentinel_path.parent.mkdir(parents=True, exist_ok=True)
with open(sentinel_path, "w") as f:
    f.write(f"triggered_at={now}\nreason=stop.sh graceful shutdown\n")
print(f"[stop.sh]   Sentinel file written: {sentinel_path}")
PYEOF

# ---------------------------------------------------------------------------
# Step 2: If no tmux session, we're done
# ---------------------------------------------------------------------------
if ! tmux has-session -t quantstack-loops 2>/dev/null; then
    echo "[stop.sh] No tmux session running — system was already stopped."
    echo "[stop.sh] Kill switch is active. Done."
    echo ""
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3: Wait up to 30s for in-flight iterations to finish
#   A loop writes status='running' at the start of an iteration and
#   status='completed' at the end. Once no 'running' row exists from
#   the last 60s, no iteration is in flight.
# ---------------------------------------------------------------------------
echo "[stop.sh] Waiting up to 30s for in-flight iterations to complete..."

DEADLINE=$((SECONDS + 30))
LOOPS_IDLE=false

while [[ $SECONDS -lt $DEADLINE ]]; do
    RUNNING=$(python3 - <<'PYEOF' 2>/dev/null || echo "0"
import os, psycopg2
pg_url = os.environ.get("TRADER_PG_URL", "")
if not pg_url:
    print("0"); exit()
conn = psycopg2.connect(pg_url)
cur = conn.cursor()
cur.execute("""
    SELECT COUNT(*) FROM loop_heartbeats
    WHERE status = 'running'
      AND started_at > NOW() - INTERVAL '60 seconds'
""")
print(cur.fetchone()[0])
conn.close()
PYEOF
)

    if [[ "$RUNNING" == "0" ]]; then
        LOOPS_IDLE=true
        echo "[stop.sh]   No active iterations — loops are idle."
        break
    fi

    echo "[stop.sh]   ${RUNNING} iteration(s) still running. Waiting 3s..."
    sleep 3
done

if ! $LOOPS_IDLE; then
    echo "[stop.sh]   WARNING: 30s timeout reached — loops may be mid-iteration. Proceeding anyway."
fi

# ---------------------------------------------------------------------------
# Step 4: Kill the tmux session
# ---------------------------------------------------------------------------
if tmux has-session -t quantstack-loops 2>/dev/null; then
    tmux kill-session -t quantstack-loops
    echo "[stop.sh] tmux session 'quantstack-loops' killed."
else
    echo "[stop.sh] tmux session already gone."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  QuantStack stopped. Kill switch remains active.          ║"
echo "║                                                          ║"
echo "║  Restart:       ./start.sh                               ║"
echo "║  Reset switch:  python3 -c \"from quantstack.execution.   ║"
echo "║    kill_switch import get_kill_switch;                   ║"
echo "║    get_kill_switch().reset('manual')\"                    ║"
echo "║  Status:        ./status.sh                              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
