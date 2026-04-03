#!/usr/bin/env bash
# QuantStack — graceful shutdown via Docker Compose.
# Usage: ./stop.sh
#
# 1. Activates kill switch in both DB and sentinel file.
# 2. Runs `docker compose down` (SIGTERM + grace period).
# 3. Crew runners flush state and exit cleanly.
#
# Idempotent: safe to run when the system is already stopped.
# To restart:  ./start.sh
# To reset kill switch:
#   python3 -c "from quantstack.execution.kill_switch import get_kill_switch; get_kill_switch().reset('manual')"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  QuantStack — Graceful Shutdown                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Activate kill switch — both layers
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
    print(f"[stop.sh]   WARNING: DB write failed ({e}) — sentinel file is fallback", file=sys.stderr)

# Layer B: Sentinel file
sentinel_path = Path(os.environ.get("KILL_SWITCH_SENTINEL", "~/.quantstack/KILL_SWITCH_ACTIVE")).expanduser()
sentinel_path.parent.mkdir(parents=True, exist_ok=True)
with open(sentinel_path, "w") as f:
    f.write(f"triggered_at={now}\nreason=stop.sh graceful shutdown\n")
print(f"[stop.sh]   Sentinel file written: {sentinel_path}")
PYEOF

# ---------------------------------------------------------------------------
# Step 2: Docker Compose down (SIGTERM → grace period → SIGKILL)
# ---------------------------------------------------------------------------
echo "[stop.sh] Stopping containers (docker compose down)..."
docker compose down

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  QuantStack stopped. Kill switch remains active.        ║"
echo "║                                                         ║"
echo "║  Restart:       ./start.sh                              ║"
echo "║  Reset switch:  python3 -c \"from quantstack.execution.  ║"
echo "║    kill_switch import get_kill_switch;                   ║"
echo "║    get_kill_switch().reset('manual')\"                   ║"
echo "║  Status:        ./status.sh                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
