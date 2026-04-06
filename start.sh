#!/usr/bin/env bash
# QuantStack — Docker Compose launcher for the autonomous trading system.
# Usage: ./start.sh [--build]
# Starts infrastructure (postgres, ollama, langfuse) then graph services.
# Pass --build to rebuild images before starting.
# See ./status.sh for health dashboard, ./stop.sh for graceful shutdown.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse CLI args
BUILD_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --build) BUILD_FLAG="--build" ;;
        *) echo "Unknown argument: $arg"; echo "Usage: ./start.sh [--build]"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Load .env
# ---------------------------------------------------------------------------
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.example and fill in credentials." >&2
    exit 1
fi
set -a; source .env; set +a

# ---------------------------------------------------------------------------
# 2. Check prerequisites
# ---------------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker is not installed." >&2
    echo "  Fix: https://docs.docker.com/get-docker/" >&2
    exit 1
fi

if ! docker compose version &>/dev/null; then
    echo "ERROR: docker compose (V2 plugin) is not available." >&2
    echo "  Fix: https://docs.docker.com/compose/install/" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Validate required env vars
# ---------------------------------------------------------------------------
MISSING=""
for var in TRADER_PG_URL ALPACA_API_KEY ALPACA_SECRET_KEY ALPHA_VANTAGE_API_KEY; do
    if [[ -z "${!var:-}" ]]; then
        MISSING="${MISSING}  - ${var}\n"
    fi
done
if [[ -n "$MISSING" ]]; then
    echo "ERROR: Missing required environment variables:" >&2
    echo -e "$MISSING" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Start infrastructure services
# ---------------------------------------------------------------------------
echo "[start.sh] Starting infrastructure services..."
docker compose up -d $BUILD_FLAG postgres langfuse-db ollama langfuse

# ---------------------------------------------------------------------------
# 5. Wait for infrastructure health checks
# ---------------------------------------------------------------------------
echo "[start.sh] Waiting for infrastructure to become healthy..."
INFRA_SERVICES="postgres ollama langfuse"
DEADLINE=$((SECONDS + 120))

for svc in $INFRA_SERVICES; do
    while [[ $SECONDS -lt $DEADLINE ]]; do
        HEALTH=$(docker compose ps --format json "$svc" 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if isinstance(data, list): data = data[0]
    print(data.get('Health', data.get('health', 'unknown')))
except: print('unknown')
" 2>/dev/null || echo "unknown")
        if [[ "$HEALTH" == *"healthy"* ]]; then
            echo "  $svc: healthy"
            break
        fi
        printf "."
        sleep 3
    done
    if [[ $SECONDS -ge $DEADLINE ]]; then
        echo ""
        echo "ERROR: Timed out waiting for $svc to become healthy." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 6. Pull Ollama models (idempotent)
# ---------------------------------------------------------------------------
echo "[start.sh] Pulling Ollama models..."
docker compose exec ollama ollama pull mxbai-embed-large || true
docker compose exec ollama ollama pull llama3.2 || true

# ---------------------------------------------------------------------------
# 7. Run DB migrations (including pgvector extension)
# ---------------------------------------------------------------------------
echo "[start.sh] Running DB migrations..."
docker compose run --rm trading-graph python -c "
from quantstack.db import run_migrations, open_db
conn = open_db()
run_migrations(conn)
conn.close()
print('Migrations complete')
"

# Enable pgvector extension
docker compose exec postgres psql -U quantstack -d quantstack -c "CREATE EXTENSION IF NOT EXISTS vector;" || true

# ---------------------------------------------------------------------------
# 8. Bootstrap universe if empty
# ---------------------------------------------------------------------------
UNIVERSE_COUNT=$(docker compose run --rm trading-graph python -c "
from quantstack.db import open_db
conn = open_db()
row = conn.execute('SELECT COUNT(*) FROM universe WHERE is_active = TRUE').fetchone()
conn.close()
print(row[0])
" 2>/dev/null | tail -1 || echo "0")

if [[ "$UNIVERSE_COUNT" -eq 0 ]]; then
    echo "[start.sh] Empty universe — running bootstrap..."
    docker compose run --rm trading-graph quantstack-bootstrap
    echo "[start.sh] Bootstrap complete"
fi

# ---------------------------------------------------------------------------
# 9. Preflight checks
# ---------------------------------------------------------------------------
echo "[start.sh] Running preflight checks..."
docker compose run --rm trading-graph python -c "
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
# 10. Data freshness check
# ---------------------------------------------------------------------------
echo "[start.sh] Checking data freshness..."
STALE_DAYS=$(docker compose run --rm trading-graph python -c "
from quantstack.db import open_db
from datetime import datetime
conn = open_db()
row = conn.execute(
    \"SELECT MAX(timestamp) FROM ohlcv WHERE symbol = 'SPY' AND timeframe IN ('1D', '1d', 'daily')\"
).fetchone()
conn.close()
if row and row[0]:
    latest = row[0] if hasattr(row[0], 'date') else datetime.fromisoformat(str(row[0]))
    if latest.tzinfo:
        latest = latest.replace(tzinfo=None)
    age = (datetime.now() - latest).days
    print(age)
else:
    print(999)
" 2>/dev/null | tail -1 || echo "999")

if [[ "$STALE_DAYS" -gt 1 ]]; then
    echo "[start.sh] OHLCV data is ${STALE_DAYS} days stale — running sync (background)..."
    docker compose run -d --rm trading-graph python scripts/acquire_historical_data.py \
        --phases ohlcv_daily ohlcv_5min ohlcv_1h fundamentals news
else
    echo "[start.sh] OHLCV data is fresh (${STALE_DAYS} days old)"
fi

# ---------------------------------------------------------------------------
# 11. RAG ingestion (first-run only, pgvector)
# ---------------------------------------------------------------------------
echo "[start.sh] Checking RAG embeddings..."
RAG_EMPTY=$(docker compose run --rm trading-graph python -c "
import os, psycopg2
try:
    conn = psycopg2.connect(os.environ.get('TRADER_PG_URL', 'postgresql://quantstack:quantstack@postgres:5432/quantstack'))
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM embeddings')
    count = cur.fetchone()[0]
    conn.close()
    print('yes' if count == 0 else 'no')
except:
    print('yes')
" 2>/dev/null | tail -1 || echo "yes")

if [[ "$RAG_EMPTY" == "yes" ]]; then
    echo "[start.sh] Embeddings table empty — running memory ingestion..."
    docker compose run --rm trading-graph python -m quantstack.rag.ingest || true
fi

# ---------------------------------------------------------------------------
# 12. Credit regime display (informational)
# ---------------------------------------------------------------------------
echo "[start.sh] Credit regime check skipped (MCP layer removed)"

# ---------------------------------------------------------------------------
# 13. Start graph services
# ---------------------------------------------------------------------------
echo "[start.sh] Starting graph + ML + dashboard services..."
docker compose up -d $BUILD_FLAG trading-graph research-graph supervisor-graph finrl-worker dashboard

# ---------------------------------------------------------------------------
# 14. Wait for graph health checks (best-effort, warning only)
# ---------------------------------------------------------------------------
echo "[start.sh] Waiting for graph containers to pass initial health check..."
GRAPH_DEADLINE=$((SECONDS + 60))
for svc in trading-graph research-graph supervisor-graph; do
    while [[ $SECONDS -lt $GRAPH_DEADLINE ]]; do
        HEALTH=$(docker compose ps --format json "$svc" 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if isinstance(data, list): data = data[0]
    print(data.get('Health', data.get('health', 'unknown')))
except: print('unknown')
" 2>/dev/null || echo "unknown")
        if [[ "$HEALTH" == *"healthy"* ]]; then
            echo "  $svc: healthy"
            break
        fi
        sleep 5
    done
done

# ---------------------------------------------------------------------------
# 15. Print status summary
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  QuantStack is running (Docker Compose).                ║"
echo "║                                                         ║"
echo "║  Dashboard: http://localhost:8421                       ║"
echo "║  Langfuse:  http://localhost:3100                       ║"
echo "║  Logs:      docker compose logs -f trading-graph        ║"
echo "║  Stop:      ./stop.sh                                   ║"
echo "║  Status:    ./status.sh                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
