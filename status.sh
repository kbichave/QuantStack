#!/usr/bin/env bash
# QuantStack — health dashboard (Docker Compose).
# Usage:
#   ./status.sh                         # print once and exit
#   ./status.sh --watch                 # live refresh every 10s
#   ./status.sh --watch --interval 30   # custom refresh interval

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

# Rewrite Docker-internal PG URL for host-side access
# Inside Docker, services talk to "postgres" hostname; from the host we use localhost.
if [[ "${TRADER_PG_URL:-}" == *"@postgres:"* ]]; then
    export TRADER_PG_URL="${TRADER_PG_URL//@postgres:/@localhost:}"
fi

WATCH=false
INTERVAL=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch) WATCH=true; shift ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

print_status() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  QuantStack Status Dashboard                            ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # --- Container Health ---
    echo "=== Container Health ==="
    docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || echo "  Docker Compose not running"
    echo ""

    # --- Graph Heartbeats ---
    echo "=== Graph Heartbeats ==="
    # Heartbeat files live at /tmp/{graph}-heartbeat INSIDE the Docker containers.
    # We docker-exec to read them, falling back to DB query if that fails.
    CONTAINER_MAP="trading:quantstack-trading-graph research:quantstack-research-graph supervisor:quantstack-supervisor-graph"
    THRESHOLDS="trading:120 research:600 supervisor:360"

    for entry in $CONTAINER_MAP; do
        graph="${entry%%:*}"
        container="${entry#*:}"
        threshold=$(echo "$THRESHOLDS" | tr ' ' '\n' | grep "^${graph}:" | cut -d: -f2)

        # Try reading heartbeat file from inside the container
        ts=$(docker exec "$container" cat "/tmp/${graph}-heartbeat" 2>/dev/null || true)
        if [[ -n "$ts" ]]; then
            now=$(date +%s)
            age=$(( now - ${ts%%.*} ))
            if [[ $age -lt $threshold ]]; then
                status="OK"
            else
                status="STALE"
            fi
            printf "  %-12s  age=%ds  threshold=%ds  [%s]\n" "$graph" "$age" "$threshold" "$status"
        else
            # DB fallback
            python3 -c "
import os
try:
    import psycopg2
    pg_url = os.environ.get('TRADER_PG_URL', '')
    if pg_url:
        conn = psycopg2.connect(pg_url)
        cur = conn.cursor()
        cur.execute(
            'SELECT MAX(cycle_timestamp) FROM crew_checkpoints WHERE crew_name = %s',
            ('${graph}',)
        )
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            print(f'  ${graph:12s}  last={row[0]}  (from DB)')
        else:
            print('  ${graph}          NO HEARTBEAT')
    else:
        print('  ${graph}          NO HEARTBEAT')
except:
    print('  ${graph}          NO HEARTBEAT')
" 2>/dev/null || printf "  %-12s  NO HEARTBEAT\n" "$graph"
        fi
    done
    echo ""

    # --- Active Positions ---
    echo "=== Active Positions ==="
    python3 -c "
import os
try:
    import psycopg2
    pg_url = os.environ.get('TRADER_PG_URL', '')
    conn = psycopg2.connect(pg_url)
    cur = conn.cursor()
    cur.execute(\"SELECT COUNT(*), COALESCE(SUM(unrealized_pnl), 0) FROM positions WHERE status = 'open'\")
    count, pnl = cur.fetchone()
    print(f'  Open positions: {count}  |  Unrealized P&L: \${pnl:,.2f}')
    conn.close()
except Exception as e:
    print(f'  Could not query positions: {e}')
" 2>/dev/null || echo "  DB not available"
    echo ""

    # --- Current Regime ---
    echo "=== Market Regime ==="
    python3 -c "
import os
try:
    import psycopg2
    pg_url = os.environ.get('TRADER_PG_URL', '')
    conn = psycopg2.connect(pg_url)
    cur = conn.cursor()
    cur.execute(\"SELECT value FROM loop_iteration_context WHERE key = 'current_regime' ORDER BY updated_at DESC LIMIT 1\")
    row = cur.fetchone()
    print(f'  Regime: {row[0]}' if row else '  Regime: unknown')
    conn.close()
except Exception as e:
    print(f'  Could not query regime: {e}')
" 2>/dev/null || echo "  DB not available"
    echo ""

    # --- Services ---
    echo "=== Service URLs ==="
    echo "  Langfuse:  http://localhost:3000"
    echo "  Ollama:    http://localhost:11434"
    echo ""
}

if $WATCH; then
    while true; do
        clear
        print_status
        echo "Refreshing every ${INTERVAL}s... (Ctrl+C to quit)"
        sleep "$INTERVAL"
    done
else
    print_status
fi
