#!/usr/bin/env bash
# Record a loop heartbeat in PostgreSQL.
# Usage: bash scripts/heartbeat.sh <loop_name> <running|completed>
# Called by start.sh tmux wrappers before/after each Claude session.

set -euo pipefail

LOOP_NAME="${1:?Usage: heartbeat.sh <loop_name> <status>}"
STATUS="${2:?Usage: heartbeat.sh <loop_name> <status>}"

python3 -c "
import psycopg2, os, sys

conn = psycopg2.connect(os.environ['TRADER_PG_URL'])
conn.autocommit = True
cur = conn.cursor()

loop_name = sys.argv[1]
status = sys.argv[2]

cur.execute('SELECT COALESCE(MAX(iteration), 0) FROM loop_heartbeats WHERE loop_name = %s', [loop_name])
it = cur.fetchone()[0]

if status == 'running':
    it += 1
    cur.execute(
        'INSERT INTO loop_heartbeats (loop_name, iteration, started_at, status, errors) VALUES (%s, %s, NOW(), %s, 0)',
        [loop_name, it, 'running']
    )
else:
    cur.execute(
        'UPDATE loop_heartbeats SET finished_at = NOW(), status = %s WHERE loop_name = %s AND iteration = %s',
        [status, loop_name, it]
    )

conn.close()
print(f'[HEARTBEAT] {loop_name} iter={it} {status}')
" "$LOOP_NAME" "$STATUS" || echo "[HEARTBEAT] WARNING: failed to record heartbeat for $LOOP_NAME"
