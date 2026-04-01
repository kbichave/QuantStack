#!/usr/bin/env bash
# Record a loop heartbeat in PostgreSQL.
# Usage: bash scripts/heartbeat.sh <loop_name> <running|completed>
# Called by start.sh tmux wrappers before/after each Claude session.
#
# The 'running' call prints "HEARTBEAT_ITERATION=N" to stdout so the caller
# can export it and pass it to the subsequent 'completed' call, avoiding a
# second MAX() query that could target the wrong row.

set -euo pipefail

LOOP_NAME="${1:?Usage: heartbeat.sh <loop_name> <status>}"
STATUS="${2:?Usage: heartbeat.sh <loop_name> <status>}"

python3 -c "
import psycopg2, os, sys

conn = psycopg2.connect(os.environ['TRADER_PG_URL'])
conn.autocommit = True
cur = conn.cursor()

loop_name = sys.argv[1]
status    = sys.argv[2]

# Mark stale orphans before reading MAX — any row still 'running' after
# 30 minutes is a dead session (killed mid-iteration). Marking it 'orphaned'
# removes it from the MAX() count and keeps the table honest.
cur.execute('''
    UPDATE loop_heartbeats
       SET status = 'orphaned'
     WHERE loop_name = %s
       AND status = 'running'
       AND finished_at IS NULL
       AND started_at < NOW() - INTERVAL '30 minutes'
''', [loop_name])

if status == 'running':
    # Count only rows that have a finished_at (definitively closed).
    # This makes the counter immune to stale 'running' rows.
    cur.execute('''
        SELECT COALESCE(MAX(iteration), 0)
          FROM loop_heartbeats
         WHERE loop_name = %s
           AND finished_at IS NOT NULL
    ''', [loop_name])
    it = cur.fetchone()[0] + 1

    cur.execute(
        'INSERT INTO loop_heartbeats (loop_name, iteration, started_at, status, errors) '
        'VALUES (%s, %s, NOW(), %s, 0) '
        'ON CONFLICT (loop_name, iteration) DO UPDATE SET started_at = NOW(), status = %s',
        [loop_name, it, 'running', 'running']
    )
    # Print iteration so the shell wrapper can export it for the 'completed' call.
    print(f'HEARTBEAT_ITERATION={it}')

else:
    # Use iteration number passed via env var (set by the 'running' call in
    # this same shell loop iteration). Fall back to MAX(running) if absent.
    it_env = os.environ.get('HEARTBEAT_ITERATION')
    if it_env is not None:
        it = int(it_env)
    else:
        cur.execute('''
            SELECT COALESCE(MAX(iteration), 0)
              FROM loop_heartbeats
             WHERE loop_name = %s
               AND status = 'running'
        ''', [loop_name])
        it = cur.fetchone()[0]

    cur.execute(
        'UPDATE loop_heartbeats SET finished_at = NOW(), status = %s '
        'WHERE loop_name = %s AND iteration = %s',
        [status, loop_name, it]
    )

conn.close()
print(f'[HEARTBEAT] {loop_name} iter={it} {status}', file=sys.stderr)
" "$LOOP_NAME" "$STATUS" || echo "[HEARTBEAT] WARNING: failed to record heartbeat for $LOOP_NAME"
