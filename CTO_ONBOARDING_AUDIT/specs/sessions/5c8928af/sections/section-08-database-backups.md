# Section 8: Automated Database Backups

## Problem Statement

QuantStack stores all system state in PostgreSQL across 60+ tables -- positions, orders, strategies, signals, checkpoints, event bus cursors, dead letter queue entries, and more. The database runs inside a Docker container with volumes on local disk. There is no backup mechanism. A disk failure, accidental `DROP TABLE`, or corrupted write means total data loss with no recovery path.

This section implements daily full backups via `pg_dump`, continuous WAL archiving for point-in-time recovery between daily dumps, backup verification, retention policies, concurrent-run protection, supervisor health monitoring for stale backups, and a restore runbook.

## Dependencies

- **None.** This section is parallelizable with all other sections in Batch 1.
- The backup script connects directly to PostgreSQL via `pg_dump` / `pg_restore` CLI tools -- it does not depend on the psycopg3 migration (Section 1) or any application-level DB code.
- Integration with the scheduler (Section 9) is optional -- the backup can run via cron or as a scheduler job. If Section 9 is completed first, add the backup as a scheduler job. Otherwise, use host-level cron or an in-container cron entry.

## Tests (Write These First)

All tests live in `tests/integration/test_database_backups.py` unless otherwise noted.

### Backup script tests

```python
# Test: pg_dump creates valid backup file
# - Run backup.sh against a test database with known tables and row counts
# - Assert the output file exists at the expected path
# - Assert the file is non-empty and in pg_dump custom format

# Test: pg_restore --list succeeds on backup file (integrity check)
# - Run backup.sh to produce a dump file
# - Run pg_restore --list on the dump file
# - Assert exit code 0
# - Assert the TOC listing contains expected table names

# Test: backup script exits non-zero on pg_dump failure
# - Run backup.sh with an invalid PGHOST or PGDATABASE
# - Assert the script exits with non-zero status
# - Assert no partial/corrupt dump file is left behind

# Test: backups older than 30 days are deleted
# - Create fake backup files with timestamps older than 30 days in the backup directory
# - Run backup.sh (or the pruning portion)
# - Assert old files are deleted
# - Assert files newer than 30 days are preserved

# Test: WAL archive files older than 7 days are pruned
# - Create fake WAL files with mtime older than 7 days in the WAL archive directory
# - Run the WAL pruning step
# - Assert old WAL files are deleted
# - Assert recent WAL files are preserved

# Test: backup script uses flock to prevent concurrent runs
# - Acquire an flock on the same lock file the script uses
# - Run backup.sh in the background
# - Assert it exits immediately (or within a short timeout) without producing a dump
# - Release the lock, run again, assert it succeeds
```

### Restore procedure tests

```python
# Test: full restore from pg_dump -- all tables intact with correct row counts
# - Seed a test database with known data across multiple tables
# - Run backup.sh to produce a dump
# - Drop the test database (or create a fresh one)
# - Run pg_restore into the fresh database
# - Assert all tables exist with correct row counts

# Test: PITR restore from WAL -- data consistent to target timestamp
# - Enable WAL archiving on a test database
# - Insert data at time T1
# - Record timestamp T2
# - Insert more data at time T3
# - Restore base backup, replay WAL to T2
# - Assert data from T1 is present, data from T3 is absent
```

### Monitoring tests

```python
# Test: supervisor health check detects backup older than 36 hours
# - Set the most recent backup file's mtime to 37 hours ago
# - Run the backup staleness health check
# - Assert it returns a warning/alert status

# Test: supervisor raises warning event for stale backup
# - Trigger the staleness check with a stale backup
# - Assert a warning event is published (or logged, depending on supervisor integration)
# - Run with a fresh backup (< 36 hours old), assert no warning
```

## Implementation Details

### 1. Backup Script: `scripts/backup.sh`

Create a new shell script at `scripts/backup.sh`. The script must be executable (`chmod +x`).

**Behavior:**

- Uses `flock` on a lock file (`/tmp/quantstack_backup.lock`) to prevent concurrent runs. If the lock cannot be acquired, the script exits immediately with a message -- it does not queue or wait.
- Runs `pg_dump --format=custom` against the `quantstack` database. Custom format supports selective table restore, compression, and integrity verification via `pg_restore --list`.
- Output path: `/data/quantstack/backups/quantstack_YYYY-MM-DD.dump` (date from the script's execution time).
- After a successful dump, runs `pg_restore --list <dump_file>` to verify the dump is readable. If verification fails, the script exits non-zero and does NOT delete any old backups (the failed dump should remain for investigation).
- On successful dump + verification, prunes backups older than 30 days from the backup directory.
- Prunes WAL archive files older than 7 days from `/data/quantstack/wal_archive/`.
- Exits non-zero on any failure (pg_dump failure, verification failure, missing directories). Uses `set -euo pipefail`.

**Key script structure (signatures/pseudocode, not full implementation):**

```bash
#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="/data/quantstack/backups"
WAL_ARCHIVE_DIR="/data/quantstack/wal_archive"
LOCK_FILE="/tmp/quantstack_backup.lock"
RETENTION_DAYS=30
WAL_RETENTION_DAYS=7
DB_NAME="${QUANTSTACK_DB_NAME:-quantstack}"

# Acquire exclusive lock (non-blocking)
exec 200>"$LOCK_FILE"
flock -n 200 || { echo "Another backup is already running"; exit 1; }

# Ensure directories exist
mkdir -p "$BACKUP_DIR" "$WAL_ARCHIVE_DIR"

# Generate dump filename
DUMP_FILE="$BACKUP_DIR/${DB_NAME}_$(date +%Y-%m-%d).dump"

# Run pg_dump (custom format for compression + selective restore)
pg_dump --format=custom --dbname="$DB_NAME" --file="$DUMP_FILE"

# Verify dump integrity
pg_restore --list "$DUMP_FILE" > /dev/null

# Prune old backups (only after successful dump + verify)
find "$BACKUP_DIR" -name "*.dump" -mtime +$RETENTION_DAYS -delete

# Prune old WAL archives
find "$WAL_ARCHIVE_DIR" -type f -mtime +$WAL_RETENTION_DAYS -delete

echo "Backup completed successfully: $DUMP_FILE"
```

### 2. WAL Archiving Configuration

PostgreSQL must be configured for continuous WAL archiving. This requires changes to `postgresql.conf` (or equivalent Docker entrypoint configuration):

- `wal_level = replica` -- enables WAL sufficient for archiving and replication.
- `archive_mode = on` -- turns on WAL archiving.
- `archive_command = 'cp %p /data/quantstack/wal_archive/%f'` -- copies each completed WAL segment to the archive directory.
- `archive_timeout = 300` -- forces a WAL segment switch every 5 minutes even during low activity, ensuring recent transactions are archived promptly.

**Where to configure:** If PostgreSQL runs via the `postgres` service in `docker-compose.yml`, add these settings via:
- A custom `postgresql.conf` mounted into the container, OR
- Command-line flags in the docker-compose `command:` directive, OR
- An init script in `/docker-entrypoint-initdb.d/`

The WAL archive directory (`/data/quantstack/wal_archive/`) must be mounted as a Docker volume so archives persist across container restarts.

### 3. Integration with Scheduler

The backup job should run daily at 02:00 UTC. Two integration paths:

**Option A (preferred): Add to scheduler.** If Section 9 (containerize scheduler) is complete, add the backup as an APScheduler job in `scripts/scheduler.py`:

```python
# In the scheduler's job registration, add:
# - Job name: "database_backup"
# - Trigger: CronTrigger(hour=2, minute=0, timezone="UTC")
# - Function: calls subprocess.run(["bash", "scripts/backup.sh"], check=True)
# - Misfire grace time: 3600 (1 hour -- if the scheduler was down at 02:00, run when it starts)
```

**Option B (fallback): Host cron or in-container cron.** Add a cron entry that runs the backup script inside the PostgreSQL container (or from the host targeting the container's database):

```
0 2 * * * docker exec quantstack-postgres /app/scripts/backup.sh >> /var/log/quantstack_backup.log 2>&1
```

### 4. Supervisor Health Check for Stale Backups

Add a backup staleness check to the supervisor graph's health monitoring. This check runs as part of the supervisor's periodic health sweep.

**Logic:**

- Scan the backup directory for the most recent `.dump` file.
- Compare its modification time to the current time.
- If the most recent backup is older than 36 hours (allows for a missed run plus buffer), raise a warning.
- The 36-hour threshold (not 24) accounts for slight scheduling drift and avoids false alarms on weekends or during brief outages.

**Integration point:** The supervisor graph already runs health checks. Add this as a new check function that the supervisor calls. If the check fails, the supervisor should publish a warning event (or log at WARNING level) -- it should NOT halt trading. A stale backup is a maintenance concern, not a safety emergency.

**File to modify:** The supervisor health check module -- likely in `src/quantstack/health/` or within `src/quantstack/graphs/supervisor/nodes.py` where other health checks are defined.

### 5. Restore Runbook

Add a restore section to `docs/ops-runbook.md` (the existing operations runbook referenced in `CLAUDE.md`).

**Full restore from pg_dump:**

```bash
# Stop all services first
./stop.sh

# Restore from the most recent dump
pg_restore --dbname=quantstack --clean --if-exists \
    /data/quantstack/backups/quantstack_YYYY-MM-DD.dump

# Verify key table row counts
psql quantstack -c "SELECT 'positions' AS tbl, COUNT(*) FROM positions
    UNION ALL SELECT 'orders', COUNT(*) FROM orders
    UNION ALL SELECT 'strategies', COUNT(*) FROM strategies;"

# Restart services
./start.sh
```

**Point-in-time recovery (PITR) from WAL:**

```bash
# Stop PostgreSQL
docker-compose stop postgres

# Restore base backup
pg_restore --dbname=quantstack --clean --if-exists \
    /data/quantstack/backups/quantstack_YYYY-MM-DD.dump

# Create recovery.signal file (PostgreSQL 12+)
touch /var/lib/postgresql/data/recovery.signal

# Set recovery target in postgresql.conf (or postgresql.auto.conf)
# restore_command = 'cp /data/quantstack/wal_archive/%f %p'
# recovery_target_time = '2026-04-05 14:30:00 UTC'
# recovery_target_action = 'promote'

# Start PostgreSQL -- it replays WAL to the target timestamp, then promotes
docker-compose start postgres

# Verify data consistency at the target time
# Restart application services
./start.sh
```

**Verification after any restore:**
- Check that all 60+ tables exist: `SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';`
- Spot-check row counts for critical tables: `positions`, `orders`, `strategies`, `signals`, `bracket_legs`, `loop_cursors`.
- Run one trading cycle in paper mode before resuming normal operation.

### 6. Directory Structure and Volume Mounts

Ensure these directories exist and are mounted as Docker volumes in `docker-compose.yml`:

- `/data/quantstack/backups/` -- daily pg_dump output files
- `/data/quantstack/wal_archive/` -- continuous WAL archive segments

Add to `docker-compose.yml` under the `postgres` service's `volumes:` section:

```yaml
volumes:
  - quantstack_backups:/data/quantstack/backups
  - quantstack_wal_archive:/data/quantstack/wal_archive
```

And declare the named volumes at the top level of docker-compose.yml.

## Files to Create or Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/backup.sh` | Create | Daily backup script with pg_dump, verification, retention, flock |
| `docker-compose.yml` | Modify | Add backup/WAL volume mounts to postgres service; add WAL config |
| `scripts/scheduler.py` | Modify | Add daily backup job at 02:00 UTC (if Section 9 is done) |
| Supervisor health check module | Modify | Add backup staleness check (>36 hours triggers warning) |
| `docs/ops-runbook.md` | Modify | Add restore procedures (full restore + PITR) |
| PostgreSQL config | Modify | Enable WAL archiving (wal_level, archive_mode, archive_command) |

## Failure Modes

| Failure | Impact | Mitigation |
|---------|--------|------------|
| pg_dump fails (disk full, PG down) | No backup produced | Script exits non-zero; supervisor detects stale backup within 36 hours |
| Backup file corrupted | Restore would fail | `pg_restore --list` verification catches this immediately after dump |
| WAL archive directory fills disk | PostgreSQL stops accepting writes | 7-day WAL pruning in backup script; monitor disk usage |
| Concurrent backup runs | Corrupt or duplicate dumps | `flock` prevents concurrent execution |
| Scheduler misses 02:00 window | Backup delayed | APScheduler misfire grace time (1 hour) runs it late; 36-hour alert threshold provides buffer |
| Restore to wrong timestamp (PITR) | Data loss or inconsistency | Runbook includes verification steps; always stop services before restore |
