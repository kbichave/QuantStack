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

echo "Starting backup: $DUMP_FILE"

# Run pg_dump (custom format for compression + selective restore)
pg_dump --format=custom --dbname="$DB_NAME" --file="$DUMP_FILE"

# Verify dump integrity
pg_restore --list "$DUMP_FILE" > /dev/null
echo "Backup verified successfully"

# Prune old backups (only after successful dump + verify)
find "$BACKUP_DIR" -name "*.dump" -mtime +$RETENTION_DAYS -delete

# Prune old WAL archives
if [ -d "$WAL_ARCHIVE_DIR" ]; then
    find "$WAL_ARCHIVE_DIR" -type f -mtime +$WAL_RETENTION_DAYS -delete
fi

echo "Backup completed successfully: $DUMP_FILE"
