"""OHLCV table partitioning migration.

Converts the monolithic ohlcv table (7.6M+ rows) to monthly range
partitions on the `timestamp` column. Run during a maintenance window
with Docker services stopped.

Usage:
    python scripts/migrations/partition_ohlcv.py

Requires TRADER_PG_URL environment variable.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import psycopg
from loguru import logger


def get_connection_url() -> str:
    url = os.environ.get("TRADER_PG_URL")
    if not url:
        logger.error("TRADER_PG_URL environment variable not set")
        sys.exit(1)
    return url


def get_date_range(conn: psycopg.Connection) -> tuple[date, date]:
    """Return (min_date, max_date) from existing ohlcv table."""
    row = conn.execute(
        "SELECT MIN(timestamp)::date, MAX(timestamp)::date FROM ohlcv"
    ).fetchone()
    if row is None or row[0] is None:
        logger.error("ohlcv table is empty — nothing to migrate")
        sys.exit(1)
    return row[0], row[1]


def generate_month_ranges(
    start: date, end: date, future_months: int = 4
) -> list[tuple[date, date]]:
    """Generate (first_of_month, first_of_next_month) pairs covering the range
    plus future_months beyond the current date."""
    first = start.replace(day=1)
    # Extend beyond end date and current date
    last_needed = max(end, date.today())
    # Add future months
    for _ in range(future_months):
        if last_needed.month == 12:
            last_needed = last_needed.replace(year=last_needed.year + 1, month=1, day=1)
        else:
            last_needed = last_needed.replace(month=last_needed.month + 1, day=1)

    ranges = []
    current = first
    while current <= last_needed:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        ranges.append((current, next_month))
        current = next_month
    return ranges


def get_ohlcv_columns(conn: psycopg.Connection) -> str:
    """Get the CREATE TABLE column definitions from the existing ohlcv table."""
    rows = conn.execute("""
        SELECT column_name, data_type, is_nullable, column_default,
               character_maximum_length, numeric_precision, numeric_scale
        FROM information_schema.columns
        WHERE table_name = 'ohlcv' AND table_schema = 'public'
        ORDER BY ordinal_position
    """).fetchall()

    cols = []
    for r in rows:
        col_name, data_type, nullable, default, max_len, num_prec, num_scale = r
        type_str = data_type.upper()
        if max_len:
            type_str = f"VARCHAR({max_len})"
        elif data_type == "numeric" and num_prec:
            type_str = f"NUMERIC({num_prec},{num_scale or 0})"
        elif data_type == "timestamp with time zone":
            type_str = "TIMESTAMPTZ"
        elif data_type == "double precision":
            type_str = "DOUBLE PRECISION"
        elif data_type == "bigint":
            type_str = "BIGINT"
        elif data_type == "integer":
            type_str = "INTEGER"

        col_def = f"    {col_name} {type_str}"
        if nullable == "NO":
            col_def += " NOT NULL"
        if default and "nextval" not in str(default):
            col_def += f" DEFAULT {default}"
        cols.append(col_def)
    return ",\n".join(cols)


def create_partitioned_table(
    conn: psycopg.Connection, month_ranges: list[tuple[date, date]]
) -> None:
    """Create ohlcv_new as a partitioned table with monthly child tables."""
    col_defs = get_ohlcv_columns(conn)

    conn.execute(f"""
        CREATE TABLE ohlcv_new (
        {col_defs},
            PRIMARY KEY (symbol, timeframe, timestamp)
        ) PARTITION BY RANGE (timestamp)
    """)
    logger.info("Created ohlcv_new (partitioned)")

    # Create monthly partitions
    for start, end in month_ranges:
        part_name = f"ohlcv_{start.year}_{start.month:02d}"
        conn.execute(f"""
            CREATE TABLE {part_name} PARTITION OF ohlcv_new
            FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')
        """)
    logger.info(f"Created {len(month_ranges)} monthly partitions")

    # Default partition for out-of-range data
    conn.execute(
        "CREATE TABLE ohlcv_default PARTITION OF ohlcv_new DEFAULT"
    )
    logger.info("Created default partition")
    conn.commit()


def migrate_data(
    conn: psycopg.Connection, month_ranges: list[tuple[date, date]]
) -> int:
    """Copy data month-by-month from ohlcv to ohlcv_new. Returns total rows copied."""
    total = 0
    for i, (start, end) in enumerate(month_ranges):
        result = conn.execute(f"""
            INSERT INTO ohlcv_new
            SELECT * FROM ohlcv
            WHERE timestamp >= '{start.isoformat()}'
              AND timestamp < '{end.isoformat()}'
        """)
        rows = result.rowcount
        total += rows
        conn.commit()
        if rows > 0:
            logger.info(
                f"  [{i+1}/{len(month_ranges)}] {start:%Y-%m}: {rows:,} rows (cumulative: {total:,})"
            )
    return total


def verify_row_counts(conn: psycopg.Connection) -> bool:
    """Compare row counts between ohlcv and ohlcv_new. Returns True if equal."""
    old_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    new_count = conn.execute("SELECT COUNT(*) FROM ohlcv_new").fetchone()[0]
    logger.info(f"Row count verification: old={old_count:,} new={new_count:,}")
    if old_count != new_count:
        logger.error(
            f"ROW COUNT MISMATCH! old={old_count:,} vs new={new_count:,} "
            f"(diff={old_count - new_count:,}). Aborting swap."
        )
        return False
    return True


def atomic_swap(conn: psycopg.Connection) -> None:
    """Rename ohlcv -> ohlcv_old, ohlcv_new -> ohlcv in a single transaction."""
    conn.execute("ALTER TABLE ohlcv RENAME TO ohlcv_old")
    conn.execute("ALTER TABLE ohlcv_new RENAME TO ohlcv")
    conn.commit()
    logger.info("Atomic swap complete: ohlcv_old <- ohlcv <- ohlcv_new")


def recreate_indexes(conn: psycopg.Connection) -> None:
    """Recreate non-PK indexes that existed on the original table."""
    # Check for indexes on ohlcv_old (the original table)
    rows = conn.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'ohlcv_old' AND indexname NOT LIKE '%pkey%'
    """).fetchall()

    for idx_name, idx_def in rows:
        # Rewrite index to point at new table
        new_def = idx_def.replace("ohlcv_old", "ohlcv").replace(idx_name, idx_name.replace("_old", ""))
        try:
            conn.execute(new_def)
            logger.info(f"Recreated index: {idx_name}")
        except Exception as exc:
            logger.warning(f"Failed to recreate index {idx_name}: {exc}")
    conn.commit()


def main() -> None:
    """Orchestrate the full migration."""
    url = get_connection_url()

    logger.info("=" * 60)
    logger.info("OHLCV Partitioning Migration")
    logger.info("=" * 60)

    with psycopg.connect(url, autocommit=False) as conn:
        # Pre-flight
        original_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        logger.info(f"Original ohlcv row count: {original_count:,}")

        min_date, max_date = get_date_range(conn)
        logger.info(f"Date range: {min_date} to {max_date}")

        month_ranges = generate_month_ranges(min_date, max_date)
        logger.info(f"Will create {len(month_ranges)} monthly partitions")

        # Check for existing ohlcv_new (previous failed run)
        existing = conn.execute(
            "SELECT 1 FROM pg_tables WHERE tablename = 'ohlcv_new'"
        ).fetchone()
        if existing:
            logger.warning("ohlcv_new already exists — dropping it (previous failed run?)")
            conn.execute("DROP TABLE ohlcv_new CASCADE")
            conn.commit()

        # Create partitioned table
        create_partitioned_table(conn, month_ranges)

        # Migrate data
        logger.info("Migrating data...")
        total = migrate_data(conn, month_ranges)
        logger.info(f"Migration complete: {total:,} rows copied")

        # Verify
        if not verify_row_counts(conn):
            logger.error("ABORTING — row count mismatch. Dropping ohlcv_new.")
            conn.execute("DROP TABLE ohlcv_new CASCADE")
            conn.commit()
            sys.exit(1)

        # Swap
        atomic_swap(conn)
        recreate_indexes(conn)

        # Post-swap validation
        new_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        logger.info(f"Post-swap row count: {new_count:,}")

        logger.info("=" * 60)
        logger.info("Migration successful!")
        logger.info("REMINDER: Drop ohlcv_old after 1 week of validation:")
        logger.info("  DROP TABLE ohlcv_old CASCADE;")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
