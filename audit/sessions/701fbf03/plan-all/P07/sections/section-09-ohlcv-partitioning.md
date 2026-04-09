# Section 09: OHLCV Table Partitioning

## Objective

Convert the monolithic `ohlcv` table to a partitioned table structure: LIST partition by `timeframe` (daily, hourly, 5min), then RANGE sub-partition by `timestamp` (yearly for daily, monthly for intraday). Includes a safe migration path with a validation period before dropping the legacy table.

## Dependencies

- **section-02-migration-versioning** — partitioning DDL should be added as numbered migrations.

## Files to Create/Modify

### Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/db/migrations.py` | Add migrations for: creating partitioned parent table, creating timeframe partitions, creating sub-partitions, data copy, rename swap |

### New Files

| File | Description |
|------|-------------|
| `src/quantstack/db/partitions.py` | Utility functions: `ensure_partitions_exist()` (auto-create next month's partitions), `create_yearly_partition()`, `create_monthly_partition()` |
| `scripts/partition_maintenance.py` | Cron-callable script that creates upcoming partitions before they are needed |

## Implementation Details

### Step 1: Partitioned Table Structure

Create the new partitioned parent table alongside the existing `ohlcv`:

```sql
-- Parent: partitioned by LIST on timeframe
CREATE TABLE ohlcv_partitioned (
    LIKE ohlcv INCLUDING ALL
) PARTITION BY LIST (timeframe);

-- Timeframe partitions, each sub-partitioned by RANGE on timestamp
CREATE TABLE ohlcv_daily PARTITION OF ohlcv_partitioned
    FOR VALUES IN ('1d')
    PARTITION BY RANGE (timestamp);

CREATE TABLE ohlcv_hourly PARTITION OF ohlcv_partitioned
    FOR VALUES IN ('1h')
    PARTITION BY RANGE (timestamp);

CREATE TABLE ohlcv_5min PARTITION OF ohlcv_partitioned
    FOR VALUES IN ('5m')
    PARTITION BY RANGE (timestamp);
```

### Step 2: Sub-Partitions

**Daily:** yearly sub-partitions

```sql
CREATE TABLE ohlcv_daily_2024 PARTITION OF ohlcv_daily
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE ohlcv_daily_2025 PARTITION OF ohlcv_daily
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
CREATE TABLE ohlcv_daily_2026 PARTITION OF ohlcv_daily
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
```

**Hourly/5min:** monthly sub-partitions (creates 12+ partitions per year per timeframe)

```sql
CREATE TABLE ohlcv_5min_2026_01 PARTITION OF ohlcv_5min
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
-- ... etc for each month
```

### Step 3: Data Migration (Batch, Overnight)

```python
def migration_NNN_copy_ohlcv_to_partitioned(conn):
    """Batch copy from ohlcv to ohlcv_partitioned.
    
    Uses INSERT...SELECT in batches by timeframe to avoid locking
    the source table for extended periods.
    """
    for timeframe in ['1d', '1h', '5m']:
        conn.execute("""
            INSERT INTO ohlcv_partitioned
            SELECT * FROM ohlcv
            WHERE timeframe = %s
            ON CONFLICT DO NOTHING
        """, [timeframe])
        logger.info("[Partition] Copied timeframe=%s", timeframe)
```

### Step 4: Rename Swap

After data is verified:

```python
def migration_NNN_swap_ohlcv_tables(conn):
    """Atomic rename: ohlcv -> ohlcv_legacy, ohlcv_partitioned -> ohlcv.
    
    This migration should only run after verifying row counts match.
    """
    # Verify counts
    old_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    new_count = conn.execute("SELECT COUNT(*) FROM ohlcv_partitioned").fetchone()[0]
    
    if abs(old_count - new_count) > 0:
        raise RuntimeError(
            f"Row count mismatch: ohlcv={old_count}, ohlcv_partitioned={new_count}. "
            "Aborting swap — manual investigation required."
        )
    
    conn.execute("ALTER TABLE ohlcv RENAME TO ohlcv_legacy")
    conn.execute("ALTER TABLE ohlcv_partitioned RENAME TO ohlcv")
    logger.info("[Partition] Swapped: ohlcv -> ohlcv_legacy, ohlcv_partitioned -> ohlcv")
```

### Step 5: Legacy Table Cleanup

After a 7-day validation period, drop the legacy table:

```python
def migration_NNN_drop_ohlcv_legacy(conn):
    """Drop legacy ohlcv table after 7-day validation.
    
    This migration should be added manually after validation confirms
    all queries work against the partitioned table.
    """
    conn.execute("DROP TABLE IF EXISTS ohlcv_legacy")
```

This migration is NOT added to the initial batch — it is added manually after 7 days of production validation.

### Step 6: Automatic Partition Creation

```python
# src/quantstack/db/partitions.py

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

def ensure_partitions_exist(conn, months_ahead: int = 2) -> None:
    """Create partitions for the next N months if they don't exist.
    
    Called by cron or at startup to prevent INSERT failures when
    data arrives for a time period without a partition.
    """
    today = date.today()
    
    # Daily: create yearly partitions
    for year_offset in range(0, 2):
        year = today.year + year_offset
        _create_yearly_partition(conn, "ohlcv_daily", year)
    
    # Hourly + 5min: create monthly partitions
    for tf, parent in [("ohlcv_hourly", "ohlcv_hourly"), ("ohlcv_5min", "ohlcv_5min")]:
        for month_offset in range(0, months_ahead + 1):
            target = today + relativedelta(months=month_offset)
            _create_monthly_partition(conn, parent, target.year, target.month)

def _create_yearly_partition(conn, parent: str, year: int) -> None:
    name = f"{parent}_{year}"
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {name} PARTITION OF {parent}
        FOR VALUES FROM ('{year}-01-01') TO ('{year + 1}-01-01')
    """)

def _create_monthly_partition(conn, parent: str, year: int, month: int) -> None:
    name = f"{parent}_{year}_{month:02d}"
    next_month = date(year, month, 1) + relativedelta(months=1)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {name} PARTITION OF {parent}
        FOR VALUES FROM ('{year}-{month:02d}-01') TO ('{next_month.strftime('%Y-%m-%d')}')
    """)
```

### Step 7: Maintenance Script

```python
# scripts/partition_maintenance.py
"""Monthly cron: create upcoming partitions.

Usage: python scripts/partition_maintenance.py
Crontab: 0 0 1 * * python /path/to/scripts/partition_maintenance.py
"""
from quantstack.db import pg_conn
from quantstack.db.partitions import ensure_partitions_exist

with pg_conn() as conn:
    ensure_partitions_exist(conn, months_ahead=3)
```

## Test Requirements

### TDD Tests

```python
# Test: partitioned table returns same results as unpartitioned
def test_partitioned_query_equivalence(populated_ohlcv_db):
    """Insert identical data into both tables. Same SELECT returns same rows."""
    old_rows = conn.execute("SELECT * FROM ohlcv_legacy WHERE symbol = 'AAPL' ORDER BY timestamp").fetchall()
    new_rows = conn.execute("SELECT * FROM ohlcv WHERE symbol = 'AAPL' ORDER BY timestamp").fetchall()
    assert len(old_rows) == len(new_rows)
    for old, new in zip(old_rows, new_rows):
        assert dict(old) == dict(new)

# Test: partition auto-creation for future dates
def test_ensure_partitions_creates_future(test_db):
    ensure_partitions_exist(conn, months_ahead=2)
    # Verify partition exists for next month
    next_month = date.today() + relativedelta(months=1)
    name = f"ohlcv_5min_{next_month.year}_{next_month.month:02d}"
    exists = conn.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
        [name]
    ).fetchone()[0]
    assert exists is True

# Test: query performance with partition pruning
def test_partition_pruning(populated_partitioned_db):
    """EXPLAIN should show only the relevant partition being scanned."""
    plan = conn.execute(
        "EXPLAIN SELECT * FROM ohlcv WHERE timeframe = '1d' AND timestamp >= '2025-01-01' AND timestamp < '2025-02-01'"
    ).fetchall()
    plan_text = "\n".join(str(row) for row in plan)
    assert "ohlcv_daily_2025" in plan_text
    # Should NOT scan hourly or 5min partitions

# Test: row count matches after copy
def test_migration_copy_preserves_rows(test_db_with_ohlcv):
    old_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    migration_NNN_copy_ohlcv_to_partitioned(conn)
    new_count = conn.execute("SELECT COUNT(*) FROM ohlcv_partitioned").fetchone()[0]
    assert old_count == new_count

# Test: swap aborts on count mismatch
def test_swap_aborts_on_mismatch(test_db):
    """If counts don't match, swap raises RuntimeError."""
    # Insert different row counts into each table
    with pytest.raises(RuntimeError, match="Row count mismatch"):
        migration_NNN_swap_ohlcv_tables(conn)
```

## Acceptance Criteria

1. `ohlcv` table is partitioned by LIST on `timeframe` (1d, 1h, 5m)
2. Daily timeframe has yearly sub-partitions; hourly/5min have monthly sub-partitions
3. All existing data is copied to the partitioned structure with zero row loss
4. Rename swap is atomic and verifies row counts before executing
5. `ohlcv_legacy` is retained for 7 days before manual cleanup
6. `ensure_partitions_exist()` auto-creates upcoming partitions
7. `EXPLAIN` confirms partition pruning works for timeframe + timestamp queries
8. All existing queries against `ohlcv` work unchanged after the swap
9. Maintenance script is cron-callable for monthly partition creation
