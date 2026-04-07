# Section 12: OHLCV Table Partitioning

## Background

The `ohlcv` table holds 7.6M rows with a composite primary key `(symbol, timeframe, timestamp)`. The data breaks down as: 6M rows of 5-minute bars (79%), 1.2M hourly bars (16%), and 400K daily bars (5%). The table grows by approximately 100K rows per day. There is no partitioning, so every query scans the full index regardless of the time range requested. This section introduces monthly range partitioning on `timestamp`, a migration script to convert the existing table, and a startup hook to maintain future partitions automatically.

**No dependencies on other sections.** This section is fully self-contained and can be implemented in isolation. The migration itself should run during a weekend maintenance window with Docker services stopped.

---

## Tests (Write First)

All tests go in `tests/unit/test_ohlcv_partitioning.py`.

```python
# --- Migration script tests ---

# Test: migration script creates partitioned table with correct schema
#   Create a non-partitioned ohlcv table with known rows, run migration logic,
#   verify the resulting table is declared PARTITION BY RANGE (timestamp).

# Test: migration script creates monthly partitions covering all existing data
#   Insert rows spanning Jan 2024 through Apr 2026, run migration, verify
#   partitions ohlcv_2024_01 through ohlcv_2026_04 all exist.

# Test: migration script pre-creates 4 months of future partitions
#   After migration, verify partitions exist for the next 4 calendar months
#   beyond the current date.

# Test: row count after migration matches original table exactly
#   Insert a known number of rows (e.g. 1000), run migration, compare
#   SELECT COUNT(*) on the new partitioned table to the original count.

# Test: composite PK (symbol, timeframe, timestamp) preserved on partitioned table
#   After migration, attempt to insert a duplicate (symbol, timeframe, timestamp).
#   Verify it raises a unique constraint violation.

# Test: default partition catches out-of-range data
#   After migration, insert a row with a timestamp far in the future (beyond
#   pre-created partitions). Verify it lands in ohlcv_default, not rejected.

# Test: EXPLAIN on query with timestamp filter shows partition pruning
#   Run EXPLAIN on SELECT * FROM ohlcv WHERE timestamp >= '2025-06-01'
#   AND timestamp < '2025-07-01'. Verify only one partition is scanned.

# Test: EXPLAIN on query without timestamp filter (document expected behavior)
#   Run EXPLAIN on SELECT * FROM ohlcv WHERE symbol = 'AAPL'. Verify all
#   partitions appear (no pruning without timestamp predicate). This is
#   expected behavior, not a bug.

# Test: rollback procedure works — rename tables back
#   After migration, simulate rollback: rename ohlcv to ohlcv_failed,
#   rename ohlcv_old to ohlcv. Verify original data is accessible.

# --- Startup hook tests ---

# Test: ensure_ohlcv_partitions() creates missing future partitions
#   Set up a partitioned ohlcv table with partitions only through the current
#   month. Call ensure_ohlcv_partitions(). Verify the next 4 months now exist.

# Test: ensure_ohlcv_partitions() is idempotent
#   Call ensure_ohlcv_partitions() twice in sequence. Verify no errors on the
#   second call and partition count is unchanged.
```

**Testing approach:** These tests require a real PostgreSQL instance (partitioning is a DDL feature, not mockable). Use a test database or a `pytest` fixture that creates a temporary schema. If the project already has a database test fixture in `tests/unit/conftest.py`, use it. Otherwise, mark these tests with `@pytest.mark.requires_db` so they can be skipped in CI environments without Postgres.

---

## Implementation Details

### 1. Migration Script

**File to create:** `scripts/migrations/partition_ohlcv.py`

This script performs the full expand-contract migration. It is run manually during a weekend maintenance window, not as part of normal application startup.

**Script responsibilities (in order):**

1. **Pre-flight checks:**
   - Connect to the database using `TRADER_PG_URL` from environment.
   - Count rows in the existing `ohlcv` table and log the count.
   - Verify no active connections to the database (besides this script) by querying `pg_stat_activity`.
   - Determine the date range of existing data: `SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv`.

2. **Create new partitioned table:**
   - Create `ohlcv_new` with the identical column schema as `ohlcv`, declared as `PARTITION BY RANGE (timestamp)`.
   - The composite primary key `(symbol, timeframe, timestamp)` must be included in the partition key (PostgreSQL requires the partition column in the PK). Since `timestamp` is already in the PK, this is satisfied.

3. **Create monthly partitions:**
   - Generate one partition per month from the earliest data month through 4 months beyond the current date.
   - Naming convention: `ohlcv_YYYY_MM` (e.g., `ohlcv_2024_01`, `ohlcv_2025_12`).
   - Each partition: `CREATE TABLE ohlcv_YYYY_MM PARTITION OF ohlcv_new FOR VALUES FROM ('YYYY-MM-01') TO ('YYYY-{MM+1}-01')`.
   - Create a default partition `ohlcv_default PARTITION OF ohlcv_new DEFAULT` to catch any rows outside defined ranges.

4. **Migrate data in monthly batches:**
   - For each month in the date range: `INSERT INTO ohlcv_new SELECT * FROM ohlcv WHERE timestamp >= 'YYYY-MM-01' AND timestamp < 'YYYY-{MM+1}-01'`.
   - Commit after each month to avoid long-running transactions.
   - Log progress: row count per month, cumulative progress.

5. **Verify row counts:**
   - Compare `SELECT COUNT(*) FROM ohlcv` to `SELECT COUNT(*) FROM ohlcv_new`.
   - If counts do not match, abort and log the discrepancy. Do not proceed to the swap.

6. **Atomic swap:**
   ```sql
   BEGIN;
   ALTER TABLE ohlcv RENAME TO ohlcv_old;
   ALTER TABLE ohlcv_new RENAME TO ohlcv;
   COMMIT;
   ```

7. **Recreate indexes:**
   - Recreate any non-PK indexes that existed on the original table. Indexes on a partitioned table automatically propagate to all partitions.

8. **Post-swap validation:**
   - Run sample queries and verify results.
   - Run `EXPLAIN` on a time-filtered query to confirm partition pruning.

9. **Cleanup note:**
   - The script logs a reminder to drop `ohlcv_old` after a 1-week validation period. It does not drop it automatically.

**Script structure (signatures only):**

```python
"""OHLCV table partitioning migration.

Run during a maintenance window with Docker services stopped.
Usage: python scripts/migrations/partition_ohlcv.py
"""

def get_date_range(conn) -> tuple[date, date]:
    """Return (min_date, max_date) from existing ohlcv table."""

def generate_month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Generate (first_of_month, first_of_next_month) pairs covering the range,
    plus 4 months of future partitions."""

def create_partitioned_table(conn, month_ranges: list[tuple[date, date]]) -> None:
    """Create ohlcv_new as a partitioned table with monthly child tables
    and a default partition."""

def migrate_data(conn, month_ranges: list[tuple[date, date]]) -> int:
    """Copy data month-by-month from ohlcv to ohlcv_new. Returns total rows copied."""

def verify_row_counts(conn) -> bool:
    """Compare row counts between ohlcv and ohlcv_new. Returns True if equal."""

def atomic_swap(conn) -> None:
    """Rename ohlcv -> ohlcv_old, ohlcv_new -> ohlcv in a single transaction."""

def recreate_indexes(conn) -> None:
    """Recreate non-PK indexes on the new partitioned table."""

def main() -> None:
    """Orchestrate the full migration with pre-flight, migrate, verify, swap."""
```

**Estimated runtime:** 7.6M rows migrated in monthly batches takes approximately 5-10 minutes for data copy. The full process including verification: 15-20 minutes.

### 2. Startup Partition Maintenance Hook

**File to modify:** `src/quantstack/db.py`

Add a function `ensure_ohlcv_partitions()` that runs during application startup. This function:

1. Checks whether the `ohlcv` table is partitioned (query `pg_class` for `relkind = 'p'`). If not partitioned (e.g., fresh install before migration), skip silently.
2. Determines the current date and computes the next 4 month boundaries.
3. For each future month, checks if the partition exists (query `pg_inherits` or `information_schema`).
4. Creates any missing partitions.
5. Logs what it created (or "all partitions up to date" if none needed).

This function is idempotent — calling it multiple times is safe. It should be called from whatever startup path initializes the database (likely near `ensure_tables()` or equivalent).

**Function signature:**

```python
def ensure_ohlcv_partitions(conn) -> None:
    """Create OHLCV monthly partitions for the next 4 months if they don't exist.
    
    Idempotent. Skips silently if the ohlcv table is not partitioned.
    Called during application startup.
    """
```

The SQL for creating a single partition is straightforward:

```sql
CREATE TABLE IF NOT EXISTS ohlcv_YYYY_MM 
PARTITION OF ohlcv 
FOR VALUES FROM ('YYYY-MM-01') TO ('YYYY-{MM+1}-01');
```

### 3. Schema DDL Update

**File to modify:** `src/quantstack/data/_schema.py`

Update the `ohlcv` table DDL so that fresh installations (no existing data) create the table as partitioned from the start. This means changing the `CREATE TABLE ohlcv` statement to include `PARTITION BY RANGE (timestamp)` and creating a default partition plus partitions for the current and next 4 months.

Existing installations are handled by the migration script; the DDL change is only for new environments.

---

## Rollback Procedure

If the migration fails at any point or issues are discovered after the swap:

1. Stop application services.
2. Run: `ALTER TABLE ohlcv RENAME TO ohlcv_failed; ALTER TABLE ohlcv_old RENAME TO ohlcv;`
3. Restart services.
4. Investigate the failure before retrying.

The `ohlcv_old` table is kept for exactly this purpose and should not be dropped for at least 1 week after a successful migration.

---

## Application Code Impact

**Zero application code changes required.** The partitioned table has the same name, schema, and primary key as the original. All existing queries (inserts, selects, upserts) work unchanged. The only difference is PostgreSQL's query planner now prunes irrelevant partitions when a `WHERE timestamp >= X` filter is present.

Queries that filter only by `symbol` (without a timestamp predicate) will scan all partitions. This is expected and acceptable — those queries are not the hot path.

---

## Risk

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Data corruption during migration | Low | Critical | Full backup before migration. Row count validation gate. 1-week rollback window with `ohlcv_old` retained. |
| Migration takes longer than expected | Low | Low | Monthly batching with per-batch commits and progress logging. Can be paused and resumed. |
| Startup hook fails on non-partitioned table | Low | Medium | `ensure_ohlcv_partitions()` checks `relkind` first; skips silently if table is not partitioned. |
