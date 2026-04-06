# Section 12: Database Migrations

## Overview

Add the `market_holidays` table and seed it with US market holidays. The `benchmark_daily` table already exists (created in `_migrate_attribution_pg`) and is populated by the benchmark pipeline (`src/quantstack/performance/benchmark.py`), so this section only adds the new table and its seeding logic.

## Dependencies

- **None** for implementation — this section has no prerequisites.
- **Blocks Section 07 (Data & Signals Tab):** The `MarketCalendarWidget` queries `market_holidays`. That widget cannot function until this migration runs.

## Existing State

The migration system lives in `src/quantstack/db.py`. All migrations run inside `run_migrations_pg()`, which:
- Acquires a PostgreSQL advisory lock (`pg_try_advisory_lock`) so only one process runs migrations
- Runs in autocommit mode (each DDL commits immediately)
- Uses `CREATE TABLE IF NOT EXISTS` throughout for idempotency
- Is called once at service startup via `run_migrations(conn)`

Individual migration functions follow the naming convention `_migrate_<name>_pg(conn: PgConnection)` and are called sequentially inside `run_migrations_pg()`.

### benchmark_daily — Already Exists

The `benchmark_daily` table is already created in `_migrate_attribution_pg()` at line ~1006 of `db.py`:

```sql
CREATE TABLE IF NOT EXISTS benchmark_daily (
    date                DATE NOT NULL,
    benchmark           TEXT NOT NULL,
    close_price         DOUBLE PRECISION NOT NULL,
    daily_return_pct    DOUBLE PRECISION,
    cumulative_return   DOUBLE PRECISION,
    PRIMARY KEY (date, benchmark)
)
```

It is populated by `src/quantstack/performance/benchmark.py`, which extracts SPY closes from `ohlcv` and computes returns. No migration work is needed for this table.

**Important for query layer (Section 02):** The column is `benchmark` (not `symbol`) and `close_price` (not `close`). The `fetch_benchmark()` query must use `WHERE benchmark = %s`, not `WHERE symbol = %s`.

## New Table: market_holidays

### Schema

```sql
CREATE TABLE IF NOT EXISTS market_holidays (
    date            DATE NOT NULL,
    name            TEXT NOT NULL,
    market_status   TEXT NOT NULL DEFAULT 'closed',  -- 'closed' | 'early_close'
    close_time      TIME,                            -- NULL for full close, e.g. '13:00' for early close
    exchange        TEXT NOT NULL DEFAULT 'NYSE',
    PRIMARY KEY (date, exchange)
)
```

Design notes:
- Composite PK `(date, exchange)` allows future multi-exchange support without migration. Single-exchange queries add `WHERE exchange = 'NYSE'`.
- `market_status` is constrained to `'closed'` or `'early_close'` by convention (not a CHECK constraint — keeps DDL simple and avoids migration pain if a third status is ever needed).
- `close_time` is `TIME` (not `TIMESTAMPTZ`) because early close times are exchange-local and do not change with daylight saving.

### Seeding

After creating the table, seed US market holidays for the current year and next year. Use `INSERT ... ON CONFLICT DO NOTHING` for idempotency — re-running the migration never duplicates rows.

**Full closures (market_status = 'closed', close_time = NULL):**
- New Year's Day (Jan 1, or observed date if falls on weekend)
- Martin Luther King Jr. Day (3rd Monday of January)
- Presidents' Day (3rd Monday of February)
- Good Friday (date varies — Easter-dependent)
- Memorial Day (last Monday of May)
- Juneteenth (Jun 19, or observed date)
- Independence Day (Jul 4, or observed date)
- Labor Day (1st Monday of September)
- Thanksgiving (4th Thursday of November)
- Christmas (Dec 25, or observed date)

**Early closures (market_status = 'early_close', close_time = '13:00:00'):**
- Day before Independence Day (Jul 3, unless Jul 4 is Monday — then no early close)
- Black Friday (day after Thanksgiving)
- Christmas Eve (Dec 24, unless Christmas is Monday — then no early close)

**Weekend observance rules:**
- If a holiday falls on Saturday, it is observed on Friday.
- If a holiday falls on Sunday, it is observed on Monday.

The seeding function must compute these dates programmatically for the current year and next year using Python's `datetime` and `calendar` modules. Good Friday requires an Easter calculation — use the anonymous Gregorian algorithm (Meeus) or `dateutil.easter` if available. Since `dateutil` is already a transitive dependency of `pandas`, importing `from dateutil.easter import easter` is acceptable.

### Implementation

**File to modify:** `src/quantstack/db.py`

Add a new migration function and register it in the migration sequence:

```python
def _migrate_market_holidays_pg(conn: PgConnection) -> None:
    """Create market_holidays table and seed US holidays for current + next year."""
    ...
```

The function should:

1. Create the table with `CREATE TABLE IF NOT EXISTS`.
2. Call a helper `_seed_market_holidays(conn, year)` for `datetime.date.today().year` and `datetime.date.today().year + 1`.
3. The helper computes all holiday dates for the given year and executes a batch `INSERT ... ON CONFLICT DO NOTHING`.

Register the call in `run_migrations_pg()` alongside the existing migration calls:

```python
_migrate_market_holidays_pg(conn)
```

Place it after `_migrate_market_data_pg(conn)` (which creates `earnings_calendar` and other market data tables), since `market_holidays` is logically a market data table.

## Tests

Test file: `tests/unit/test_tui/test_migrations.py`

All tests mock the database layer. They verify the migration logic (DDL correctness, seeding idempotency, holiday date computation) without requiring a running PostgreSQL instance.

```python
# tests/unit/test_tui/test_migrations.py

# --- market_holidays schema ---
# Test: _migrate_market_holidays_pg calls CREATE TABLE IF NOT EXISTS with correct columns
#   - Verify the DDL string contains: date, name, market_status, close_time, exchange
#   - Verify PRIMARY KEY is (date, exchange)

# --- market_holidays seeding ---
# Test: seeding is idempotent (uses INSERT ... ON CONFLICT DO NOTHING)
#   - Call _seed_market_holidays twice for the same year
#   - Verify no DuplicateKey error (mock conn.execute, inspect SQL for ON CONFLICT)

# Test: seeded holidays include all 10 major US holidays for a known year
#   - Use a fixed year (e.g. 2026) where dates are deterministic
#   - Verify: New Year's (Jan 1), MLK (Jan 19), Presidents' (Feb 16),
#     Good Friday (Apr 3), Memorial Day (May 25), Juneteenth (Jun 19),
#     Independence Day (Jul 3 observed — Jul 4 is Saturday),
#     Labor Day (Sep 7), Thanksgiving (Nov 26), Christmas (Dec 25)

# Test: early closes have market_status='early_close' and close_time='13:00:00'
#   - For 2026: Black Friday (Nov 27), Christmas Eve (Dec 24)
#   - Verify close_time value in the INSERT parameters

# Test: weekend observance rules applied correctly
#   - 2027: New Year's Day is Friday Jan 1 (no shift), Jul 4 is Sunday -> observed Mon Jul 5
#   - Verify the seeded date matches the observed date, not the calendar date

# --- benchmark_daily (no-op verification) ---
# Test: benchmark_daily table already exists in _migrate_attribution_pg
#   - Verify the DDL in _migrate_attribution_pg contains benchmark_daily CREATE TABLE
#   - This is a documentation test — confirms no duplicate table creation needed
```

### Test Implementation Notes

- Mock `PgConnection.execute()` to capture SQL strings and parameters.
- For date computation tests, extract the holiday-computing helper as a pure function `_compute_us_holidays(year: int) -> list[tuple[date, str, str, time | None]]` that returns `(date, name, market_status, close_time)` tuples. This makes it testable without any DB mock.
- For idempotency, verify the SQL string contains `ON CONFLICT` or `ON CONFLICT DO NOTHING`.

## Implementation Checklist

1. Write tests in `tests/unit/test_tui/test_migrations.py`
2. Add `_compute_us_holidays(year: int)` pure function in `db.py` (or a helper module) that returns holiday tuples for a given year
3. Add `_seed_market_holidays(conn, year)` that calls `_compute_us_holidays` and executes the INSERT
4. Add `_migrate_market_holidays_pg(conn)` that creates the table and calls the seeder
5. Register `_migrate_market_holidays_pg(conn)` in `run_migrations_pg()` after `_migrate_market_data_pg(conn)`
6. Run tests: `uv run pytest tests/unit/test_tui/test_migrations.py -v`
