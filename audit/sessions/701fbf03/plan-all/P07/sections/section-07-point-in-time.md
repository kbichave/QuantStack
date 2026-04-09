# Section 07: Point-in-Time Data Store

## Objective

Add `available_date` columns to fundamental data tables, implement backfill logic for existing rows, and provide a `pit_query()` helper that ensures research and signal generation never use data that was not yet publicly available on a given date (preventing look-ahead bias).

## Dependencies

- **section-02-migration-versioning** — the schema changes should be added as numbered migrations.

## Files to Create/Modify

### New Files

| File | Description |
|------|-------------|
| `src/quantstack/data/pit.py` | `pit_query()` helper function and `backfill_available_dates()` utility |

### Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/db/migrations.py` | Add migrations to ALTER TABLE and add `available_date` columns; add backfill migration |
| `src/quantstack/core/features/fundamental.py` | Update fundamental feature queries to use `pit_query()` |
| `src/quantstack/signal_engine/collectors/*.py` | Update any collector that reads fundamentals to use `pit_query()` |

## Implementation Details

### Step 1: Schema Migrations

Add new numbered migrations in `migrations.py`:

```python
def migration_NNN_add_available_date_financial_statements(conn):
    conn.execute("""
        ALTER TABLE financial_statements
        ADD COLUMN IF NOT EXISTS available_date DATE
    """)

def migration_NNN_add_available_date_earnings_calendar(conn):
    conn.execute("""
        ALTER TABLE earnings_calendar
        ADD COLUMN IF NOT EXISTS available_date DATE
    """)

def migration_NNN_add_available_date_insider_trades(conn):
    conn.execute("""
        ALTER TABLE insider_trades
        ADD COLUMN IF NOT EXISTS available_date DATE
    """)

def migration_NNN_add_available_date_institutional_holdings(conn):
    conn.execute("""
        ALTER TABLE institutional_holdings
        ADD COLUMN IF NOT EXISTS available_date DATE
    """)
```

### Step 2: Backfill Migration

A separate migration that sets `available_date` for existing rows where it is NULL:

```python
def migration_NNN_backfill_available_dates(conn):
    """Conservative backfill for existing data.
    
    Rules:
      - financial_statements: available_date = reported_date + 1 day
        (conservative: assumes data available day after reporting)
      - insider_trades: available_date = filed_date
        (SEC Form 4 is public immediately upon filing)
      - institutional_holdings: available_date = filed_date + 45 days
        (13F filings have up to 45-day reporting delay)
      - earnings_calendar: available_date = announcement_date
        (earnings are public at announcement)
    """
    conn.execute("""
        UPDATE financial_statements
        SET available_date = reported_date + INTERVAL '1 day'
        WHERE available_date IS NULL AND reported_date IS NOT NULL
    """)
    
    conn.execute("""
        UPDATE insider_trades
        SET available_date = filed_date
        WHERE available_date IS NULL AND filed_date IS NOT NULL
    """)
    
    conn.execute("""
        UPDATE institutional_holdings
        SET available_date = filed_date + INTERVAL '45 days'
        WHERE available_date IS NULL AND filed_date IS NOT NULL
    """)
    
    conn.execute("""
        UPDATE earnings_calendar
        SET available_date = announcement_date
        WHERE available_date IS NULL AND announcement_date IS NOT NULL
    """)
```

### Step 3: pit_query() Helper

```python
# src/quantstack/data/pit.py

from datetime import date
from quantstack.db import pg_conn

def pit_query(
    table: str,
    symbol: str,
    as_of: date,
    columns: str = "*",
    extra_where: str = "",
    extra_params: list | None = None,
    order_by: str = "available_date DESC",
    limit: int | None = None,
) -> list[dict]:
    """Query with point-in-time filtering.
    
    Guarantees: only returns rows where available_date <= as_of.
    Rows with NULL available_date are EXCLUDED (conservative — prevents
    look-ahead from data without known publication date).
    
    Args:
        table: Table name (must be whitelisted to prevent SQL injection)
        symbol: Stock symbol to filter on
        as_of: Only return data available on or before this date
        columns: Column selection (default "*")
        extra_where: Additional WHERE clause (AND-joined)
        extra_params: Parameters for extra_where
        order_by: ORDER BY clause (default: most recent first)
        limit: Optional row limit
    
    Returns:
        List of row dicts
    """
    _ALLOWED_TABLES = {
        "financial_statements",
        "earnings_calendar", 
        "insider_trades",
        "institutional_holdings",
    }
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"pit_query: table '{table}' not in allowed list")
    
    params = [symbol, as_of]
    where = f"symbol = %s AND available_date IS NOT NULL AND available_date <= %s"
    
    if extra_where:
        where += f" AND ({extra_where})"
        if extra_params:
            params.extend(extra_params)
    
    sql = f"SELECT {columns} FROM {table} WHERE {where} ORDER BY {order_by}"
    if limit:
        sql += f" LIMIT {limit}"
    
    with pg_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]
```

### Step 4: Update Consumers

Any code that reads fundamentals for backtesting or signal generation should use `pit_query()` instead of direct SQL. Key locations:

1. **`src/quantstack/core/features/fundamental.py`** — feature extraction from financial statements
2. **Signal collectors** that reference earnings or insider data

Before:
```python
rows = conn.execute("SELECT * FROM financial_statements WHERE symbol = %s", [symbol]).fetchall()
```

After:
```python
from quantstack.data.pit import pit_query
rows = pit_query("financial_statements", symbol, as_of=backtest_date)
```

### Step 5: New Data Population

When new fundamental data is acquired (via AV, FMP, or EDGAR), the acquisition pipeline should populate `available_date`:

- For SEC filings: use the filing date from the SEC metadata
- For earnings releases: use the announcement date
- For financial statements from AV/FMP: use the `reportedDate` or `fillingDate` field

## Test Requirements

### TDD Tests

```python
# Test: pit_query filters by available_date <= as_of
def test_pit_query_filters_by_date(populated_db):
    """Insert rows with available_date 2025-01-15 and 2025-03-15.
    Query as_of=2025-02-01 should return only the Jan row."""
    rows = pit_query("financial_statements", "AAPL", as_of=date(2025, 2, 1))
    assert len(rows) == 1
    assert rows[0]["available_date"] == date(2025, 1, 15)

# Test: pit_query excludes rows with NULL available_date
def test_pit_query_excludes_null_available_date(populated_db):
    """Insert row with available_date=NULL. Should not appear in results."""
    rows = pit_query("financial_statements", "AAPL", as_of=date(2026, 1, 1))
    for row in rows:
        assert row["available_date"] is not None

# Test: pit_query returns most recent available data before cutoff
def test_pit_query_returns_most_recent(populated_db):
    """Insert rows for Q1, Q2, Q3. Query as_of mid-Q3 should return Q2 as most recent."""
    rows = pit_query("financial_statements", "AAPL", as_of=date(2025, 8, 1), limit=1)
    assert len(rows) == 1
    # Should be Q2 data (available ~July), not Q3 (not yet available)

# Test: backfill sets available_date correctly for financial_statements
def test_backfill_financial_statements(db_with_null_dates):
    backfill_available_dates(conn)
    row = conn.execute(
        "SELECT available_date, reported_date FROM financial_statements WHERE symbol = %s",
        ["AAPL"]
    ).fetchone()
    assert row["available_date"] == row["reported_date"] + timedelta(days=1)

# Test: backfill sets available_date correctly for insider_trades
def test_backfill_insider_trades(db_with_null_dates):
    backfill_available_dates(conn)
    row = conn.execute(
        "SELECT available_date, filed_date FROM insider_trades WHERE symbol = %s",
        ["AAPL"]
    ).fetchone()
    assert row["available_date"] == row["filed_date"]

# Test: SQL injection prevention via table whitelist
def test_pit_query_rejects_unknown_table():
    with pytest.raises(ValueError, match="not in allowed list"):
        pit_query("users; DROP TABLE positions", "AAPL", as_of=date(2025, 1, 1))
```

## Acceptance Criteria

1. `available_date DATE` column exists on `financial_statements`, `earnings_calendar`, `insider_trades`, `institutional_holdings`
2. Backfill migration populates `available_date` for all existing rows using conservative defaults
3. `pit_query()` filters by `available_date <= as_of` and excludes NULL available_date rows
4. `pit_query()` rejects table names not in the whitelist (SQL injection prevention)
5. Feature extraction code in `fundamental.py` uses `pit_query()` for backtesting contexts
6. New data acquisition populates `available_date` at insert time
7. All migrations are tracked in `schema_migrations`
