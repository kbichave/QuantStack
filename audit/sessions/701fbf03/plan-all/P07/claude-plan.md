# P07 Implementation Plan: Data Architecture Evolution

## 1. Background

QuantStack's data layer relies almost entirely on Alpha Vantage, creating a single point of failure for 9 of 12 data acquisition phases. The database layer (db.py, 3,473 LOC) lacks migration versioning, point-in-time support, and partitioning. This phase adds provider redundancy, data integrity features, and incremental architectural improvements.

## 2. Anti-Goals

- **Do NOT use Alembic or any ORM migration framework** — simple `schema_migrations` table with named functions is sufficient for this codebase's idempotent DDL pattern.
- **Do NOT decompose db.py into microservices** — extract only connection.py and migrations.py. Queries stay where they are.
- **Do NOT add streaming data** — batch acquisition is the primary pattern. Alpaca streaming exists optionally but is not the focus here.
- **Do NOT add expensive data providers** — CBOE options ($500+/mo), Bloomberg ($2K+/mo) are out of scope for <$100K capital.
- **Do NOT rewrite the acquisition pipeline** — add fallback providers into the existing phase-based pipeline, don't redesign it.

## 3. Multi-Provider Fallback

### 3.1 Provider Architecture

Extend the existing `DataProvider` pattern with a fallback chain:

```python
class ProviderChain:
    """Try providers in order until one succeeds."""
    def __init__(self, providers: list[DataProvider]): ...
    def fetch(self, symbol, start, end, **kwargs) -> DataFrame: ...
```

### 3.2 OHLCV Fallback: AV → Polygon → Yahoo

**Polygon.io adapter** (`src/quantstack/data/providers/polygon.py`):
- REST API for historical bars, aggregates
- Rate limit: 5 calls/min (free), unlimited (paid $29/mo)
- Data: daily + intraday OHLCV, 15+ years history
- Auth: API key in `POLYGON_API_KEY` env var

**Yahoo Finance adapter** (`src/quantstack/data/providers/yahoo.py`):
- Uses `yfinance` library (already in many Python envs)
- No auth required, rate limit: self-imposed 1 req/sec
- Data: daily OHLCV, limited intraday (7 days for 1-min, 60 days for 1-hour)
- Cache: 24h for daily, 1h for intraday (aggressive caching due to unreliability)
- Last-resort only — unreliable, can break without notice

### 3.3 Fundamentals Fallback: AV → FMP → EDGAR

**FMP adapter** (`src/quantstack/data/providers/fmp.py`):
- Financial Modeling Prep API ($14/mo for 300 req/day)
- Data: income statement, balance sheet, cash flow, ratios
- Auth: `FMP_API_KEY` env var

**EDGAR fallback** already partially exists. Enhance for quarterly filing extraction.

### 3.4 Integration into Acquisition Pipeline

In `data/fetcher.py` and `data/acquisition_pipeline.py`:
- Each acquisition phase gets a `ProviderChain` instead of a single provider
- If primary (AV) fails after circuit breaker triggers, try next provider
- Log which provider succeeded for monitoring
- Metrics: per-provider success rate, latency

## 4. Point-in-Time Data Store

### 4.1 Schema Changes

Add `available_date DATE` column to:
- `financial_statements` — set to SEC filing date (or earnings release date)
- `earnings_calendar` — set to announcement date
- `insider_trades` — set to SEC Form 4 filing date
- `institutional_holdings` — set to 13F filing date

### 4.2 Backfill

For existing data where `available_date` is NULL:
- `financial_statements`: `available_date = reported_date + 1 day` (conservative default)
- `insider_trades`: `available_date = filed_date` (from SEC filing metadata)
- `institutional_holdings`: `available_date = filed_date + 45 days` (13F delay)

### 4.3 Query Pattern

Add a helper function `pit_query()` that automatically adds PIT filtering:

```python
def pit_query(table: str, symbol: str, as_of: date, **filters) -> list[dict]:
    """Query with point-in-time filtering: WHERE available_date <= as_of."""
```

Signal collectors that read fundamentals should use `pit_query()` instead of direct SQL.

## 5. Data Staleness Tiering

### 5.1 Tiered Thresholds

Replace flat 8-hour threshold in `data/validator.py`:

```python
STALE_THRESHOLDS = {
    "market_hours": timedelta(minutes=30),     # 09:30-16:00 ET Mon-Fri
    "extended_hours": timedelta(hours=8),      # 04:00-09:30, 16:00-20:00 ET
    "after_hours": timedelta(hours=24),        # overnight + weekends
}
```

### 5.2 Auto-Disable on Stale Data

When a data type is stale during market hours:
1. Log warning with last_update timestamp
2. Fire `data_stale` event via EventBus
3. Auto-disable dependent signal collectors (mapped in config)
4. Alert via `system_alerts` table

### 5.3 Freshness Dashboard

Per-symbol freshness query helper:

```python
def get_freshness_report() -> dict[str, dict[str, datetime]]:
    """Return {symbol: {data_type: last_updated}} for all universe symbols."""
```

## 6. db.py Decomposition

### 6.1 Extract connection.py

New `src/quantstack/db/connection.py`:
- Move: `_pg_pool`, `db_conn()`, `get_pool()`, connection pool setup
- Keep `from quantstack.db import db_conn` working via re-export in `db/__init__.py`
- Zero behavior change — pure extraction

### 6.2 Extract migrations.py

New `src/quantstack/db/migrations.py`:
- Move: all `CREATE TABLE IF NOT EXISTS` blocks
- Each migration becomes a named function: `def migration_001_create_positions(): ...`
- `run_migrations()` function that checks `schema_migrations` and runs pending ones

### 6.3 Backward Compatibility

`src/quantstack/db/__init__.py` re-exports everything from both modules:

```python
from quantstack.db.connection import db_conn, get_pool
from quantstack.db.migrations import run_migrations
```

Existing code like `from quantstack.db import db_conn` continues to work unchanged.

## 7. Schema Migration Versioning

### 7.1 Migration Table

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_name TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    checksum TEXT
);
```

### 7.2 Migration Runner

On startup (after connection pool init):
1. Create `schema_migrations` table if not exists
2. Query applied migrations
3. Compare against registered migrations list
4. Execute new ones in order
5. Record in `schema_migrations` with checksum (SHA-256 of function source)

### 7.3 Idempotency

All migrations use `IF NOT EXISTS` / `IF NOT EXISTS` patterns. Re-running a migration that's already applied is a no-op (but it won't be re-run because it's tracked).

## 8. OHLCV Table Partitioning

### 8.1 Partitioning Strategy

Convert `ohlcv` table to partitioned table:

```sql
-- Parent table (partitioned by LIST on timeframe)
CREATE TABLE ohlcv_partitioned (
    LIKE ohlcv INCLUDING ALL
) PARTITION BY LIST (timeframe);

-- Timeframe partitions (each sub-partitioned by RANGE on timestamp)
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

### 8.2 Sub-Partitions

- Daily: yearly partitions (e.g., `ohlcv_daily_2025`, `ohlcv_daily_2026`)
- Hourly/5min: monthly partitions (e.g., `ohlcv_5min_2026_01`)

### 8.3 Migration Path

1. Create new partitioned table structure
2. Copy data from old `ohlcv` to new (batch, overnight)
3. Rename: `ohlcv` → `ohlcv_legacy`, `ohlcv_partitioned` → `ohlcv`
4. Verify all queries work against partitioned table
5. Drop `ohlcv_legacy` after 7-day validation period

### 8.4 Automatic Partition Creation

Monthly cron job creates next month's partitions before they're needed.

## 9. Testing Strategy

### Unit Tests
- ProviderChain: fallback logic, circuit breaker integration
- Polygon adapter: mock API responses, rate limiting
- PIT query: correct filtering with available_date
- Staleness: tiered threshold detection during market/after hours
- Migration runner: tracks applied, executes new, skips applied

### Integration Tests
- End-to-end fallback: mock AV failure → Polygon succeeds
- OHLCV partitioned queries return same results as unpartitioned

### Edge Cases
- All providers fail → log error, return empty, alert
- PIT query with NULL available_date → exclude row (conservative)
- Partition for future date doesn't exist → auto-create
- db.py decomposition backward compat → all existing imports work
