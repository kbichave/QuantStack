# P07: Data Architecture Evolution

**Objective:** Eliminate Alpha Vantage single-point-of-failure, add multi-provider fallback, point-in-time data store, and real-time streaming architecture.

**Scope:** data/, db.py, signal_engine/

**Depends on:** None

**Enables:** P11 (Alternative Data), P12 (Multi-Asset)

**Effort estimate:** 1 week

---

## What Changes

### 7.1 Multi-Provider Fallback
- AV is sole source for 12/14 acquisition phases (QS: DC2)
- Add secondary providers:
  - OHLCV: Alpaca (existing fallback) + Polygon.io + Yahoo Finance
  - Options: CBOE + Polygon options
  - Fundamentals: SEC EDGAR + Financial Modeling Prep
  - Macro: FRED API (existing but limited)
  - Earnings: FinancialDatasets + Alpaca calendar

### 7.2 Point-in-Time Data Store
- Add `available_date` column to fundamental data tables
- Query pattern: `WHERE available_date < signal_timestamp`
- Enables look-ahead-bias-free backtesting (P04 dependency)

### 7.3 Data Staleness Alerting
- After 3+ consecutive phase failures: alert via EventBus
- Per-symbol freshness dashboard: when was OHLCV/options/fundamentals last refreshed?
- Auto-disable signal collectors when upstream data is stale

### 7.4 db.py Decomposition Strategy
- Current: 140k LOC monolith
- Split into: `db/schema.py`, `db/migrations.py`, `db/queries.py`, `db/connection.py`
- Migration versioning: `schema_migrations` table (QS-I4)
- NOT a rewrite — incremental extraction

### 7.5 Table Partitioning for OHLCV
- `ohlcv` table will grow to hundreds of millions of rows
- Partition by `(timeframe, timestamp)` — monthly partitions
- Add indexes for common access patterns

## Files to Create/Modify

| File | Change |
|------|--------|
| New: `src/quantstack/data/providers/polygon.py` | Polygon.io adapter |
| New: `src/quantstack/data/providers/fmp.py` | Financial Modeling Prep adapter |
| `src/quantstack/data/fetcher.py` | Multi-provider fallback chain |
| `src/quantstack/db.py` | Add available_date, schema_migrations, partitioning |

## Acceptance Criteria

1. OHLCV fetch works when AV is down (Polygon/Yahoo fallback)
2. Point-in-time queries enforce `available_date < signal_time`
3. Staleness alerts fire after 3+ phase failures
4. `schema_migrations` table tracks which migrations have run
