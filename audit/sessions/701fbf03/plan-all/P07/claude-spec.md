# P07 Spec: Data Architecture Evolution

## Context
QuantStack depends almost entirely on Alpha Vantage for 9/12 data acquisition phases. db.py is a 3,473-line monolith. No point-in-time data support for look-ahead-bias-free backtesting. No schema migration versioning. OHLCV table will hit performance issues as data grows.

## What This Phase Must Deliver

### 1. Multi-Provider Fallback
Add Polygon.io ($29/mo) and Yahoo Finance (free) as OHLCV fallback. Add FMP ($14/mo) as fundamentals fallback. Provider chain: AV → Polygon → Yahoo for OHLCV. AV → FMP → EDGAR for fundamentals.

### 2. Point-in-Time Data Store
Add `available_date` column to financial_statements, earnings_calendar, insider_trades, institutional_holdings. Query pattern: `WHERE available_date < signal_timestamp`.

### 3. Data Staleness Tiering
Replace 8-hour flat threshold with tiered: 30min market hours, 8h after-hours, 24h weekends. Auto-disable signal collectors on stale data.

### 4. db.py Decomposition (Incremental)
Extract connection.py (pool + db_conn) and migrations.py (DDL). Keep queries in db.py.

### 5. Schema Migration Versioning
New `schema_migrations` table. Track which migrations have run. Execute new ones on startup.

### 6. OHLCV Table Partitioning
Partition by timeframe (list) then timestamp (range: monthly for intraday, yearly for daily).

## Constraints
- Budget: <$50/mo for new data providers
- Incremental decomposition (strangler fig, not big-bang)
- Backward compatible — existing code must keep working
- No Alembic — simple version tracking is sufficient
