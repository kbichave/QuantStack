# P07 Codebase Research: Data Architecture Evolution

## Current State (from Wave 1 Deep Audit)

### db.py Monolith
- 3,473 LOC, 120+ tables
- Idempotent migration via `CREATE TABLE IF NOT EXISTS` + advisory locks
- Connection pool: psycopg3, 4-20 size, 1h lifetime, 10m idle
- No `schema_versions` table, no rollback mechanism, zero tests
- Backward-compatible `_DictRow` for legacy code

### Data Providers
- **Alpha Vantage**: Primary for 9/12 data types, 75/min premium rate limit
- **FRED**: Fallback for macro only
- **EDGAR**: Fallback for insider/institutional
- **Alpaca**: Paper trading + intraday bars (IEX, 15-min delayed)
- **IBKR**: Stubs only
- **Polygon**: Research mentions, not integrated

### Rate Limiting
- PostgreSQL token bucket (`consume_token()`) shared across containers
- Daily quota gate: 25K calls/day hard limit
- Per-provider circuit breaker: 3 consecutive failures → 10 min cooldown

### Data Freshness
- 8-hour staleness threshold (too loose for intraday)
- No streaming redundancy
- Only AV truly primary for most data types

### Key Gaps
1. AV single-point-of-failure for 9/12 data types
2. No point-in-time `available_date` on fundamentals
3. No schema migration versioning
4. No OHLCV table partitioning
5. db.py is a 3,473-line monolith
