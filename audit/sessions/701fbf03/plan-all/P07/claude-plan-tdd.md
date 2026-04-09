# P07 TDD Plan: Data Architecture Evolution

## Provider Fallback
```python
# Test: ProviderChain returns data from primary when available
# Test: ProviderChain falls back to secondary on primary failure
# Test: ProviderChain falls back to tertiary on primary+secondary failure
# Test: ProviderChain returns error when all providers fail
# Test: Polygon adapter returns correct OHLCV for valid symbol
# Test: Polygon adapter respects rate limit
# Test: Yahoo adapter returns correct daily OHLCV
# Test: Yahoo adapter caches responses for 24h
# Test: FMP adapter returns financial statements
```

## Point-in-Time
```python
# Test: pit_query filters by available_date <= as_of
# Test: pit_query excludes rows with NULL available_date
# Test: pit_query returns most recent available data before cutoff
# Test: backfill sets available_date correctly for financial_statements
# Test: backfill sets available_date correctly for insider_trades
```

## Staleness
```python
# Test: market hours threshold is 30 minutes
# Test: after hours threshold is 8 hours
# Test: weekend threshold is 24 hours
# Test: stale data fires event and disables collector
# Test: freshness report returns correct last_updated per symbol
```

## db.py Decomposition
```python
# Test: from quantstack.db import db_conn still works
# Test: from quantstack.db.connection import db_conn works
# Test: run_migrations executes pending and skips applied
# Test: migration checksum prevents duplicate execution
# Test: schema_migrations table tracks applied migrations
```

## OHLCV Partitioning
```python
# Test: partitioned table returns same results as unpartitioned
# Test: partition auto-creation for future dates
# Test: query performance with partition pruning (EXPLAIN shows partition scan)
```
