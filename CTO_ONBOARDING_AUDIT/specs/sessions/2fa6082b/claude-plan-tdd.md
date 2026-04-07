# TDD Plan: Phase 8 — Data Pipeline Hardening

Testing framework: **pytest** with async support via `run_async` fixture. Existing fixtures in `tests/unit/conftest.py` provide OHLCV generators, settings mocks, and async helpers. Test files follow `tests/unit/test_*.py` naming.

---

## 2. Signal Cache Auto-Invalidation (Item 8.1)

### Tests to write BEFORE implementing:

```python
# Test: after intraday refresh completes for AAPL, cache.get("AAPL") returns None (invalidated)
# Test: after intraday refresh for [AAPL, MSFT], only those two symbols are invalidated, others remain cached
# Test: after EOD refresh, cache.clear() is called (existing behavior preserved)
# Test: if data write fails for a symbol, that symbol's cache is NOT invalidated (stale > missing)
# Test: cache stats logged after invalidation cycle (mock logger, verify stats message)
# Test: race condition documentation — if signal engine caches during refresh window, brief is re-cached (accepted behavior)
```

---

## 3. Staleness Rejection in Collectors (Item 8.2)

### Staleness helper tests:

```python
# Test: check_freshness returns True when data_metadata.last_timestamp is within max_days
# Test: check_freshness returns False when data_metadata.last_timestamp exceeds max_days
# Test: check_freshness returns False when no data_metadata row exists for symbol/table
# Test: check_freshness uses calendar days (not trading days) — 4 days covers weekends
# Test: check_freshness logs warning with collector name, symbol, actual age, threshold when stale
```

### Per-collector staleness tests (one per category):

```python
# Test: technical collector returns {} when OHLCV data is 5 days old (exceeds 4-day threshold)
# Test: technical collector computes normally when OHLCV data is 1 day old
# Test: macro collector returns {} when macro_indicators data is 50 days old (exceeds 45-day threshold)
# Test: macro collector computes normally when macro_indicators data is 30 days old
# Test: sentiment collector returns {} when news_sentiment data is 10 days old (exceeds 7-day threshold)
# Test: fundamentals collector returns {} when company_overview data is 100 days old (exceeds 90-day threshold)
# Test: ml_signal collector does NOT check staleness (no max_staleness defined)
```

### All-stale edge case:

```python
# Test: when ALL collectors return {} due to staleness, synthesis returns a valid but low-confidence brief (no division by zero)
# Test: when all collectors return {}, brief has confidence < 0.1 or explicit "insufficient_data" flag
```

### Performance benchmark:

```python
# Test: signal engine latency with staleness checks enabled adds < 2s overhead (50 symbols, all data_metadata populated)
# Test: data_metadata rows exist for ALL data source types (prerequisite validation)
```

---

## 4. Data Provider Redundancy — FRED and EDGAR (Item 8.3)

### DataProvider ABC tests:

```python
# Test: provider that only implements name() and fetch_macro_indicator() raises NotImplementedError for all other methods
# Test: provider that doesn't implement a method is distinguishable from provider that returns None (NotImplementedError vs None)
```

### Provider Registry tests:

```python
# Test: registry routes macro_indicator to AV (primary), falls back to FRED on AV failure
# Test: registry routes insider_transactions to EDGAR (primary), falls back to AV on EDGAR failure
# Test: registry increments consecutive_failures on provider failure
# Test: registry resets consecutive_failures to 0 on success
# Test: registry inserts alert into system_events after 3 consecutive failures
# Test: circuit breaker — after 3 failures within 10 minutes, registry skips primary and goes directly to fallback
# Test: circuit breaker resets — after 10 minutes, registry tries primary again
# Test: registry logs structured fields: provider_name, data_type, symbol, latency_ms, success, fallback_used
# Test: NotImplementedError from provider does NOT count as a failure (skips to next provider silently)
```

### Provider initialization tests:

```python
# Test: FREDProvider raises ConfigurationError if FRED_API_KEY not set
# Test: EDGARProvider raises ConfigurationError if EDGAR_USER_AGENT not set
# Test: registry excludes misconfigured providers with warning log
# Test: AVProvider initializes successfully with existing AlphaVantageClient
```

### FRED Provider tests:

```python
# Test: fetch_macro_indicator("DGS10") returns DataFrame with (date, value) columns
# Test: fetch_macro_indicator normalizes FRED series to QuantStack indicator names
# Test: FRED rate limiting respects 120 req/min
# Test: FRED provider returns None (not error) when series has no recent data
```

### EDGAR Provider tests:

```python
# Test: fetch_insider_transactions("AAPL") returns DataFrame matching insider_trades schema
# Test: EDGAR Form 4 data normalized to (ticker, transaction_date, owner_name, transaction_type, shares, price_per_share)
# Test: fetch_fundamentals("AAPL") returns dict matching financial_statements schema
# Test: EDGAR rate limiting respects 10 req/sec
# Test: EDGAR startup health check validates CIK resolution for a known ticker
# Test: EDGAR provider handles ticker with no CIK mapping gracefully (returns None)
```

### AV Provider adapter tests:

```python
# Test: AVProvider.fetch_macro_indicator delegates to AlphaVantageClient.fetch_economic_indicator
# Test: AVProvider.fetch_insider_transactions delegates to AlphaVantageClient.fetch_insider_transactions
# Test: AVProvider preserves existing rate limiting behavior
```

### Failure tracking tests:

```python
# Test: data_provider_failures table created with correct schema
# Test: consecutive failures tracked correctly across multiple calls
# Test: alert inserted on 3rd consecutive failure
# Test: success resets failure counter
```

---

## 5. Drift Detection Pre-Cache (Item 8.4)

### TTLCache per-entry TTL tests:

```python
# Test: TTLCache.set(key, value) with no ttl parameter uses default TTL (backward compatible)
# Test: TTLCache.set(key, value, ttl=300) uses 300s TTL for that entry
# Test: TTLCache.get() returns None for entry past its per-entry TTL even if default TTL hasn't expired
# Test: TTLCache.get() returns value for entry within its per-entry TTL
# Test: existing TTLCache consumers (IC output cache) still work correctly after change
```

### Drift → cache behavior tests:

```python
# Test: NONE drift → brief cached with default TTL, no confidence penalty
# Test: WARNING drift → brief cached with half TTL (1800s), confidence reduced by 0.10
# Test: CRITICAL drift → brief cached with short TTL (300s), confidence reduced by 0.30
# Test: CRITICAL drift → DRIFT_CRITICAL event inserted into system_events
# Test: WARNING drift → no event inserted (log only)
# Test: drift check runs BEFORE cache.put() (verify ordering via mock)
# Test: confidence penalty doesn't push below 0.0
```

---

## 6. SEC Filings Population (Item 8.6)

```python
# Test: sec_filings table created with correct schema (accession_number PK)
# Test: SEC filings acquisition phase populates 10-K, 10-Q, 8-K for a test symbol
# Test: Form 4 insider data normalized and upserted into insider_trades table
# Test: upsert is idempotent — running twice produces same row count
# Test: freshness check skips symbol with filing_date < 90 days ago
# Test: EDGAR rate limiting respected during bulk acquisition (50 symbols)
```

---

## 7. OHLCV Partitioning (Item 8.7)

```python
# Test: migration script creates partitioned table with correct schema
# Test: migration script creates monthly partitions covering all existing data
# Test: migration script pre-creates 4 months of future partitions
# Test: row count after migration matches original table exactly
# Test: EXPLAIN on query with timestamp filter shows partition pruning
# Test: EXPLAIN on query without timestamp filter (document expected behavior)
# Test: composite PK (symbol, timeframe, timestamp) preserved on partitioned table
# Test: ensure_ohlcv_partitions() creates missing future partitions (startup hook)
# Test: ensure_ohlcv_partitions() is idempotent (running twice is safe)
# Test: rollback procedure works — rename tables back
# Test: default partition catches out-of-range data
```

---

## 8. Options Refresh Expansion (Item 8.8)

```python
# Test: OPTIONS_REFRESH_TOP_N env var controls number of symbols refreshed (default 30)
# Test: setting OPTIONS_REFRESH_TOP_N=50 refreshes 50 symbols
# Test: strategy-aware refresh includes symbols from active options strategies
# Test: pre-trade refresh triggers when options data is older than current trading day
# Test: pre-trade refresh has timeout to avoid blocking trading pipeline
# Test: rate limit budget respected when refreshing expanded symbol list
```

---

## Cross-Cutting Tests

```python
# Test: data_metadata populated for ALL data source types after acquisition pipeline run
# Test: .env.example includes FRED_API_KEY and EDGAR_USER_AGENT
# Test: data_provider_failures table DDL in _schema.py
# Test: sec_filings table DDL in _schema.py
```
