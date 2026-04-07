<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-staleness-helper
section-02-cache-invalidation
section-03-ttlcache-per-entry
section-04-drift-pre-cache
section-05-staleness-collectors
section-06-provider-abc
section-07-fred-provider
section-08-edgar-provider
section-09-provider-registry
section-10-pipeline-integration
section-11-sec-filings
section-12-ohlcv-partitioning
section-13-options-refresh
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-staleness-helper | - | 05 | Yes |
| section-02-cache-invalidation | - | - | Yes |
| section-03-ttlcache-per-entry | - | 04 | Yes |
| section-04-drift-pre-cache | 03 | - | No |
| section-05-staleness-collectors | 01 | - | No |
| section-06-provider-abc | - | 07, 08, 09 | Yes |
| section-07-fred-provider | 06 | 10 | Yes |
| section-08-edgar-provider | 06 | 10, 11 | Yes |
| section-09-provider-registry | 06 | 10 | No |
| section-10-pipeline-integration | 07, 08, 09 | - | No |
| section-11-sec-filings | 08 | - | No |
| section-12-ohlcv-partitioning | - | - | Yes |
| section-13-options-refresh | - | - | Yes |

## Execution Order (Batches)

1. **Batch 1** (no dependencies): section-01, section-02, section-03, section-06, section-12, section-13
2. **Batch 2** (after batch 1): section-04, section-05, section-07, section-08, section-09
3. **Batch 3** (after batch 2): section-10, section-11

## Section Summaries

### section-01-staleness-helper
Create `src/quantstack/signal_engine/staleness.py` with `check_freshness()` function and tests. Extend acquisition pipeline to populate `data_metadata` for all data source types. Corresponds to Item 8.2 foundation.

### section-02-cache-invalidation
Hook `cache.invalidate(symbol)` into `scheduled_refresh.py` after intraday data writes. Add cache stats logging. Corresponds to Item 8.1.

### section-03-ttlcache-per-entry
Modify `src/quantstack/shared/cache.py` TTLCache to support per-entry TTL overrides. Store `(value, timestamp, entry_ttl)` tuples. Backward-compatible change. Corresponds to Item 8.4 prerequisite.

### section-04-drift-pre-cache
Reorder drift detection in `engine.py` to run before cache write. Apply confidence penalties and TTL adjustments based on drift severity. Publish DRIFT_CRITICAL events. Corresponds to Item 8.4.

### section-05-staleness-collectors
Add `check_freshness()` calls to all 22 collectors (except ml_signal). Per-collector thresholds from staleness table. Handle all-stale edge case in synthesis. Corresponds to Item 8.2 bulk work.

### section-06-provider-abc
Create `src/quantstack/data/providers/` package with DataProvider ABC, ConfigurationError, and AV adapter wrapping existing fetcher.py. Corresponds to Item 8.3 foundation.

### section-07-fred-provider
Implement FREDProvider using `fredapi`. Map FRED series IDs to QuantStack indicator names. Add `FRED_API_KEY` to .env.example. Corresponds to Item 8.3 FRED.

### section-08-edgar-provider
Implement EDGARProvider using `edgartools`. MVP: CIK resolution + Form 4 insider parsing. Full: XBRL financials + 13F holdings. Add startup health check. Corresponds to Item 8.3 EDGAR.

### section-09-provider-registry
Implement ProviderRegistry with best-source routing, circuit breaker (10-min cooldown after 3 failures), failure tracking in `data_provider_failures` table, and structured observability logging. Corresponds to Item 8.3 orchestration.

### section-10-pipeline-integration
Rewire `acquisition_pipeline.py` and `scheduled_refresh.py` to use ProviderRegistry instead of direct AlphaVantageClient calls. Corresponds to Item 8.3 integration.

### section-11-sec-filings
Add `sec_filings` table DDL. Add acquisition phases 15 (SEC filings) and 16 (EDGAR insider/institutional). Normalize EDGAR data into existing tables. Corresponds to Item 8.6.

### section-12-ohlcv-partitioning
Create migration script `scripts/migrations/partition_ohlcv.py`. Add `ensure_ohlcv_partitions()` startup hook to `db.py`. Update `_schema.py` DDL. Corresponds to Item 8.7.

### section-13-options-refresh
Extract hardcoded 30 to `OPTIONS_REFRESH_TOP_N` env var. Add strategy-aware symbol selection. Add pre-trade refresh for stale options data. Corresponds to Item 8.8.
