# Implementation Summary — Phase 8: Data Pipeline Hardening

## What Was Implemented

### Section 01: Staleness Helper
Created `src/quantstack/signal_engine/staleness.py` with `check_freshness()` function and `STALENESS_THRESHOLDS` dict (10 data sources). Added `_upsert_metadata()` helper and metadata upsert calls to all non-OHLCV acquisition phases in `acquisition_pipeline.py`.

### Section 02: Cache Invalidation
Added per-symbol signal cache invalidation in intraday refresh (after bulk quotes, 5-min OHLCV, news) and full `cache.clear()` in EOD refresh. Added cache stats logging.

### Section 03: TTLCache Per-Entry TTL
Extended `TTLCache` internal tuple to `(value, timestamp, entry_ttl)`. Added optional `ttl` parameter to `set()`, with per-entry TTL taking precedence over global TTL. Signal cache `put()` pass-through.

### Section 04: Drift Pre-Cache
Reordered drift detection before cache write. Added confidence penalties (WARNING: -0.10, CRITICAL: -0.30), TTL overrides (1800s/300s), and DRIFT_CRITICAL system_events insertion.

### Section 05: Staleness Collectors
Added `check_freshness()` guard at the top of 23 signal collectors. Collectors return `{}` when data is stale, preventing signal generation from outdated data.

### Section 06: Provider ABC
Created `src/quantstack/data/providers/` package with `DataProvider` ABC (10 fetch methods), `ConfigurationError`, and `AVProvider` adapter wrapping the existing `AlphaVantageClient`.

### Section 07: FRED Provider
Created `FREDProvider` with `INDICATOR_TO_FRED` mapping (9 macro series: DGS10, DGS2, T10Y2Y, FEDFUNDS, CPIAUCSL, UNRATE, GDP, BAMLH0A0HYM2, ICSA). Rate limited at 0.5s/request. Returns `(date, value)` DataFrames with NaN removal and date sorting.

### Section 08: EDGAR Provider
Created `EDGARProvider` via `edgartools` library. Implements Form 4 insider transaction parsing (A→buy, D→sell mapping), SEC filing metadata, XBRL fundamentals, and 13F institutional holdings. Rate limited at 0.1s/request (10 req/sec).

### Section 09: Provider Registry
Created `ProviderRegistry` with best-source routing table (11 data types), circuit breaker (3 failures/10-min cooldown), DB-backed failure tracking (`data_provider_failures` table), system_events alerts on 3rd consecutive failure, structured observability logging, and `build_registry()` factory with graceful degradation.

### Section 10: Pipeline Integration
Added `registry` parameter to `AcquisitionPipeline.__init__` (expand-contract pattern). Macro, insider, institutional, and fundamentals phases route through registry when available, fall back to direct AV calls when not. Full backward compatibility preserved.

### Section 11: SEC Filings
Added `sec_filings` table DDL (accession_number PK, symbol/form_type index, filing_date index). Added Phase 15 (SEC filing metadata, 90-day freshness check) and Phase 16 (EDGAR insider transactions, 7-day freshness).

### Section 12: OHLCV Partitioning
Created `scripts/migrations/partition_ohlcv.py` with atomic swap migration (old → partitioned). Added `ensure_ohlcv_partitions()` to `db.py` that auto-creates next 4 months of partitions at startup.

### Section 13: Options Refresh Expansion
Added configurable `OPTIONS_REFRESH_TOP_N` (default 30, env var override). Added strategy-aware symbol selection (`_get_options_strategy_symbols()`) that includes symbols from active options strategies in the EOD options refresh list.

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Expand-contract for registry integration | Allows production validation before removing old code path. Old callers work unchanged. |
| NotImplementedError ≠ failure in registry | FRED raising NotImplementedError for OHLCV is expected (permanent), not transient. Only count actual failures toward circuit breaker. |
| DB-backed circuit breaker state | Failure state survives process restarts. Provider that was failing before restart still has open circuit breaker. |
| FRED as fallback (not primary) for macro | Strangler fig: new dependency starts as fallback. Promote to primary after 2+ weeks of stable production. |
| Per-entry TTL in signal cache | Drift-detected signals get shorter cache TTL (5 min for CRITICAL) without affecting healthy signals. |
| Staleness check at collector level | Each collector independently verifies data freshness. Stale data produces empty signal instead of misleading analysis. |

## Test Results

| Test File | Tests | Status |
|-----------|-------|--------|
| test_staleness_helper.py | 8 | Pass |
| test_cache_invalidation.py | 4 | Pass |
| test_ttlcache_per_entry.py | 7 | Pass |
| test_drift_pre_cache.py | 6 | Pass |
| test_staleness_collectors.py | 8 | Pass |
| test_provider_abc.py | 12 | Pass |
| test_fred_provider.py | 14 | Pass |
| test_edgar_provider.py | 12 | Pass |
| test_provider_registry.py | 14 | Pass |
| test_pipeline_integration.py | 6 | Pass |
| test_sec_filings.py | 7 | Pass |
| test_ohlcv_partitioning.py | 7 | Pass |
| test_options_refresh.py | 4 | Pass |
| **Total** | **109** | **All pass** |

## Files Created or Modified

### Created
| File | Section |
|------|---------|
| `src/quantstack/signal_engine/staleness.py` | 01 |
| `src/quantstack/data/providers/__init__.py` | 06 |
| `src/quantstack/data/providers/base.py` | 06 |
| `src/quantstack/data/providers/alpha_vantage.py` | 06 |
| `src/quantstack/data/providers/fred.py` | 07 |
| `src/quantstack/data/providers/edgar.py` | 08 |
| `src/quantstack/data/providers/registry.py` | 09 |
| `scripts/migrations/partition_ohlcv.py` | 12 |
| `tests/unit/test_staleness_helper.py` | 01 |
| `tests/unit/test_cache_invalidation.py` | 02 |
| `tests/unit/test_ttlcache_per_entry.py` | 03 |
| `tests/unit/test_drift_pre_cache.py` | 04 |
| `tests/unit/test_staleness_collectors.py` | 05 |
| `tests/unit/test_provider_abc.py` | 06 |
| `tests/unit/test_fred_provider.py` | 07 |
| `tests/unit/test_edgar_provider.py` | 08 |
| `tests/unit/test_provider_registry.py` | 09 |
| `tests/unit/test_pipeline_integration.py` | 10 |
| `tests/unit/test_sec_filings.py` | 11 |
| `tests/unit/test_ohlcv_partitioning.py` | 12 |
| `tests/unit/test_options_refresh.py` | 13 |

### Modified
| File | Sections | Changes |
|------|----------|---------|
| `src/quantstack/data/acquisition_pipeline.py` | 01, 10, 11 | _upsert_metadata(), registry param, sec_filings/edgar_insider phases |
| `src/quantstack/data/scheduled_refresh.py` | 02, 13 | Cache invalidation, OPTIONS_REFRESH_TOP_N, strategy-aware symbols |
| `src/quantstack/shared/cache.py` | 03 | Per-entry TTL in TTLCache |
| `src/quantstack/signal_engine/cache.py` | 03 | ttl param in put() |
| `src/quantstack/signal_engine/engine.py` | 04 | Drift confidence penalty + TTL override |
| `src/quantstack/signal_engine/collectors/` (23 files) | 05 | Staleness guard |
| `src/quantstack/data/_schema.py` | 09, 11 | data_provider_failures + sec_filings DDL |
| `src/quantstack/db.py` | 12 | ensure_ohlcv_partitions() |
| `.env.example` | 07, 08, 13 | FRED_API_KEY, EDGAR_USER_AGENT, OPTIONS_REFRESH_TOP_N |
| `pyproject.toml` | 08 | edgartools dependency |

## Known Issues / Remaining TODOs

1. **Contract phase for registry integration** (section 10): The `av_client` parameter on `AcquisitionPipeline` is still the primary interface. After 1+ week of stable production with registry, remove `av_client` and make `registry` required. Update all callers.
2. **FRED promotion**: After 2+ weeks of stable FRED operation, swap routing table to make FRED primary for macro_indicator (currently AV primary, FRED fallback).
3. **OHLCV partition migration**: `scripts/migrations/partition_ohlcv.py` must be run manually as a one-time migration. After that, `ensure_ohlcv_partitions()` handles future months automatically at startup.
4. **`edgartools` version pinning**: Currently `>=3.0.0` — should be pinned more tightly after verifying which version works with the current SEC EDGAR API.
5. **Scheduled refresh still uses AlphaVantageClient directly**: The scheduled_refresh.py was not wired to use the registry (section 10 focused on acquisition_pipeline.py). This is a follow-up item.
