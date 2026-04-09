# Section 10: Unit Tests

## Objective

Comprehensive test suite for all new modules introduced in P07. This section consolidates all test requirements from sections 01-09 into a single test plan with shared fixtures, mocking strategy, and file organization.

## Dependencies

- **section-06-provider-chain** — all provider adapters and chain logic must exist
- **section-07-point-in-time** — PIT query and backfill must exist
- **section-08-staleness-tiering** — tiered thresholds and freshness report must exist
- **section-09-ohlcv-partitioning** — partition utilities must exist

## Files to Create

| File | Description |
|------|-------------|
| `tests/unit/data/test_polygon_provider.py` | Polygon adapter: OHLCV fetch, rate limiting, error handling |
| `tests/unit/data/test_yahoo_provider.py` | Yahoo adapter: OHLCV fetch, caching, graceful degradation |
| `tests/unit/data/test_fmp_provider.py` | FMP adapter: fundamentals fetch, field normalization, daily limit |
| `tests/unit/data/test_provider_chain.py` | Registry fallback: primary success, fallback on failure, all-fail, circuit breaker |
| `tests/unit/data/test_pit_query.py` | PIT: date filtering, NULL exclusion, table whitelist, backfill |
| `tests/unit/data/test_staleness.py` | Staleness: tiered thresholds, session detection, freshness report |
| `tests/unit/db/test_db_decomposition.py` | db.py decomposition: import compatibility |
| `tests/unit/db/test_migrations.py` | Migration runner: pending execution, skip applied, checksum tracking |
| `tests/unit/db/test_partitions.py` | OHLCV partitioning: auto-creation, query equivalence |
| `tests/unit/data/conftest.py` | Shared fixtures: mock providers, test DB connections, populated tables |

## Shared Fixtures (`conftest.py`)

```python
import pytest
from unittest.mock import MagicMock
import pandas as pd
from datetime import datetime, date

@pytest.fixture
def mock_ohlcv_df():
    """Standard OHLCV DataFrame matching expected schema."""
    return pd.DataFrame({
        "timestamp": [datetime(2025, 1, 2), datetime(2025, 1, 3)],
        "open": [150.0, 152.0],
        "high": [155.0, 157.0],
        "low": [149.0, 151.0],
        "close": [153.0, 156.0],
        "volume": [1000000, 1200000],
    })

@pytest.fixture
def mock_av_provider():
    """Mock Alpha Vantage provider that returns data."""
    provider = MagicMock()
    provider.name.return_value = "alpha_vantage"
    return provider

@pytest.fixture
def mock_polygon_provider():
    """Mock Polygon provider."""
    provider = MagicMock()
    provider.name.return_value = "polygon"
    return provider

@pytest.fixture
def mock_yahoo_provider():
    """Mock Yahoo provider."""
    provider = MagicMock()
    provider.name.return_value = "yahoo"
    return provider

@pytest.fixture
def mock_fmp_provider():
    """Mock FMP provider."""
    provider = MagicMock()
    provider.name.return_value = "fmp"
    return provider
```

## Test Plan by Module

### 1. Polygon Provider (`test_polygon_provider.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_polygon_fetch_ohlcv_daily` | Returns DataFrame with correct columns from mocked REST response |
| `test_polygon_fetch_ohlcv_intraday` | Intraday bars parsed correctly |
| `test_polygon_rate_limit` | Rate limiter blocks after N calls/minute |
| `test_polygon_missing_api_key` | Raises `ConfigurationError` when env var absent |
| `test_polygon_http_429` | Rate limit response raises exception (not silently swallowed) |
| `test_polygon_http_5xx` | Server error raises exception for circuit breaker |
| `test_polygon_empty_results` | Empty response returns `None` |

### 2. Yahoo Provider (`test_yahoo_provider.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_yahoo_fetch_ohlcv_daily` | Returns normalized DataFrame |
| `test_yahoo_cache_hit` | Second call uses cache, not yfinance |
| `test_yahoo_cache_expiry` | Cache expires after TTL |
| `test_yahoo_missing_yfinance` | Raises `ConfigurationError` if yfinance not installed |
| `test_yahoo_error_returns_none` | yfinance exceptions caught and return `None` |
| `test_yahoo_output_columns` | Adj Close dropped, columns normalized |

### 3. FMP Provider (`test_fmp_provider.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_fmp_fetch_fundamentals` | Returns dict with AV-compatible field names |
| `test_fmp_fetch_earnings_history` | Returns DataFrame with EPS data |
| `test_fmp_missing_api_key` | Raises `ConfigurationError` |
| `test_fmp_daily_limit` | Raises after 300 requests/day |
| `test_fmp_field_normalization` | FMP fields mapped to AV schema names |

### 4. Provider Chain (`test_provider_chain.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_primary_provider_used_first` | Primary returns data, fallback not called |
| `test_fallback_on_primary_failure` | Primary fails, secondary succeeds |
| `test_fallback_to_tertiary` | Primary+secondary fail, tertiary succeeds |
| `test_all_providers_fail_returns_none` | All fail -> `None` returned |
| `test_circuit_breaker_skips_provider` | Provider with 3+ consecutive failures is skipped |
| `test_not_implemented_skipped_silently` | `NotImplementedError` is not a failure |

### 5. Point-in-Time (`test_pit_query.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_pit_filters_by_available_date` | Only rows with `available_date <= as_of` returned |
| `test_pit_excludes_null_available_date` | NULL available_date rows excluded |
| `test_pit_returns_most_recent` | ORDER BY available_date DESC, LIMIT 1 gets latest available |
| `test_pit_table_whitelist` | Non-whitelisted table raises `ValueError` |
| `test_backfill_financial_statements` | `available_date = reported_date + 1` |
| `test_backfill_insider_trades` | `available_date = filed_date` |
| `test_backfill_institutional_holdings` | `available_date = filed_date + 45 days` |

### 6. Staleness (`test_staleness.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_market_hours_threshold_30min` | 09:30-16:00 ET Mon-Fri = 30 min |
| `test_extended_hours_threshold_8h` | Pre/post market = 8 hours |
| `test_after_hours_threshold_24h` | Overnight + weekends = 24 hours |
| `test_market_session_at_open` | 09:30 ET = market_hours |
| `test_market_session_at_close` | 16:00 ET = extended_hours |
| `test_market_session_weekend` | Saturday/Sunday = after_hours |
| `test_stale_data_fires_event` | system_events row inserted |
| `test_freshness_report_structure` | Returns {symbol: {data_type: datetime}} |

### 7. DB Decomposition (`test_db_decomposition.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_db_conn_from_package` | `from quantstack.db import db_conn` works |
| `test_db_conn_from_connection` | `from quantstack.db.connection import db_conn` works |
| `test_pg_conn_from_connection` | `from quantstack.db.connection import pg_conn` works |
| `test_run_migrations_from_both_paths` | Same function object from both import paths |

### 8. Migrations (`test_migrations.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_run_migrations_creates_tables` | First run creates all tables |
| `test_run_migrations_idempotent` | Second run is a no-op |
| `test_schema_migrations_tracks_applied` | Rows inserted into schema_migrations |
| `test_checksum_recorded` | SHA-256 hex stored for each migration |
| `test_checksum_mismatch_warns` | Modified migration source logs warning |

### 9. Partitions (`test_partitions.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_ensure_partitions_creates_future` | Next month's partition created |
| `test_partition_pruning` | EXPLAIN shows only relevant partition scanned |
| `test_swap_aborts_on_mismatch` | RuntimeError on row count difference |

## Mocking Strategy

- **HTTP APIs (Polygon, FMP):** Use `responses` or `unittest.mock.patch` on `requests.get`/`httpx.get`
- **yfinance:** Mock `yfinance.download()` to return synthetic DataFrames
- **Database:** Use a real test PostgreSQL database (if available via CI) or mock `pg_conn()` with an in-memory mock for unit tests
- **Time:** Use `freezegun` or `monkeypatch` on `datetime.now()` / `time.monotonic()` for staleness tests
- **Environment variables:** Use `monkeypatch.setenv()` / `monkeypatch.delenv()` for API key tests

## Acceptance Criteria

1. All test files exist in the correct directories under `tests/unit/`
2. Every test from the TDD plan (`claude-plan-tdd.md`) is implemented
3. Tests use mocks — no real API calls, no real database connections required for unit tests
4. All tests pass with `uv run pytest tests/unit/data/ tests/unit/db/`
5. Coverage: every public function in the new modules has at least one test
6. Edge cases tested: empty responses, NULL values, rate limits, configuration errors, circuit breaker states
7. Tests are independent — no ordering dependencies, no shared mutable state
