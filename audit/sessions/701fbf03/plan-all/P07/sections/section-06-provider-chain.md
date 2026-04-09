# Section 06: Provider Chain Integration

## Objective

Wire the new providers (Polygon, Yahoo, FMP) into the existing `ProviderRegistry` and acquisition pipeline. Verify end-to-end fallback behavior: when the primary provider fails (circuit breaker trips), the registry seamlessly tries the next provider in the chain.

## Dependencies

- **section-03-polygon-adapter** — Polygon provider must exist
- **section-04-yahoo-adapter** — Yahoo provider must exist
- **section-05-fmp-adapter** — FMP provider must exist

## Files to Create/Modify

### Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/data/providers/registry.py` | Update `_ROUTING_TABLE` with full fallback chains. Update `build_registry()` to instantiate all providers. Add per-provider success/latency metrics logging |
| `src/quantstack/data/providers/__init__.py` | Export new provider classes |
| `src/quantstack/data/fetcher.py` | (if exists) Ensure fetcher uses registry for all data types instead of direct AV calls |
| `src/quantstack/data/acquisition_pipeline.py` | (if exists) Ensure each acquisition phase routes through the registry |

## Implementation Details

### Step 1: Complete Routing Table

Update `_ROUTING_TABLE` to include all new providers:

```python
_ROUTING_TABLE: dict[str, list[str]] = {
    "ohlcv_daily":            ["alpha_vantage", "polygon", "yahoo"],
    "ohlcv_intraday":         ["alpha_vantage", "polygon", "yahoo"],
    "macro_indicator":        ["alpha_vantage", "fred"],
    "fundamentals":           ["alpha_vantage", "fmp", "edgar"],
    "earnings_history":       ["alpha_vantage", "fmp", "edgar"],
    "insider_transactions":   ["edgar", "alpha_vantage"],
    "institutional_holdings": ["edgar", "alpha_vantage"],
    "options_chain":          ["alpha_vantage"],
    "news_sentiment":         ["alpha_vantage"],
    "sec_filings":            ["edgar"],
    "commodities":            ["alpha_vantage", "fred"],
}
```

### Step 2: Update `build_registry()`

```python
def build_registry() -> ProviderRegistry:
    from quantstack.data.providers.alpha_vantage import AVProvider
    from quantstack.data.providers.polygon import PolygonProvider
    from quantstack.data.providers.yahoo import YahooProvider
    from quantstack.data.providers.fmp import FMPProvider
    from quantstack.data.providers.edgar import EDGARProvider
    from quantstack.data.providers.fred import FREDProvider

    providers: list[DataProvider] = []
    for cls in [AVProvider, PolygonProvider, YahooProvider, FMPProvider, FREDProvider, EDGARProvider]:
        try:
            providers.append(cls())
        except ConfigurationError as exc:
            logger.warning("[Registry] Skipping %s: %s", cls.__name__, exc)
        except Exception as exc:
            logger.warning("[Registry] Unexpected error initializing %s: %s", cls.__name__, exc)

    return ProviderRegistry(providers)
```

### Step 3: Enhanced Metrics Logging

Add structured logging to `ProviderRegistry.fetch()` for monitoring which provider actually served data:

```python
logger.info(
    "[Registry] provider_fetch provider=%s data_type=%s symbol=%s "
    "latency_ms=%.1f success=True fallback_used=%s position_in_chain=%d",
    provider_name, data_type, symbol, latency_ms, fallback_used, i,
)
```

The `position_in_chain` field (0-indexed) tells monitoring which provider actually served the request. If this is consistently > 0 for a data type, the primary is degraded.

### Step 4: Verify Acquisition Pipeline Integration

Audit `data/fetcher.py` and `data/acquisition_pipeline.py` to ensure they call `registry.fetch(data_type, symbol)` rather than calling AV directly. Any direct AV calls should be refactored to go through the registry.

Key acquisition phases to verify:
- Phase 1 (OHLCV daily) — should use `registry.fetch("ohlcv_daily", symbol)`
- Phase 2 (OHLCV intraday) — should use `registry.fetch("ohlcv_intraday", symbol)`
- Phase 5 (fundamentals) — should use `registry.fetch("fundamentals", symbol)`
- Phase 7 (earnings) — should use `registry.fetch("earnings_history", symbol)`

### Step 5: Graceful Degradation

When ALL providers fail for a data type:
1. Registry already logs `"All providers exhausted"` and returns `None`
2. Ensure the acquisition pipeline handles `None` return by logging the gap and continuing (not crashing)
3. Insert a row into `system_events` with severity `"error"` when all providers fail

## Test Requirements

### TDD Tests (Integration-Level)

```python
# Test: ProviderChain returns data from primary when available
def test_registry_uses_primary(mock_av_success, mock_polygon):
    registry = build_test_registry([mock_av_success, mock_polygon])
    result = registry.fetch("ohlcv_daily", "AAPL")
    assert result is not None
    assert mock_av_success.fetch_ohlcv_daily.called
    assert not mock_polygon.fetch_ohlcv_daily.called

# Test: ProviderChain falls back to secondary on primary failure
def test_registry_falls_back_on_primary_failure(mock_av_failure, mock_polygon_success):
    registry = build_test_registry([mock_av_failure, mock_polygon_success])
    result = registry.fetch("ohlcv_daily", "AAPL")
    assert result is not None
    assert mock_polygon_success.fetch_ohlcv_daily.called

# Test: ProviderChain falls back to tertiary on primary+secondary failure
def test_registry_falls_to_tertiary(mock_av_failure, mock_polygon_failure, mock_yahoo_success):
    registry = build_test_registry([mock_av_failure, mock_polygon_failure, mock_yahoo_success])
    result = registry.fetch("ohlcv_daily", "AAPL")
    assert result is not None

# Test: ProviderChain returns None when all providers fail
def test_registry_all_fail(mock_av_failure, mock_polygon_failure, mock_yahoo_failure):
    registry = build_test_registry([mock_av_failure, mock_polygon_failure, mock_yahoo_failure])
    result = registry.fetch("ohlcv_daily", "AAPL")
    assert result is None

# Test: Circuit breaker skips broken provider
def test_circuit_breaker_skips_broken_provider(mock_av_circuit_broken, mock_polygon_success):
    # After 3 consecutive AV failures, registry should skip AV and go to Polygon
    ...
```

### Test Helpers

Create a `build_test_registry()` helper that accepts mock providers and constructs a registry without needing real API keys or database connections.

## Acceptance Criteria

1. `_ROUTING_TABLE` includes Polygon, Yahoo, and FMP in the correct positions
2. `build_registry()` instantiates all six providers (gracefully skipping unconfigured ones)
3. End-to-end fallback works: AV failure -> Polygon -> Yahoo for OHLCV; AV failure -> FMP -> EDGAR for fundamentals
4. Structured logging includes `position_in_chain` for monitoring
5. All providers that fail with `ConfigurationError` are excluded without crashing
6. When all providers fail, `None` is returned and a `system_events` entry is created
7. No acquisition pipeline code calls AV directly — all requests go through the registry
