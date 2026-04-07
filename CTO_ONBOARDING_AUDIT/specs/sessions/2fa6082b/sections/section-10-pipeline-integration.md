# Section 10: Pipeline Integration with ProviderRegistry

## Overview

This section rewires the two main data-fetching entry points -- `acquisition_pipeline.py` and `scheduled_refresh.py` -- to route all data requests through the `ProviderRegistry` instead of calling `AlphaVantageClient` directly. After this change, every fetch goes through the registry's best-source routing, circuit breaker, fallback chain, and structured observability logging. No new providers or abstractions are created here; this is pure integration work that connects the provider infrastructure (sections 06-09) to the existing pipeline code.

## Background

Today, `AcquisitionPipeline.__init__` takes an `AlphaVantageClient` instance (`self._av`) and every phase method calls `self._av.fetch_*()` directly. Similarly, `scheduled_refresh.py` instantiates `AlphaVantageClient()` inside `run_intraday_refresh()` and `run_eod_refresh()` and calls its methods inline. This means:

- No fallback if AV fails -- the phase just records an error and moves on.
- No circuit breaker -- if AV is down, every symbol in every phase pays the timeout penalty before failing.
- No structured observability -- errors are logged as free-text strings, not queryable structured fields.
- Adding a new provider requires touching every phase method individually.

The `ProviderRegistry` built in section-09 solves all of these by providing a single `registry.fetch(data_type, symbol)` call that handles provider selection, fallback, failure tracking, and logging. This section's job is to swap the direct `self._av.fetch_*()` calls for `self._registry.fetch()` calls.

## Dependencies

- **section-06-provider-abc** -- `DataProvider` ABC must exist.
- **section-07-fred-provider** -- `FREDProvider` must be importable and registered.
- **section-08-edgar-provider** -- `EDGARProvider` must be importable and registered.
- **section-09-provider-registry** -- `ProviderRegistry` with routing table, circuit breaker, and failure tracking must be fully implemented.

## Blocked By This Section

- Nothing. This is a terminal section in the dependency graph.

---

## Tests (Write First)

File: `tests/unit/test_pipeline_integration.py`

### AcquisitionPipeline integration tests

```python
def test_acquisition_pipeline_accepts_registry_instead_of_av_client():
    """AcquisitionPipeline.__init__ must accept a ProviderRegistry as its
    primary data source. The old av_client parameter should still work for
    backward compatibility during the transition period, but new callers
    should pass a registry."""

def test_ohlcv_daily_phase_routes_through_registry():
    """The ohlcv_daily phase must call registry.fetch('ohlcv_daily', symbol)
    instead of self._av.fetch_daily(symbol). Mock the registry, run the phase
    for one symbol, assert registry.fetch was called with correct data_type."""

def test_macro_phase_routes_through_registry():
    """The macro phase must call registry.fetch('macro_indicator', indicator)
    for each macro series. This is the phase where FRED fallback matters most.
    Mock the registry and verify all 10 MACRO_SERIES are fetched through it."""

def test_insider_phase_routes_through_registry():
    """The insider phase must call registry.fetch('insider_transactions', symbol).
    This is the phase where EDGAR becomes the primary provider (per the routing
    table). Verify the registry is called, not self._av directly."""

def test_fundamentals_phase_routes_through_registry():
    """The fundamentals phase must use registry.fetch('fundamentals', symbol).
    EDGAR serves as fallback here. Verify routing."""

def test_fallback_fires_when_primary_fails():
    """Configure registry with AV as primary for ohlcv_daily. Make AV raise
    an exception. Verify the registry falls back to Alpaca (or whatever the
    secondary is) and the phase still succeeds."""

def test_circuit_breaker_skips_broken_provider():
    """After 3 consecutive AV failures on macro_indicator, the next call
    should skip AV entirely and go straight to FRED. Verify via mock call
    counts: AV.fetch_macro_indicator called 3 times, then FRED called
    directly on the 4th without trying AV."""

def test_phase_report_counts_fallback_as_success():
    """When the primary provider fails but the fallback succeeds, the
    PhaseReport should count it as succeeded (not failed). The fallback
    usage is tracked in registry logs, not in the phase report."""
```

### scheduled_refresh integration tests

```python
def test_intraday_refresh_uses_registry():
    """run_intraday_refresh must instantiate or receive a ProviderRegistry
    and route bulk quotes, intraday OHLCV, and news through it. No direct
    AlphaVantageClient instantiation should remain in the function."""

def test_eod_refresh_uses_registry():
    """run_eod_refresh must route daily OHLCV, options, fundamentals,
    and earnings through the registry. Verify no direct client.fetch_*
    calls remain."""

def test_intraday_refresh_error_handling_preserved():
    """When the registry raises (all providers failed), the refresh function
    must catch the error, append it to report.errors, log it, and continue
    to the next symbol. The existing error-handling behavior must not regress."""

def test_eod_refresh_fundamentals_staleness_check_preserved():
    """The _get_stale_fundamentals helper that skips recently-refreshed
    symbols must continue to work. The registry integration changes what
    fetches the data, not which symbols are selected for fetching."""
```

### Backward compatibility tests

```python
def test_acquisition_pipeline_works_with_av_client_only():
    """During transition, AcquisitionPipeline must still work when given
    only an AlphaVantageClient (no registry). This supports the expand-
    contract migration: old callers keep working until they switch to
    registry-based instantiation."""

def test_registry_construction_excludes_misconfigured_providers():
    """If FRED_API_KEY is not set, the registry should initialize without
    FREDProvider (logged warning, not a crash). The pipeline must still work
    with only AV available -- just no fallback for macro data."""
```

---

## Implementation Details

### File: `src/quantstack/data/acquisition_pipeline.py`

**Constructor change.** The `AcquisitionPipeline.__init__` signature changes to accept a `ProviderRegistry` as the primary interface. For backward compatibility during the transition, keep the old `av_client` parameter as optional. If a registry is provided, use it; if only `av_client` is provided, wrap it in a minimal single-provider registry automatically.

```python
def __init__(
    self,
    registry: ProviderRegistry | None = None,
    *,
    av_client: AlphaVantageClient | None = None,
    store: DataStore,
    alpaca: AlpacaAdapter | None = None,
) -> None:
    """Initialize acquisition pipeline.

    Prefer passing a ProviderRegistry. The av_client parameter is
    retained for backward compatibility but deprecated.
    """
```

If both `registry` and `av_client` are `None`, raise a clear error at construction time (fail fast).

**Phase method changes.** Each `run_*` method and its private `_fetch_*` helper currently calls `self._av.fetch_*(...)`. Replace these with `self._registry.fetch(data_type, ...)` calls. The mapping from phase to registry data type:

| Phase method | Current call | New registry call |
|---|---|---|
| `run_ohlcv_5min` | `self._av.fetch_intraday_by_month(symbol, ...)` | `self._registry.fetch("ohlcv_intraday", symbol, interval="5min", month=month_str)` |
| `run_ohlcv_1h` | `self._av.fetch_intraday_by_month(symbol, "60min", ...)` | `self._registry.fetch("ohlcv_intraday", symbol, interval="60min", month=month_str)` |
| `run_ohlcv_daily` | `self._av.fetch_daily(symbol, ...)` | `self._registry.fetch("ohlcv_daily", symbol)` |
| `run_financials` | `self._av.fetch_income_statement(symbol)` etc. | `self._registry.fetch("fundamentals", symbol)` |
| `run_earnings_history` | `self._av.fetch_earnings(symbol)` | `self._registry.fetch("earnings_history", symbol)` |
| `run_macro` | `self._av.fetch_economic_indicator(fn, interval, maturity)` | `self._registry.fetch("macro_indicator", indicator, interval=interval)` |
| `run_insider` | `self._av.fetch_insider_transactions(symbol)` | `self._registry.fetch("insider_transactions", symbol)` |
| `run_institutional` | `self._av.fetch_institutional_ownership(symbol)` | `self._registry.fetch("institutional_holdings", symbol)` |
| `run_options` | `self._av.fetch_realtime_options(symbol)` | `self._registry.fetch("options_chain", symbol)` |
| `run_news` | `self._av.fetch_news_sentiment(...)` | `self._registry.fetch("news_sentiment", tickers=...)` |
| `run_fundamentals` | `self._av.fetch_company_overview(symbol)` | `self._registry.fetch("fundamentals", symbol)` |
| `run_commodities` | `self._av.fetch_commodity(indicator)` | `self._registry.fetch("commodities", indicator)` |

**Key design decision: `fetch()` parameter passing.** The registry's `fetch(data_type, symbol, **kwargs)` must forward extra parameters (like `interval`, `month`, `outputsize`) to the underlying provider. Each provider implementation decides which kwargs it supports. Unsupported kwargs are ignored (not errors) -- this keeps the registry generic.

**Error handling stays at the phase level.** The registry handles provider-level failures (fallback, circuit breaker, failure tracking). But the phase methods still catch exceptions from the registry's `fetch()` call, because a total failure (all providers down for this data type) still needs to be recorded in the `PhaseReport`. The existing `try/except` blocks in each phase remain -- they just wrap `self._registry.fetch()` instead of `self._av.fetch_*()`.

### File: `src/quantstack/data/scheduled_refresh.py`

**Replace inline `AlphaVantageClient()` instantiation.** Both `run_intraday_refresh()` and `run_eod_refresh()` currently do:

```python
from quantstack.data.fetcher import AlphaVantageClient
client = AlphaVantageClient()
```

Replace with:

```python
from quantstack.data.providers.registry import ProviderRegistry
registry = ProviderRegistry.default()
```

The `ProviderRegistry.default()` class method (defined in section-09) constructs a registry with all available providers based on environment variables. If `FRED_API_KEY` is not set, FRED is excluded. If `EDGAR_USER_AGENT` is not set, EDGAR is excluded. AV is always included (it only requires `ALPHA_VANTAGE_API_KEY`, which is already mandatory).

**Intraday refresh changes:**
- Bulk quotes: `client.fetch_bulk_quotes(symbols)` becomes `registry.fetch("ohlcv_intraday", symbols, bulk=True)` or keep as a special case if the registry doesn't support batch fetches. Pragmatic option: the `AVProvider` can expose a `fetch_bulk_quotes` method and the registry routes it. Only AV supports bulk quotes, so no fallback needed here.
- 5-min OHLCV: `client.fetch_intraday(sym, "5min", "compact")` becomes `registry.fetch("ohlcv_intraday", sym, interval="5min", outputsize="compact")`.
- News sentiment: `client.fetch_news_sentiment(tickers=..., topics=..., limit=...)` becomes `registry.fetch("news_sentiment", tickers=tickers_str, limit=50)`.

**EOD refresh changes:**
- Daily OHLCV: `client.fetch_daily(sym, "compact")` becomes `registry.fetch("ohlcv_daily", sym, outputsize="compact")`.
- Options: `client.fetch_realtime_options(sym)` becomes `registry.fetch("options_chain", sym)`.
- Fundamentals: `client.fetch_company_overview(sym)` becomes `registry.fetch("fundamentals", sym)`.
- Earnings calendar: `client.fetch_earnings_calendar(None, "3month")` becomes `registry.fetch("earnings_history", horizon="3month")`.

**The `PgDataStore` usage is unchanged.** The registry handles fetching; the store handles persistence. These are separate concerns and the store code is not touched.

### Callers of AcquisitionPipeline

Any code that constructs an `AcquisitionPipeline` must be updated to pass a `ProviderRegistry` instead of (or in addition to) an `AlphaVantageClient`. Search for `AcquisitionPipeline(` across the codebase to find all construction sites. Expected locations:

- Graph node code that triggers acquisition (research graph, supervisor graph)
- CLI scripts (if any exist for manual data pulls)
- Test fixtures

Each construction site should be updated to use `ProviderRegistry.default()`. The Alpaca adapter can still be passed separately since it serves a different role (real-time streaming, not historical data).

### Observability

After the integration, every data fetch automatically gets the registry's structured logging: `provider_name`, `data_type`, `symbol`, `latency_ms`, `success`, `fallback_used`, `circuit_breaker_tripped`. This replaces the ad-hoc error logging in the current phase methods. Keep the phase-level `PhaseReport` logging (total/succeeded/skipped/failed per phase) as a higher-level summary -- it serves a different purpose than per-call observability.

### Migration Strategy (Expand-Contract)

This integration follows the expand-contract pattern:

1. **Expand:** Add the `registry` parameter to `AcquisitionPipeline.__init__` alongside the existing `av_client` parameter. Both work. New code passes the registry; old code continues passing `av_client`.
2. **Validate:** Run the full acquisition pipeline with the registry. Verify all phases produce the same results as before (same row counts, same data quality). The registry should log every provider call with timing.
3. **Contract:** Once validated, remove the `av_client` parameter from `AcquisitionPipeline.__init__`. Update all callers. Delete the backward-compatibility shim.

Do not skip step 2. The expand phase must run in production (paper mode) for at least one full acquisition cycle before contracting.

### What NOT to Change

- **Rate limiting logic inside `AlphaVantageClient`** -- stays as-is. The `AVProvider` adapter wraps it.
- **`DataStore` / `PgDataStore`** -- persistence layer is unchanged. The registry only handles fetching.
- **Phase sequencing** -- phases still run sequentially in `AcquisitionPipeline.run()`. The registry does not introduce parallelism at the phase level.
- **OHLCV delta logic** -- the `_fetch_5min_delta`, `_fetch_1h_delta`, `_fetch_daily_delta` helpers contain idempotency logic (check last timestamp, skip if fresh). This logic stays; only the fetch call inside it changes.
- **`_get_active_symbols()` and `_get_watched_symbols()`** -- symbol selection is unchanged.

---

## Verification Checklist

After implementation, verify:

1. Run full acquisition pipeline for 3 symbols with all phases -- all phases succeed with the registry.
2. Simulate AV failure (mock or env var) -- verify FRED handles macro, EDGAR handles insider/fundamentals via fallback.
3. Check structured logs -- every fetch call should have `provider_name`, `data_type`, `symbol`, `latency_ms`, `success`, `fallback_used` fields.
4. Verify `PhaseReport` output matches pre-integration format (no regressions in logging).
5. Confirm no direct `AlphaVantageClient` instantiation remains in `scheduled_refresh.py`.
6. Confirm `AcquisitionPipeline` still works when constructed with only AV (no FRED/EDGAR keys set).
7. Run existing tests -- no regressions in `tests/unit/` test suite.
