# Section 06: DataProvider ABC and Alpha Vantage Adapter

## Overview

This section creates the foundation for multi-provider data redundancy: an abstract base class (`DataProvider`) that defines a uniform interface for all data sources, a custom `ConfigurationError` for provider initialization failures, and a thin adapter that wraps the existing `AlphaVantageClient` behind the new interface. All subsequent provider sections (FRED, EDGAR, registry) depend on this section.

## Background

QuantStack currently fetches all market data through `src/quantstack/data/fetcher.py`, which contains a monolithic `AlphaVantageClient` class with 30+ `fetch_*` methods. The acquisition pipeline (`acquisition_pipeline.py`) and scheduled refresh (`scheduled_refresh.py`) call this client directly. If Alpha Vantage goes down or exhausts its rate limit, nearly every data category goes dark.

The goal is to introduce a provider abstraction layer so that multiple data sources (AV, FRED, EDGAR, Alpaca) can be routed through a single registry. This section builds the foundation; subsequent sections add concrete providers and the registry.

## Dependencies

- **None.** This section has no dependencies and can be implemented in parallel with sections 01-05, 12, and 13.

## Blocked By This Section

- **section-07-fred-provider** (FREDProvider extends DataProvider)
- **section-08-edgar-provider** (EDGARProvider extends DataProvider)
- **section-09-provider-registry** (ProviderRegistry consumes DataProvider instances)

---

## Tests (Write First)

File: `tests/unit/test_provider_abc.py`

### DataProvider ABC contract tests

```python
def test_provider_subclass_only_implementing_name_raises_not_implemented_for_fetch_methods():
    """A provider that only implements name() should raise NotImplementedError
    for all fetch methods. This confirms the default implementations work correctly
    and that the ABC does NOT force subclasses to implement every method."""

def test_not_implemented_error_distinguishable_from_none_return():
    """NotImplementedError (provider doesn't support this data type) is semantically
    different from returning None (provider tried but found no data). The registry
    relies on this distinction to skip unsupported providers without counting a failure."""

def test_provider_returning_none_indicates_no_data_not_error():
    """A provider that implements fetch_macro_indicator but returns None indicates
    it tried and found nothing. This is NOT a failure for circuit breaker purposes."""
```

### ConfigurationError tests

```python
def test_configuration_error_raised_on_missing_required_env_var():
    """Providers must raise ConfigurationError at __init__ time when required
    configuration (API keys, user-agent strings) is missing. Not on first fetch."""

def test_configuration_error_is_catchable_separately_from_runtime_errors():
    """ConfigurationError must be a distinct exception type so the registry can
    catch it during provider registration and exclude the provider gracefully."""
```

### Alpha Vantage adapter tests

```python
def test_av_provider_name_returns_alpha_vantage():
    """AVProvider.name() returns 'alpha_vantage'."""

def test_av_provider_fetch_macro_indicator_delegates_to_client():
    """AVProvider.fetch_macro_indicator('TREASURY_YIELD_10Y') delegates to
    AlphaVantageClient.fetch_economic_indicator with the correct AV function name."""

def test_av_provider_fetch_insider_transactions_delegates_to_client():
    """AVProvider.fetch_insider_transactions('AAPL') delegates to
    AlphaVantageClient.fetch_insider_transactions."""

def test_av_provider_fetch_fundamentals_delegates_to_client():
    """AVProvider.fetch_fundamentals('AAPL') delegates to
    AlphaVantageClient.fetch_company_overview."""

def test_av_provider_preserves_existing_rate_limiting():
    """The AV adapter must not bypass or duplicate the rate limiting already
    built into AlphaVantageClient. Verify the underlying client's
    _wait_for_rate_limit is still called."""

def test_av_provider_initializes_with_existing_client_instance():
    """AVProvider accepts an existing AlphaVantageClient instance (dependency
    injection) rather than creating its own. This preserves shared rate limit state."""
```

---

## Implementation Details

### Package Structure

Create the `src/quantstack/data/providers/` package:

```
src/quantstack/data/providers/
    __init__.py          # Exports DataProvider, ConfigurationError
    base.py              # DataProvider ABC + ConfigurationError
    alpha_vantage.py     # AVProvider wrapping existing fetcher.py
```

The `fred.py`, `edgar.py`, and `registry.py` files are created in their own sections.

### File: `src/quantstack/data/providers/__init__.py`

Export the public API for the package:

```python
from quantstack.data.providers.base import ConfigurationError, DataProvider
from quantstack.data.providers.alpha_vantage import AVProvider

__all__ = ["DataProvider", "ConfigurationError", "AVProvider"]
```

### File: `src/quantstack/data/providers/base.py`

Define the abstract base class and the custom exception.

**DataProvider ABC design:**

- `name()` is the only `@abstractmethod`. Every provider must identify itself.
- All `fetch_*` methods have default implementations that raise `NotImplementedError`. This allows each provider to implement only the data types it supports (e.g., FRED only implements `fetch_macro_indicator`).
- Return value semantics are critical for the registry's routing logic:
  - `NotImplementedError` raised: provider doesn't support this data type. Registry skips to next provider. Does NOT count as a failure.
  - `None` returned: provider supports this type, tried, but found no data for this symbol. Counts as "no data", not a failure.
  - Empty `pd.DataFrame` returned: provider returned successfully but with no rows.
  - Populated `pd.DataFrame` returned: success.

**Method signatures for the ABC:**

```python
class DataProvider(ABC):
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider (e.g., 'alpha_vantage', 'fred')."""

    def fetch_ohlcv_daily(self, symbol: str) -> pd.DataFrame | None:
        """Fetch daily OHLCV bars for symbol."""
        raise NotImplementedError

    def fetch_ohlcv_intraday(self, symbol: str, interval: str = "5min") -> pd.DataFrame | None:
        """Fetch intraday OHLCV bars for symbol at given interval."""
        raise NotImplementedError

    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
        """Fetch macro indicator time series. Returns (date, value) DataFrame."""
        raise NotImplementedError

    def fetch_fundamentals(self, symbol: str) -> dict | None:
        """Fetch company fundamentals (overview, financial ratios)."""
        raise NotImplementedError

    def fetch_insider_transactions(self, symbol: str) -> pd.DataFrame | None:
        """Fetch insider transactions. Returns DataFrame matching insider_trades schema."""
        raise NotImplementedError

    def fetch_institutional_holdings(self, symbol: str) -> pd.DataFrame | None:
        """Fetch institutional holdings. Returns DataFrame matching institutional_ownership schema."""
        raise NotImplementedError

    def fetch_earnings_history(self, symbol: str) -> pd.DataFrame | None:
        """Fetch earnings history (reported EPS, estimates, surprises)."""
        raise NotImplementedError

    def fetch_options_chain(self, symbol: str, date: str) -> pd.DataFrame | None:
        """Fetch options chain for symbol on given date."""
        raise NotImplementedError

    def fetch_sec_filings(self, symbol: str, form_types: list[str] | None = None) -> pd.DataFrame | None:
        """Fetch SEC filing metadata. Optionally filter by form types."""
        raise NotImplementedError

    def fetch_news_sentiment(self, symbol: str) -> pd.DataFrame | None:
        """Fetch news sentiment data for symbol."""
        raise NotImplementedError
```

**ConfigurationError:**

```python
class ConfigurationError(Exception):
    """Raised when a provider cannot initialize due to missing configuration.

    The registry catches this during provider registration and excludes the
    provider with a warning log, rather than crashing the application.
    """
```

### File: `src/quantstack/data/providers/alpha_vantage.py`

A thin adapter wrapping `AlphaVantageClient` behind the `DataProvider` interface.

**Key design decisions:**

1. **Dependency injection:** `AVProvider.__init__` accepts an optional `AlphaVantageClient` instance. If none is provided, it creates one using the default settings. This allows the acquisition pipeline to pass its existing client instance, preserving shared rate limit state across all calls.

2. **Delegation, not reimplementation:** Each `fetch_*` method delegates directly to the corresponding `AlphaVantageClient` method. No data transformation happens in the adapter — the AV client already returns data in the formats the system expects.

3. **Indicator name mapping:** `fetch_macro_indicator` needs to map QuantStack indicator names (e.g., `TREASURY_YIELD_10Y`) back to AV function parameters. Store the mapping as a class-level dict. Example mappings:
   - `TREASURY_YIELD_10Y` -> `fetch_economic_indicator("TREASURY_YIELD", maturity="10year")`
   - `REAL_GDP` -> `fetch_economic_indicator("REAL_GDP")`
   - `CPI` -> `fetch_economic_indicator("CPI")`
   - `UNEMPLOYMENT` -> `fetch_economic_indicator("UNEMPLOYMENT")`
   - `FED_FUNDS_RATE` -> `fetch_economic_indicator("FEDERAL_FUNDS_RATE")`

4. **Error handling:** If the underlying AV client raises (rate limit, network error, API error), the adapter lets it propagate. The registry (section-09) handles retry/fallback logic. The adapter should NOT catch or suppress exceptions.

**Stub structure:**

```python
class AVProvider(DataProvider):
    """Alpha Vantage data provider.

    Thin adapter wrapping AlphaVantageClient behind the DataProvider ABC.
    All rate limiting and retry logic remains in the underlying client.
    """

    # QuantStack indicator name -> AV fetch parameters
    MACRO_INDICATOR_MAP: dict[str, dict] = { ... }

    def __init__(self, client: AlphaVantageClient | None = None):
        """Initialize with an existing client or create a new one.

        Args:
            client: Existing AlphaVantageClient instance. If None, creates
                    one using default settings. Pass an existing instance
                    to share rate limit state.
        """

    def name(self) -> str:
        return "alpha_vantage"

    def fetch_ohlcv_daily(self, symbol: str) -> pd.DataFrame | None:
        """Delegates to AlphaVantageClient.fetch_daily."""

    def fetch_ohlcv_intraday(self, symbol: str, interval: str = "5min") -> pd.DataFrame | None:
        """Delegates to AlphaVantageClient.fetch_intraday."""

    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
        """Maps indicator name to AV parameters, delegates to fetch_economic_indicator."""

    def fetch_fundamentals(self, symbol: str) -> dict | None:
        """Delegates to AlphaVantageClient.fetch_company_overview."""

    def fetch_insider_transactions(self, symbol: str) -> pd.DataFrame | None:
        """Delegates to AlphaVantageClient.fetch_insider_transactions."""

    def fetch_institutional_holdings(self, symbol: str) -> pd.DataFrame | None:
        """Delegates to AlphaVantageClient.fetch_institutional_holdings."""

    def fetch_earnings_history(self, symbol: str) -> pd.DataFrame | None:
        """Delegates to AlphaVantageClient.fetch_earnings_history."""

    def fetch_options_chain(self, symbol: str, date: str) -> pd.DataFrame | None:
        """Delegates to AlphaVantageClient.fetch_realtime_options."""

    def fetch_news_sentiment(self, symbol: str) -> pd.DataFrame | None:
        """Delegates to AlphaVantageClient.fetch_news_sentiment."""
```

---

## Verification Checklist

After implementation, verify:

1. `from quantstack.data.providers import DataProvider, ConfigurationError, AVProvider` works
2. A minimal subclass implementing only `name()` raises `NotImplementedError` for all fetch methods
3. `AVProvider()` initializes without error when `ALPHA_VANTAGE_API_KEY` is set
4. `AVProvider(client=existing_client)` shares the client's rate limit state
5. All tests in `tests/unit/test_provider_abc.py` pass
6. No existing tests break (the adapter is additive; nothing calls it yet)

## What This Section Does NOT Do

- Does not create `fred.py` or `edgar.py` (sections 07 and 08)
- Does not create `registry.py` (section 09)
- Does not modify `acquisition_pipeline.py` or `scheduled_refresh.py` (section 10)
- Does not add `data_provider_failures` table DDL (section 09)
- Does not add `FRED_API_KEY` or `EDGAR_USER_AGENT` to `.env.example` (sections 07 and 08)
