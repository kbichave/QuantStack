# Section 07: FRED Provider

## Overview

Implement a FRED (Federal Reserve Economic Data) provider that fetches macroeconomic indicators via the `fredapi` library and exposes them through the DataProvider ABC defined in section-06. This gives QuantStack a second source for macro data, reducing single-provider dependency on Alpha Vantage.

**Dependency:** section-06-provider-abc must be complete (provides `DataProvider` ABC, `ConfigurationError`, and the `src/quantstack/data/providers/` package).

**Blocks:** section-10-pipeline-integration (which wires the provider registry into the acquisition pipeline).

---

## Tests (Write First)

All tests go in `tests/unit/test_fred_provider.py`.

```python
# --- Initialization ---
# Test: FREDProvider raises ConfigurationError if FRED_API_KEY is not set in the environment
# Test: FREDProvider initializes successfully when FRED_API_KEY is present
# Test: FREDProvider.name() returns "fred"

# --- fetch_macro_indicator ---
# Test: fetch_macro_indicator("DGS10") returns a DataFrame with (date, value) columns
# Test: fetch_macro_indicator normalizes FRED series IDs to QuantStack indicator names
#       (e.g., passing "TREASURY_YIELD_10Y" resolves to FRED series "DGS10" internally)
# Test: fetch_macro_indicator returns None (not an exception) when the series has no recent data
# Test: fetch_macro_indicator raises NotImplementedError for an unknown/unmapped indicator name

# --- Rate limiting ---
# Test: FRED rate limiting respects 120 req/min (verify throttle mechanism exists)

# --- Unsupported methods ---
# Test: fetch_ohlcv_daily raises NotImplementedError (FRED does not provide OHLCV)
# Test: fetch_insider_transactions raises NotImplementedError
# Test: fetch_fundamentals raises NotImplementedError
# Test: fetch_options_chain raises NotImplementedError
```

Test strategy: mock `fredapi.Fred` so tests do not require a live API key or network access. Verify that the provider translates between QuantStack indicator names and FRED series IDs, and that the returned DataFrame matches the `(date, value)` schema that the macro collector and cross-asset collector expect.

---

## Implementation Details

### File to Create

`src/quantstack/data/providers/fred.py`

### Class: FREDProvider

Subclass of `DataProvider` (from `providers/base.py`). Only implements `fetch_macro_indicator`; all other fetch methods inherit the default `NotImplementedError` from the ABC.

**Constructor behavior:**

- Read `FRED_API_KEY` from `os.environ`.
- If the key is missing or empty, raise `ConfigurationError("FRED_API_KEY environment variable is required")`.
- Instantiate `fredapi.Fred(api_key=key)` and store it as `self._client`.

**`name()` method:** Return the string `"fred"`.

### Series Mapping

The provider maintains a bidirectional mapping between QuantStack indicator names (used by the macro collector) and FRED series IDs. This mapping is a module-level constant dict:

| FRED Series ID | QuantStack Indicator Name | Update Frequency |
|----------------|--------------------------|------------------|
| `DGS10` | `TREASURY_YIELD_10Y` | Daily |
| `DGS2` | `TREASURY_YIELD_2Y` | Daily |
| `T10Y2Y` | `YIELD_CURVE_SPREAD` | Daily |
| `FEDFUNDS` | `FED_FUNDS_RATE` | Monthly |
| `CPIAUCSL` | `CPI` | Monthly |
| `UNRATE` | `UNEMPLOYMENT` | Monthly |
| `GDP` | `REAL_GDP` | Quarterly |
| `BAMLH0A0HYM2` | `HIGH_YIELD_OAS` | Daily |
| `ICSA` | `INITIAL_CLAIMS` | Weekly |

Store this as two dicts (or one dict with reverse lookup): `INDICATOR_TO_FRED` maps QuantStack names to FRED series IDs, and `FRED_TO_INDICATOR` maps the reverse direction.

### `fetch_macro_indicator` Method

Signature:

```python
def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
    """Fetch a macro indicator from FRED.
    
    Args:
        indicator: Either a QuantStack indicator name (e.g., "TREASURY_YIELD_10Y")
                   or a raw FRED series ID (e.g., "DGS10").
    
    Returns:
        DataFrame with columns (date, value) matching the macro_indicators schema,
        or None if no data is available.
    
    Raises:
        NotImplementedError if the indicator is not in the series mapping.
    """
```

Implementation notes:

1. Resolve the indicator to a FRED series ID using `INDICATOR_TO_FRED`. If the indicator is already a valid FRED series ID (exists in `FRED_TO_INDICATOR`), use it directly. If neither mapping contains the indicator, raise `NotImplementedError` so the registry treats it as "this provider doesn't support this indicator" (not a failure).
2. Call `self._client.get_series(series_id)` which returns a pandas Series indexed by date.
3. If the result is empty or None, return None.
4. Convert to a DataFrame with columns `date` (datetime) and `value` (float). This matches the schema that `macro_indicators` uses and that the macro collector and cross-asset collector expect.
5. Wrap the `fredapi` call in a try/except to catch network errors and API errors. On failure, log the error with context (series ID, indicator name, error message) and re-raise so the registry can track the failure and attempt fallback to Alpha Vantage.

### Rate Limiting

The `fredapi` library does not enforce rate limits internally. FRED allows 120 requests per minute. Since the daily macro refresh needs approximately 9-15 calls (one per mapped indicator), rate limiting is unlikely to be an issue. However, add a simple throttle mechanism (e.g., `time.sleep` guard or a token bucket) to prevent accidental bursts if the provider is called in a tight loop. A lightweight approach: track `_last_request_at` and enforce a minimum 0.5-second gap between requests (conservative, allows 120/min while preventing bursts).

### Error Handling

- `ConfigurationError` on missing API key (raised at init, caught by registry during provider registration).
- `NotImplementedError` on unsupported indicators (caught by registry, does not count as failure).
- Network/API errors from `fredapi` propagate as exceptions (caught by registry, counted as failure, triggers fallback).

### Data Normalization

The DataFrame returned by `fetch_macro_indicator` must match the format that the existing AV macro data path produces. Specifically:

- Column `date`: datetime type, timezone-naive
- Column `value`: float type
- Sorted by date ascending
- No NaN values in the `value` column (drop rows with NaN, which FRED uses for missing observations)

This ensures that downstream code in the macro collector and cross-asset collector works identically whether the data came from AV or FRED.

---

## Environment Variable

Add `FRED_API_KEY` to `.env.example` with a comment:

```bash
FRED_API_KEY=           # Required for FRED macro data provider (free, register at fred.stlouisfed.org)
```

### File to Modify

`.env.example` — add the `FRED_API_KEY` line.

---

## Dependency: `fredapi` Library

Add `fredapi` to `pyproject.toml` dependencies. Pin to a specific version (check PyPI for the latest stable release at implementation time). The library is a thin wrapper over the FRED API with no heavy transitive dependencies.

---

## Integration Notes

- The FRED provider is registered in the `ProviderRegistry` (section-09) as a fallback for `macro_indicator` data type, with Alpha Vantage as primary. This strangler-fig approach means FRED only gets called when AV fails or is circuit-broken.
- After FRED proves stable in production for 2+ weeks, the routing table can be updated to make FRED the primary for macro indicators (FRED is the authoritative source for this data; AV is a reseller).
- The provider does not write to the database. The registry caller (acquisition pipeline, section-10) handles DB writes. The provider only fetches and normalizes data.
- The `NotImplementedError` return semantics are important: the registry must distinguish "provider doesn't support this" from "provider tried and failed." FRED raising `NotImplementedError` for `fetch_ohlcv_daily` means the registry silently skips FRED for OHLCV requests without counting it as a failure.
