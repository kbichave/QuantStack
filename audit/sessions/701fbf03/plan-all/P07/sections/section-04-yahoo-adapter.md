# Section 04: Yahoo Finance Adapter

## Objective

Implement a Yahoo Finance data provider as a last-resort OHLCV fallback using the `yfinance` library. Includes aggressive caching to mitigate Yahoo's unreliability.

## Dependencies

None — can be implemented in parallel with sections 01, 03, 05, 08.

## Files to Create/Modify

### New Files

| File | Description |
|------|-------------|
| `src/quantstack/data/providers/yahoo.py` | yfinance wrapper implementing `DataProvider` ABC. Supports `fetch_ohlcv_daily` and limited `fetch_ohlcv_intraday`. Self-imposed 1 req/sec rate limit. Aggressive response caching. |

### Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/data/providers/__init__.py` | Add `YahooProvider` to imports and `__all__` |
| `src/quantstack/data/providers/registry.py` | Add `YahooProvider` to `build_registry()` and update `_ROUTING_TABLE` to include `"yahoo"` as tertiary OHLCV fallback |

## Implementation Details

### Step 1: Yahoo Provider Class

```python
# src/quantstack/data/providers/yahoo.py

class YahooProvider(DataProvider):
    """Yahoo Finance adapter via yfinance — last-resort fallback only.
    
    No auth required. Self-imposed rate limit: 1 request/second.
    Data limitations:
      - Daily OHLCV: full history available
      - 1-min intraday: only 7 days of history
      - 1-hour intraday: only 60 days of history
    
    Unreliable — can break without notice. Cache aggressively.
    """
    
    def __init__(self):
        try:
            import yfinance  # noqa: F401
        except ImportError:
            raise ConfigurationError("yfinance not installed")
        self._cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._lock = threading.Lock()
        self._last_request_time = 0.0
    
    def name(self) -> str:
        return "yahoo"
    
    def fetch_ohlcv_daily(self, symbol: str) -> pd.DataFrame | None:
        ...
    
    def fetch_ohlcv_intraday(self, symbol: str, interval: str = "5min") -> pd.DataFrame | None:
        ...
```

### Step 2: Caching Strategy

Two TTLs based on data type:
- **Daily OHLCV:** 24-hour cache TTL (data doesn't change intraday)
- **Intraday OHLCV:** 1-hour cache TTL

Cache key: `f"{symbol}:{interval}"`. In-memory dict with TTL check:

```python
def _get_cached(self, key: str, ttl_seconds: int) -> pd.DataFrame | None:
    with self._lock:
        if key in self._cache:
            df, cached_at = self._cache[key]
            if time.monotonic() - cached_at < ttl_seconds:
                return df
            del self._cache[key]
    return None

def _set_cached(self, key: str, df: pd.DataFrame) -> None:
    with self._lock:
        self._cache[key] = (df, time.monotonic())
```

### Step 3: Rate Limiting

Self-imposed 1 request/second to avoid being blocked:

```python
def _rate_limit(self) -> None:
    elapsed = time.monotonic() - self._last_request_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    self._last_request_time = time.monotonic()
```

### Step 4: Output Normalization

`yfinance` returns DataFrames with columns `Open, High, Low, Close, Volume, Adj Close`. Normalize to match the standard schema: `timestamp, open, high, low, close, volume`. Drop `Adj Close` — QuantStack handles adjustments separately.

### Step 5: Registry Integration

In `registry.py`:
- Add to `build_registry()` class list (after Polygon, before FRED)
- Update `_ROUTING_TABLE`:
  ```python
  "ohlcv_daily": ["alpha_vantage", "polygon", "yahoo"],
  "ohlcv_intraday": ["alpha_vantage", "polygon", "yahoo"],
  ```

Yahoo is always last in the chain — it is the provider of last resort.

### Important Constraints

- **yfinance is an optional dependency.** If not installed, raise `ConfigurationError` so the registry excludes it gracefully.
- **No auth required.** No env var check needed, but the import check is the configuration gate.
- **Yahoo can break silently.** Wrap all yfinance calls in try/except, log the error, and return `None`.

## Test Requirements

### TDD Tests

```python
# Test: Yahoo adapter returns correct daily OHLCV
def test_yahoo_fetch_ohlcv_daily(mock_yfinance):
    provider = YahooProvider()
    df = provider.fetch_ohlcv_daily("AAPL")
    assert df is not None
    assert set(df.columns) >= {"timestamp", "open", "high", "low", "close", "volume"}

# Test: Yahoo adapter caches responses for 24h
def test_yahoo_cache_daily(mock_yfinance):
    provider = YahooProvider()
    df1 = provider.fetch_ohlcv_daily("AAPL")
    df2 = provider.fetch_ohlcv_daily("AAPL")
    # yfinance download should only be called once
    assert mock_yfinance.download.call_count == 1
    assert df1 is df2 or df1.equals(df2)

# Test: Cache expires after TTL
def test_yahoo_cache_expiry(mock_yfinance, monkeypatch):
    provider = YahooProvider()
    provider.fetch_ohlcv_daily("AAPL")
    # Advance time past TTL
    # ... (use time.monotonic mock)
    provider.fetch_ohlcv_daily("AAPL")
    assert mock_yfinance.download.call_count == 2

# Test: yfinance not installed raises ConfigurationError
def test_yahoo_missing_yfinance(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ConfigurationError):
        YahooProvider()
```

### Mock Strategy

Mock `yfinance.download()` to return a synthetic DataFrame. Do NOT make real Yahoo API calls.

## Acceptance Criteria

1. `YahooProvider` implements `DataProvider` ABC with `fetch_ohlcv_daily` and `fetch_ohlcv_intraday`
2. In-memory cache with 24h TTL for daily, 1h for intraday
3. Self-imposed 1 req/sec rate limit
4. Output DataFrame schema matches standard columns (`timestamp`, `open`, `high`, `low`, `close`, `volume`)
5. Missing `yfinance` raises `ConfigurationError` (graceful exclusion from registry)
6. Registered as last-resort fallback in `_ROUTING_TABLE`
7. All yfinance errors are caught, logged, and result in `None` return
