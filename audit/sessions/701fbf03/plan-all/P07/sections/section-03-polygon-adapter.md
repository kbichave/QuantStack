# Section 03: Polygon.io Adapter

## Objective

Implement a Polygon.io data provider for OHLCV (daily and intraday) as the first fallback after Alpha Vantage. Includes rate limiting, error handling, and integration with the existing `DataProvider` ABC.

## Dependencies

None — can be implemented in parallel with sections 01, 04, 05, 08.

## Files to Create/Modify

### New Files

| File | Description |
|------|-------------|
| `src/quantstack/data/providers/polygon.py` | Polygon.io REST adapter implementing `DataProvider` ABC. Supports `fetch_ohlcv_daily` and `fetch_ohlcv_intraday` |

### Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/data/providers/__init__.py` | Add `PolygonProvider` to imports and `__all__` |
| `src/quantstack/data/providers/registry.py` | Add `PolygonProvider` to `build_registry()` factory and update `_ROUTING_TABLE` to include `"polygon"` as fallback for OHLCV |

## Implementation Details

### Step 1: Polygon Provider Class

```python
# src/quantstack/data/providers/polygon.py

class PolygonProvider(DataProvider):
    """Polygon.io REST API adapter for OHLCV data.
    
    Free tier: 5 API calls/minute.
    Paid tier ($29/mo): unlimited calls.
    Data: daily + intraday OHLCV, 15+ years history.
    Auth: POLYGON_API_KEY env var.
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self):
        self._api_key = os.environ.get("POLYGON_API_KEY")
        if not self._api_key:
            raise ConfigurationError("POLYGON_API_KEY not set")
        self._rate_limiter = ...  # See Step 2
    
    def name(self) -> str:
        return "polygon"
    
    def fetch_ohlcv_daily(self, symbol: str) -> pd.DataFrame | None:
        """GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}"""
        ...
    
    def fetch_ohlcv_intraday(self, symbol: str, interval: str = "5min") -> pd.DataFrame | None:
        """GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}"""
        ...
```

### Step 2: Rate Limiting

Use a simple token-bucket or sliding-window limiter:

```python
import time
import threading

class _RateLimiter:
    """Sliding window rate limiter: max N calls per minute."""
    
    def __init__(self, max_per_minute: int = 5):
        self._max = max_per_minute
        self._timestamps: list[float] = []
        self._lock = threading.Lock()
    
    def wait(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            now = time.monotonic()
            # Purge timestamps older than 60s
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self._max:
                sleep_time = 60 - (now - self._timestamps[0])
                time.sleep(max(0, sleep_time))
            self._timestamps.append(time.monotonic())
```

Default to 5/min (free tier). Override via `POLYGON_RATE_LIMIT` env var for paid tier.

### Step 3: Response Parsing

Polygon returns bars in this format:
```json
{
  "results": [
    {"t": 1672531200000, "o": 130.28, "h": 130.90, "l": 128.12, "c": 129.93, "v": 112117471}
  ]
}
```

Map to the standard DataFrame columns: `timestamp`, `open`, `high`, `low`, `close`, `volume` — matching Alpha Vantage's output schema so downstream code is unaware of the provider.

### Step 4: Error Handling

- HTTP 429 (rate limited): log warning, respect `Retry-After` header, raise to let registry try next provider
- HTTP 403 (auth error): raise `ConfigurationError` — this is permanent, not transient
- HTTP 5xx: raise generic exception to trigger circuit breaker
- Empty `results` array: return `None` (no data, not an error)

### Step 5: Registry Integration

In `registry.py`:
- Add `PolygonProvider` to `build_registry()`:
  ```python
  from quantstack.data.providers.polygon import PolygonProvider
  # Add to the class list
  for cls in [AVProvider, PolygonProvider, FREDProvider, EDGARProvider]:
  ```
- Update `_ROUTING_TABLE`:
  ```python
  "ohlcv_daily": ["alpha_vantage", "polygon"],
  "ohlcv_intraday": ["alpha_vantage", "polygon"],
  ```

## Test Requirements

### TDD Tests

```python
# Test: Polygon adapter returns correct OHLCV for valid symbol
def test_polygon_fetch_ohlcv_daily(mock_polygon_api):
    provider = PolygonProvider()
    df = provider.fetch_ohlcv_daily("AAPL")
    assert df is not None
    assert set(df.columns) >= {"timestamp", "open", "high", "low", "close", "volume"}
    assert len(df) > 0

# Test: Polygon adapter respects rate limit
def test_polygon_rate_limit():
    limiter = _RateLimiter(max_per_minute=2)
    t0 = time.monotonic()
    limiter.wait()  # 1st — immediate
    limiter.wait()  # 2nd — immediate
    limiter.wait()  # 3rd — should block
    elapsed = time.monotonic() - t0
    assert elapsed >= 50  # blocked for ~58-60s to wait for slot

# Test: Missing API key raises ConfigurationError
def test_polygon_missing_api_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        PolygonProvider()

# Test: HTTP 429 raises exception (not silently swallowed)
def test_polygon_rate_limit_response(mock_polygon_api_429):
    provider = PolygonProvider()
    with pytest.raises(Exception):
        provider.fetch_ohlcv_daily("AAPL")
```

### Mock Strategy

Use `responses` or `pytest-httpserver` to mock Polygon REST endpoints. Do NOT make real API calls in unit tests.

## Acceptance Criteria

1. `PolygonProvider` implements `DataProvider` ABC with `fetch_ohlcv_daily` and `fetch_ohlcv_intraday`
2. Rate limiter enforces 5 calls/min by default (configurable via env var)
3. Output DataFrame schema matches Alpha Vantage's output (column names, dtypes)
4. Missing `POLYGON_API_KEY` raises `ConfigurationError` (caught by registry, not a crash)
5. Provider is registered in `build_registry()` and appears in `_ROUTING_TABLE` as OHLCV fallback
6. All error codes are handled appropriately (429, 403, 5xx, empty results)
