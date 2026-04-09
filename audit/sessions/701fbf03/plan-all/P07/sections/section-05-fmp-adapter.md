# Section 05: Financial Modeling Prep (FMP) Adapter

## Objective

Implement a Financial Modeling Prep adapter for fundamentals data (income statement, balance sheet, cash flow, ratios) as a fallback after Alpha Vantage for the `fundamentals` and `earnings_history` data types.

## Dependencies

None — can be implemented in parallel with sections 01, 03, 04, 08.

## Files to Create/Modify

### New Files

| File | Description |
|------|-------------|
| `src/quantstack/data/providers/fmp.py` | FMP REST adapter implementing `DataProvider` ABC. Supports `fetch_fundamentals` and `fetch_earnings_history` |

### Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/data/providers/__init__.py` | Add `FMPProvider` to imports and `__all__` |
| `src/quantstack/data/providers/registry.py` | Add `FMPProvider` to `build_registry()` and update `_ROUTING_TABLE` to include `"fmp"` as fundamentals fallback |

## Implementation Details

### Step 1: FMP Provider Class

```python
# src/quantstack/data/providers/fmp.py

class FMPProvider(DataProvider):
    """Financial Modeling Prep API adapter.
    
    Pricing: $14/mo for 300 requests/day.
    Data: income statement, balance sheet, cash flow, financial ratios.
    Auth: FMP_API_KEY env var.
    Base URL: https://financialmodelingprep.com/api/v3
    """
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self):
        self._api_key = os.environ.get("FMP_API_KEY")
        if not self._api_key:
            raise ConfigurationError("FMP_API_KEY not set")
    
    def name(self) -> str:
        return "fmp"
    
    def fetch_fundamentals(self, symbol: str) -> dict | None:
        """Fetch company profile + key ratios.
        
        Combines:
          GET /profile/{symbol}
          GET /ratios/{symbol}
        into a single dict matching AV's fundamentals schema.
        """
        ...
    
    def fetch_earnings_history(self, symbol: str) -> pd.DataFrame | None:
        """Fetch historical earnings (EPS actual, estimated, surprise).
        
        GET /historical/earning_calendar/{symbol}
        """
        ...
```

### Step 2: API Endpoints

| Data | Endpoint | Notes |
|------|----------|-------|
| Company profile | `GET /profile/{symbol}?apikey=...` | Market cap, sector, description |
| Income statement | `GET /income-statement/{symbol}?period=quarter&apikey=...` | Quarterly IS |
| Balance sheet | `GET /balance-sheet-statement/{symbol}?period=quarter&apikey=...` | Quarterly BS |
| Cash flow | `GET /cash-flow-statement/{symbol}?period=quarter&apikey=...` | Quarterly CF |
| Key ratios | `GET /ratios/{symbol}?period=quarter&apikey=...` | PE, PB, ROE, etc. |
| Earnings history | `GET /historical/earning_calendar/{symbol}?apikey=...` | EPS actuals vs estimates |

### Step 3: Response Normalization

FMP field names differ from Alpha Vantage. Map to a consistent schema:

```python
_FUNDAMENTALS_MAP = {
    "marketCap": "MarketCapitalization",
    "peRatio": "PERatio",
    "pbRatio": "PriceToBookRatio",
    "returnOnEquity": "ReturnOnEquityTTM",
    # ... map all fields used by quantstack
}
```

The goal: downstream code that reads `fundamentals["PERatio"]` works regardless of whether AV or FMP provided the data.

### Step 4: Rate Limiting

FMP allows 300 requests/day on the $14 plan. Track daily usage:

```python
_DAILY_LIMIT = int(os.environ.get("FMP_DAILY_LIMIT", "300"))

class FMPProvider(DataProvider):
    def __init__(self):
        ...
        self._daily_count = 0
        self._day_start = date.today()
    
    def _check_limit(self) -> None:
        if date.today() != self._day_start:
            self._daily_count = 0
            self._day_start = date.today()
        if self._daily_count >= _DAILY_LIMIT:
            raise Exception(f"FMP daily limit ({_DAILY_LIMIT}) reached")
        self._daily_count += 1
```

### Step 5: Error Handling

- HTTP 401/403: `ConfigurationError` — bad or expired API key
- HTTP 429: daily limit hit — raise to let registry try next provider (EDGAR)
- HTTP 5xx: transient — raise for circuit breaker
- Empty response body: return `None`

### Step 6: Registry Integration

In `registry.py`:
- Add `FMPProvider` to `build_registry()` class list
- Update `_ROUTING_TABLE`:
  ```python
  "fundamentals": ["alpha_vantage", "fmp", "edgar"],
  "earnings_history": ["alpha_vantage", "fmp", "edgar"],
  ```

## Test Requirements

### TDD Tests

```python
# Test: FMP adapter returns financial statements
def test_fmp_fetch_fundamentals(mock_fmp_api):
    provider = FMPProvider()
    result = provider.fetch_fundamentals("AAPL")
    assert result is not None
    assert isinstance(result, dict)
    # Check normalized field names match AV schema
    assert "MarketCapitalization" in result or "PERatio" in result

# Test: FMP earnings history returns DataFrame
def test_fmp_fetch_earnings(mock_fmp_api):
    provider = FMPProvider()
    df = provider.fetch_earnings_history("AAPL")
    assert df is not None
    assert len(df) > 0

# Test: Missing API key raises ConfigurationError
def test_fmp_missing_api_key(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        FMPProvider()

# Test: Daily limit enforcement
def test_fmp_daily_limit():
    provider = FMPProvider()
    provider._daily_count = 300
    with pytest.raises(Exception, match="daily limit"):
        provider._check_limit()
```

### Mock Strategy

Use `responses` or `respx` to mock FMP REST endpoints. Test field mapping explicitly — verify AV-compatible field names.

## Acceptance Criteria

1. `FMPProvider` implements `DataProvider` ABC with `fetch_fundamentals` and `fetch_earnings_history`
2. Response fields are normalized to match Alpha Vantage's schema
3. Daily request limit is tracked and enforced (default 300)
4. Missing `FMP_API_KEY` raises `ConfigurationError`
5. Registered in `build_registry()` and `_ROUTING_TABLE` as fundamentals fallback (between AV and EDGAR)
6. All HTTP error codes handled appropriately
