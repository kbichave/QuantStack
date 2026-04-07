# Section 09: Provider Registry

## Overview

Implement `ProviderRegistry` in `src/quantstack/data/providers/registry.py` — the orchestration layer that routes data requests to the best available provider, handles failover, tracks failures in the database, and enforces circuit-breaker logic to avoid repeatedly hitting known-broken providers.

**Depends on:** section-06-provider-abc (the `DataProvider` ABC and `ConfigurationError` must exist)

**Blocks:** section-10-pipeline-integration (the pipeline rewiring depends on a working registry)

---

## Background

QuantStack currently calls `AlphaVantageClient` directly from the acquisition pipeline. With FRED and EDGAR providers added (sections 07 and 08), the system needs a routing layer that:

1. Maps each data type to an ordered list of providers (primary, then fallbacks)
2. Tries the primary provider first, falls back on failure
3. Tracks consecutive failures per (provider, data_type) pair in the database
4. Implements a circuit breaker so known-broken providers are skipped for a cooldown period
5. Fires alerts when a provider accumulates 3+ consecutive failures
6. Logs structured observability fields on every fetch call

The registry is the single entry point that the acquisition pipeline and scheduled refresh will call (wired in section-10).

---

## Tests First

Write these in `tests/unit/test_provider_registry.py`.

### Routing and Fallback

```python
# Test: registry routes macro_indicator to AV (primary), falls back to FRED on AV failure
# Test: registry routes insider_transactions to EDGAR (primary), falls back to AV on EDGAR failure
# Test: NotImplementedError from a provider does NOT count as a failure — registry silently skips to next provider
```

### Failure Tracking

```python
# Test: registry increments consecutive_failures on provider failure
# Test: registry resets consecutive_failures to 0 on success
# Test: registry inserts alert into system_events after 3 consecutive failures
```

### Circuit Breaker

```python
# Test: after 3 failures within 10 minutes, registry skips primary and goes directly to fallback
# Test: after 10 minutes, registry tries primary again (circuit breaker resets)
```

### Observability

```python
# Test: registry logs structured fields: provider_name, data_type, symbol, latency_ms, success, fallback_used
```

### Provider Initialization

```python
# Test: registry excludes misconfigured providers with warning log (catches ConfigurationError at registration time)
# Test: AVProvider initializes successfully with existing AlphaVantageClient
```

### Failure Persistence

```python
# Test: data_provider_failures table created with correct schema
# Test: consecutive failures tracked correctly across multiple calls
# Test: alert inserted on 3rd consecutive failure
# Test: success resets failure counter
```

All tests should use mocked providers (subclasses of `DataProvider` that return canned data or raise on demand). No real API calls.

---

## Best-Source Routing Table

The registry holds this mapping as configuration (not hardcoded at each call site):

| Data Type | Primary | Fallback(s) |
|-----------|---------|-------------|
| `ohlcv_daily` | AlphaVantage | Alpaca |
| `ohlcv_intraday` | AlphaVantage | Alpaca |
| `macro_indicator` | AlphaVantage | FRED |
| `fundamentals` | AlphaVantage | EDGAR |
| `earnings_history` | AlphaVantage | EDGAR |
| `insider_transactions` | EDGAR | AlphaVantage |
| `institutional_holdings` | EDGAR | AlphaVantage |
| `options_chain` | AlphaVantage | (none) |
| `news_sentiment` | AlphaVantage | (none) |
| `sec_filings` | EDGAR | (none) |
| `commodities` | AlphaVantage | FRED (partial) |

**Strangler fig note:** Macro indicators start with AV as primary and FRED as fallback. After FRED proves stable over 2+ weeks of production operation, swap FRED to primary via configuration change (no code change needed).

---

## Implementation Details

### File: `src/quantstack/data/providers/registry.py`

The `ProviderRegistry` class has one primary public method — `fetch()` — and internal methods for failure tracking and circuit breaker logic.

**Class sketch (signatures and docstrings only):**

```python
class ProviderRegistry:
    """Routes data requests to the best available provider with failover.
    
    Holds an ordered provider chain per data type. Tries providers in order,
    skipping any whose circuit breaker is open. Tracks failures in the
    data_provider_failures table and fires alerts after 3 consecutive failures.
    """

    def __init__(self, providers: list[DataProvider]):
        """Initialize with available providers.
        
        Catches ConfigurationError from any provider and excludes it with
        a warning log. Builds the routing table from the successfully
        initialized providers.
        """

    def fetch(self, data_type: str, symbol: str, **kwargs) -> pd.DataFrame | dict | None:
        """Route a data request to the best available provider.
        
        Algorithm:
        1. Look up the provider chain for data_type
        2. For each provider in chain:
           a. Check circuit breaker — if open (>=3 failures in last 10 min), skip
           b. Call the corresponding fetch method on the provider
           c. If NotImplementedError: skip silently (not a failure)
           d. If success: reset consecutive_failures, log, return result
           e. If failure: increment consecutive_failures, log, continue to next
        3. If all providers exhausted: log error, return None
        
        Every call logs structured fields:
          provider_name, data_type, symbol, latency_ms, success, fallback_used,
          circuit_breaker_tripped
        """

    def _check_circuit_breaker(self, provider_name: str, data_type: str) -> bool:
        """Return True if the circuit breaker is open (should skip this provider).
        
        Open when consecutive_failures >= 3 AND last_failure_at > now() - 10 minutes.
        """

    def _record_failure(self, provider_name: str, data_type: str, error: str) -> None:
        """Increment consecutive_failures in data_provider_failures table.
        
        If consecutive_failures reaches 3, insert an alert into system_events.
        """

    def _record_success(self, provider_name: str, data_type: str) -> None:
        """Reset consecutive_failures to 0 in data_provider_failures table."""
```

### Data Type to Method Mapping

The registry needs to map `data_type` strings to the correct `DataProvider` method name:

| `data_type` string | `DataProvider` method |
|--------------------|-----------------------|
| `ohlcv_daily` | `fetch_ohlcv_daily()` |
| `ohlcv_intraday` | `fetch_ohlcv_daily()` (or a future intraday method) |
| `macro_indicator` | `fetch_macro_indicator()` |
| `fundamentals` | `fetch_fundamentals()` |
| `earnings_history` | `fetch_earnings_history()` |
| `insider_transactions` | `fetch_insider_transactions()` |
| `institutional_holdings` | `fetch_institutional_holdings()` |
| `options_chain` | `fetch_options_chain()` |
| `sec_filings` | `fetch_sec_filings()` |

Use `getattr(provider, method_name)` to dispatch. If the method raises `NotImplementedError`, that means the provider does not support this data type — skip to next provider without counting it as a failure.

### Circuit Breaker Logic

The circuit breaker is per (provider, data_type) pair:

- **Closed (normal):** Provider is tried on every request. Failures increment the counter.
- **Open (tripped):** Provider is skipped. Condition: `consecutive_failures >= 3 AND last_failure_at > now() - timedelta(minutes=10)`.
- **Half-open (auto-reset):** After 10 minutes, the circuit breaker resets. The next request tries the provider again. If it fails, the breaker reopens immediately.

The failure state lives in the `data_provider_failures` database table so it survives process restarts. For performance, the registry may cache the failure state in memory and refresh from DB periodically (every 60 seconds), but the DB is the source of truth.

### Failure Tracking Table

This table must already exist in `_schema.py` (from section-06 or added here):

```sql
CREATE TABLE IF NOT EXISTS data_provider_failures (
    provider VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    consecutive_failures INTEGER DEFAULT 0,
    last_failure_at TIMESTAMP,
    last_error TEXT,
    PRIMARY KEY (provider, data_type)
);
```

**On failure:**
```sql
INSERT INTO data_provider_failures (provider, data_type, consecutive_failures, last_failure_at, last_error)
VALUES (:provider, :data_type, 1, NOW(), :error)
ON CONFLICT (provider, data_type)
DO UPDATE SET
    consecutive_failures = data_provider_failures.consecutive_failures + 1,
    last_failure_at = NOW(),
    last_error = :error;
```

**On success:**
```sql
UPDATE data_provider_failures
SET consecutive_failures = 0
WHERE provider = :provider AND data_type = :data_type;
```

**Alert on 3rd failure:**
```sql
INSERT INTO system_events (event_type, symbol, severity, details, created_at)
VALUES ('PROVIDER_FAILURE', :provider, 'warning',
        :json_details, NOW());
```

The `json_details` should include: `provider_name`, `data_type`, `consecutive_failures`, `last_error`.

### Structured Observability Logging

Every `fetch()` call logs a structured message with these fields:

```python
logger.info(
    "provider_fetch",
    provider_name=provider.name(),
    data_type=data_type,
    symbol=symbol,
    latency_ms=round(elapsed * 1000, 1),
    success=True,  # or False
    fallback_used=fallback_used,  # True if not the first provider in chain
    circuit_breaker_tripped=cb_tripped,  # True if any provider was skipped
)
```

This enables querying provider health and latency distribution from logs.

### Provider Registration and Initialization

When the registry is constructed, it receives a list of `DataProvider` instances. Construction should be wrapped to handle `ConfigurationError`:

```python
def build_registry() -> ProviderRegistry:
    """Factory that instantiates all known providers and builds the registry.
    
    Providers that raise ConfigurationError (e.g., missing FRED_API_KEY)
    are excluded with a warning log. The system continues with whatever
    providers are available.
    """
```

This means the system degrades gracefully — if FRED is not configured, all macro requests still go to AV (no fallback available, but no crash either).

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/quantstack/data/providers/registry.py` | `ProviderRegistry` class |
| `tests/unit/test_provider_registry.py` | All tests listed above |

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/data/providers/__init__.py` | Export `ProviderRegistry` and `build_registry` |
| `src/quantstack/data/_schema.py` | Add `data_provider_failures` table DDL (if not already added in section-06) |

---

## Key Design Decisions

1. **DB-backed failure tracking (not in-memory only):** Failure state must survive process restarts. A provider that was failing before a restart should still have its circuit breaker open when the process comes back up. The DB is the source of truth; in-memory caching is an optimization.

2. **NotImplementedError is not a failure:** A provider that does not support a data type is fundamentally different from a provider that tried and failed. The first is expected and permanent; the second is transient. Only transient failures increment the counter.

3. **10-minute circuit breaker cooldown:** Long enough to avoid hammering a broken provider during a transient outage. Short enough that recovery is detected quickly. This is a conservative starting point — can be tuned based on observed provider behavior.

4. **Strangler fig for FRED:** Starting FRED as fallback (not primary) for macro indicators is deliberate risk management. A new, untested dependency should not be the primary source on day one. Promote to primary after 2+ weeks of stable production operation.

5. **Single `fetch()` entry point:** All call sites use the same method. The registry decides which provider to call based on the data type. This keeps the call sites clean and centralizes routing decisions.
