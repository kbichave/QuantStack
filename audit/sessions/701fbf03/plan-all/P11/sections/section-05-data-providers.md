# Section 05: Data Provider Integration

## Objective

Integrate each alt-data collector's API adapter into the P07 DataProvider pattern so that all four sources benefit from the existing rate limiting, circuit breaker, and freshness tracking infrastructure. This section does NOT duplicate collector logic — it wires the API transport layer into the shared provider framework.

## Dependencies

Requires section-01 through section-04 to be complete (collectors must exist before providers can wrap them).

## Files to Create

### `src/quantstack/data/providers/quiver.py`

DataProvider implementation for Quiver Quantitative API (congressional trades).

**Implementation details:**

1. Subclass `DataProvider` from `src/quantstack/data/providers/base.py`.
2. `name()` returns `"quiver"`.
3. Add method `fetch_congressional_trades(self, symbol: str) -> pd.DataFrame | None`:
   - HTTP GET to Quiver API endpoint.
   - Parse response into DataFrame with columns: `date`, `representative`, `party`, `transaction_type`, `amount_range`, `ticker`, `committee`.
   - Apply rate limiting via shared token bucket (Quiver free tier: ~100 req/day).
4. Constructor reads `QUIVER_API_KEY` from env. Raises `ConfigurationError` if missing.
5. Circuit breaker: after 3 consecutive failures, open circuit for 5 minutes.

### `src/quantstack/data/providers/similarweb.py`

DataProvider implementation for SimilarWeb API.

**Implementation details:**

1. Subclass `DataProvider`.
2. `name()` returns `"similarweb"`.
3. Add method `fetch_web_traffic(self, domain: str, months: int = 12) -> pd.DataFrame | None`:
   - HTTP GET to SimilarWeb traffic endpoint.
   - Parse into DataFrame: `date`, `visits`, `avg_visit_duration`, `pages_per_visit`, `bounce_rate`.
   - Rate limiting: SimilarWeb tier-dependent, default 10 req/min.
4. Constructor reads `SIMILARWEB_API_KEY` from env. Raises `ConfigurationError` if missing.
5. Circuit breaker: 3 failures → 10 minute cooldown (SimilarWeb rate limits are strict).

### `src/quantstack/data/providers/thinknum.py`

DataProvider implementation for Thinknum API (job postings).

**Implementation details:**

1. Subclass `DataProvider`.
2. `name()` returns `"thinknum"`.
3. Add method `fetch_job_postings(self, symbol: str, months: int = 12) -> pd.DataFrame | None`:
   - HTTP GET to Thinknum job listings endpoint.
   - Parse into DataFrame: `date`, `title`, `location`, `department`, `posting_url`.
   - Rate limiting: 60 req/min.
4. Constructor reads `THINKNUM_API_KEY` from env. Raises `ConfigurationError` if missing.
5. Circuit breaker: 3 failures → 5 minute cooldown.

### `src/quantstack/data/providers/patentsview.py`

DataProvider implementation for USPTO PatentsView API.

**Implementation details:**

1. Subclass `DataProvider`.
2. `name()` returns `"patentsview"`.
3. Add method `fetch_patents(self, assignee: str, months: int = 24) -> pd.DataFrame | None`:
   - POST to PatentsView query endpoint with assignee filter and date range.
   - Parse into DataFrame: `patent_number`, `date`, `title`, `assignee`, `cpc_codes`, `citation_count`.
   - Rate limiting: 45 req/min (USPTO documented limit).
4. No API key required. Constructor always succeeds.
5. Circuit breaker: 3 failures → 5 minute cooldown.

## Files to Modify

### `src/quantstack/data/providers/registry.py`

Register the four new providers:

```python
from quantstack.data.providers.quiver import QuiverProvider
from quantstack.data.providers.similarweb import SimilarWebProvider
from quantstack.data.providers.thinknum import ThinknumProvider
from quantstack.data.providers.patentsview import PatentsViewProvider
```

Add to the provider initialization list, each wrapped in try/except `ConfigurationError` (consistent with existing pattern — missing API keys log a warning but don't crash).

### `src/quantstack/data/providers/base.py`

Add abstract method stubs for the new data types so the interface is discoverable:

```python
def fetch_congressional_trades(self, symbol: str) -> pd.DataFrame | None:
    raise NotImplementedError

def fetch_web_traffic(self, domain: str, months: int = 12) -> pd.DataFrame | None:
    raise NotImplementedError

def fetch_job_postings(self, symbol: str, months: int = 12) -> pd.DataFrame | None:
    raise NotImplementedError

def fetch_patents(self, assignee: str, months: int = 24) -> pd.DataFrame | None:
    raise NotImplementedError
```

## Test Requirements

### `tests/unit/data/test_alt_data_providers.py`

1. **test_quiver_provider_init** — With API key set → provider initializes.
2. **test_quiver_provider_missing_key** — Without API key → raises `ConfigurationError`.
3. **test_quiver_fetch_success** — Mock HTTP 200 → returns DataFrame with expected columns.
4. **test_quiver_circuit_breaker** — 3 failures → subsequent calls return None without HTTP call.
5. **test_similarweb_fetch_success** — Mock HTTP 200 → DataFrame with traffic columns.
6. **test_similarweb_rate_limit** — Exceeding rate → requests delayed, not dropped.
7. **test_thinknum_fetch_success** — Mock HTTP 200 → DataFrame with job posting columns.
8. **test_patentsview_no_key_needed** — Provider initializes without any env var.
9. **test_patentsview_fetch_success** — Mock HTTP 200 → DataFrame with patent columns.
10. **test_registry_graceful_degradation** — Missing API keys → providers excluded, others still registered.

## Acceptance Criteria

- [ ] All four providers subclass `DataProvider` and implement `name()`.
- [ ] Rate limiting prevents API abuse for all four sources.
- [ ] Circuit breaker opens after 3 consecutive failures; closes after cooldown.
- [ ] Missing API keys raise `ConfigurationError` (caught by registry, not application).
- [ ] PatentsView provider works without any API key.
- [ ] Registry registers available providers and logs warnings for unavailable ones.
- [ ] `base.py` has abstract stubs for all four new fetch methods.
- [ ] All 10 unit tests pass.
