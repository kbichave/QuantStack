# Section 08: EDGAR Provider

## Overview

Implement `EDGARProvider` as a concrete `DataProvider` (from section-06-provider-abc) that fetches SEC EDGAR data via the `edgartools` library. This provider gives QuantStack a free, independent source for insider transactions (Form 4), institutional holdings (13F), financial statements (XBRL), and SEC filing metadata -- reducing sole dependence on Alpha Vantage.

**Dependencies:** section-06-provider-abc must be completed first (provides `DataProvider` ABC and `ConfigurationError`).

**Blocks:** section-10-pipeline-integration (registry wiring) and section-11-sec-filings (acquisition phases 15/16 that consume this provider).

## Implementation Phases

The EDGAR provider ships in two milestones:

1. **MVP (priority):** CIK resolution + Form 4 insider transaction parsing. This unblocks section-11 early.
2. **Full:** XBRL financial statements + 13F institutional holdings.

## Tests (Write First)

All tests go in `tests/unit/test_edgar_provider.py`. Use `unittest.mock` to avoid real SEC network calls.

```python
# --- Initialization ---
# Test: EDGARProvider raises ConfigurationError if EDGAR_USER_AGENT env var is not set
# Test: EDGARProvider initializes successfully when EDGAR_USER_AGENT is set
# Test: EDGAR startup health check validates CIK resolution for a known ticker (e.g., "AAPL")
# Test: EDGAR provider handles ticker with no CIK mapping gracefully (returns None, does not raise)

# --- Form 4 / Insider Transactions (MVP) ---
# Test: fetch_insider_transactions("AAPL") returns a DataFrame matching insider_trades schema
# Test: EDGAR Form 4 data normalized to columns: (ticker, transaction_date, owner_name, transaction_type, shares, price_per_share)
# Test: transaction_type derived correctly — 'A' (acquired) maps to 'buy', 'D' (disposed) maps to 'sell'
# Test: fetch_insider_transactions returns None (not error) for a ticker with no Form 4 filings

# --- XBRL Financials (Full) ---
# Test: fetch_fundamentals("AAPL") returns dict matching financial_statements schema
# Test: fetch_fundamentals returns None for a ticker with no XBRL data

# --- 13F Institutional Holdings (Full) ---
# Test: fetch_institutional_holdings("AAPL") returns DataFrame matching institutional_ownership schema
# Test: fetch_institutional_holdings returns None for a ticker with no 13F filings

# --- SEC Filing Metadata ---
# Test: fetch_sec_filings("AAPL", ["10-K", "10-Q", "8-K"]) returns DataFrame with columns: accession_number, symbol, form_type, filing_date, period_of_report, primary_doc_url
# Test: fetch_sec_filings returns None for unknown ticker

# --- Rate Limiting ---
# Test: EDGAR rate limiting respects 10 req/sec (mock time or use call counter to verify throttle)

# --- ABC Contract ---
# Test: EDGARProvider.name() returns "edgar"
# Test: Methods not implemented by EDGAR (e.g., fetch_ohlcv_daily, fetch_options_chain) raise NotImplementedError
```

## File to Create

**`src/quantstack/data/providers/edgar.py`**

This file implements the `EDGARProvider` class inheriting from `DataProvider` (defined in `src/quantstack/data/providers/base.py` per section-06).

## Configuration

The EDGAR provider requires one environment variable:

```bash
EDGAR_USER_AGENT="QuantStack admin@quantstack.dev"   # SEC policy: company name + contact email
```

This must be validated at `__init__` time. If missing, raise `ConfigurationError` immediately so the provider registry (section-09) can exclude it with a warning log rather than failing silently on the first fetch call.

Add `EDGAR_USER_AGENT` to `.env.example` (if not already added by another section).

## Library Dependency

Use `edgartools` (pin to a specific version in `pyproject.toml`). This library handles:
- Ticker-to-CIK resolution internally
- XBRL Company Facts API
- Form 4 and 13F filing retrieval
- SEC rate limit compliance (10 req/sec)

## Class Structure

```python
class EDGARProvider(DataProvider):
    """SEC EDGAR data provider via edgartools.
    
    Capabilities:
    - fetch_insider_transactions: Form 4 filings → insider_trades schema
    - fetch_institutional_holdings: 13F filings → institutional_ownership schema  
    - fetch_fundamentals: XBRL Company Facts → financial_statements schema
    - fetch_sec_filings: Filing metadata (10-K, 10-Q, 8-K)
    
    All other DataProvider methods raise NotImplementedError (EDGAR does not
    provide OHLCV, options, news, or macro data).
    """
    
    def __init__(self):
        """Validate EDGAR_USER_AGENT is set. Run startup health check."""
        # 1. Read EDGAR_USER_AGENT from env; raise ConfigurationError if missing
        # 2. Configure edgartools with the user agent header
        # 3. Health check: resolve CIK for a known ticker (e.g., "AAPL")
        #    to verify connectivity. Log warning if health check fails but
        #    do not raise — allow degraded operation.
    
    def name(self) -> str:
        return "edgar"
```

## CIK Resolution

`edgartools` handles ticker-to-CIK mapping internally. Cache the mapping in a dict on the provider instance to avoid repeated lookups for the same ticker within a session. If a ticker has no CIK mapping (e.g., very new IPOs), return `None` from the fetch method rather than raising.

## Data Normalization

EDGAR returns data in different schemas than QuantStack's existing tables. The provider is responsible for normalizing before returning.

### Form 4 (Insider Transactions) → `insider_trades` schema

| EDGAR Field | Target Column | Transformation |
|------------|---------------|----------------|
| ticker (from lookup) | `ticker` | Pass-through |
| `transactionDate` | `transaction_date` | Parse to `date` |
| `ownerName` | `owner_name` | Pass-through |
| `transactionAcquiredDisposedCode` | `transaction_type` | `'A'` → `'buy'`, `'D'` → `'sell'` |
| `transactionShares` | `shares` | Absolute value (EDGAR may report negative for dispositions) |
| `pricePerShare` | `price_per_share` | Float, may be 0 for gifts/exercises |

### 13F (Institutional Holdings) → `institutional_ownership` schema

Map 13F filing fields to the existing `institutional_ownership` table columns. The exact field names depend on `edgartools`' API surface -- consult the library documentation during implementation.

### XBRL Company Facts → `financial_statements` schema

XBRL facts provide revenue, net income, total assets, etc. Map the standard US-GAAP taxonomy concepts to the existing `financial_statements` table structure. Focus on the most commonly available facts:
- `us-gaap:Revenues` or `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax`
- `us-gaap:NetIncomeLoss`
- `us-gaap:Assets`
- `us-gaap:Liabilities`
- `us-gaap:StockholdersEquity`
- `us-gaap:EarningsPerShareBasic`

Return `None` if XBRL data is unavailable (e.g., pre-2009 filings or foreign filers using IFRS).

## Rate Limiting

SEC EDGAR allows 10 requests per second. Implement a simple throttle (e.g., `time.sleep` with a token bucket or sliding window) inside the provider. This is independent of Alpha Vantage's rate limiting -- no shared quota.

With 50 universe symbols, a full EDGAR sweep takes approximately 15 seconds:
- Filing metadata: ~50 calls → 5 seconds
- Form 4: ~50 calls → 5 seconds
- 13F: ~50 calls → 5 seconds

## Startup Health Check

At `__init__` time, after configuring the user agent, attempt to resolve the CIK for a well-known ticker (e.g., `"AAPL"` → CIK `0000320193`). This validates that:
1. The `edgartools` library is functional
2. SEC EDGAR endpoints are reachable
3. The user agent header is accepted

If the health check fails, log a warning but do not prevent the provider from being registered. The provider registry's circuit breaker (section-09) will handle runtime failures.

## Error Handling

- Network errors from SEC EDGAR: catch, log with full context (ticker, endpoint, HTTP status), return `None`. The registry treats `None` as "no data available" and tries the next provider.
- Malformed EDGAR responses: catch parsing errors, log the raw response snippet for debugging, return `None`.
- CIK resolution failures: return `None` for the specific ticker. Do not let one bad ticker poison the entire provider.
- Never catch and silence. Every caught exception must be logged with sufficient context for debugging.

## Risks

| Risk | Mitigation |
|------|-----------|
| `edgartools` library breaks due to SEC endpoint changes | Pin version. Startup health check detects breakage early. Fallback to AV for fundamentals/insider data via the registry. |
| XBRL data gaps for older filings | Skip pre-2009 filings. Focus on recent years where XBRL coverage is strong. |
| SEC rate limit changes | Current 10 req/sec is well-documented. Monitor for 429 responses and back off automatically. |
| Ticker-to-CIK resolution gaps (new IPOs) | Return `None` gracefully. AV serves as fallback for these tickers. |
