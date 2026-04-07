# Section 11: SEC Filings Population

## Overview

The `sec_filings` and `insider_trades` tables exist in the schema but have never been populated with data. This section adds two new acquisition phases (15 and 16) that use the EDGAR provider (built in section-08) to fetch SEC filing metadata, insider transactions (Form 4), and institutional holdings (13F) for all universe symbols. Data is normalized into existing table schemas and upserted idempotently.

**Dependency:** Section 08 (EDGAR Provider) must be completed first. This section consumes `EDGARProvider.fetch_sec_filings()`, `fetch_insider_transactions()`, and `fetch_institutional_holdings()` methods.

---

## Tests (Write First)

All tests go in `tests/unit/test_sec_filings.py`.

```python
# Test: sec_filings table created with correct schema (accession_number PK)
def test_sec_filings_table_schema(): ...

# Test: SEC filings acquisition phase populates 10-K, 10-Q, 8-K for a test symbol
def test_sec_filings_phase_populates_filings(): ...

# Test: Form 4 insider data normalized and upserted into insider_trades table
def test_form4_data_normalized_to_insider_trades(): ...

# Test: upsert is idempotent — running twice produces same row count
def test_sec_filings_upsert_idempotent(): ...

# Test: freshness check skips symbol with filing_date < 90 days ago
def test_freshness_check_skips_recent_filings(): ...

# Test: EDGAR rate limiting respected during bulk acquisition (50 symbols)
def test_edgar_rate_limiting_bulk_acquisition(): ...
```

Additionally, the cross-cutting test in the main test suite should verify:

```python
# Test: sec_filings table DDL in _schema.py
def test_sec_filings_ddl_exists(): ...
```

### Test Design Notes

- Mock `EDGARProvider` methods to return fixture data matching EDGAR response shapes. Do not call real EDGAR API in unit tests.
- For the idempotency test, run the acquisition phase twice with the same fixture data and assert row counts are identical.
- For the rate limiting test, mock the provider and assert that calls are spaced at no more than 10 per second (EDGAR's rate limit).
- For the freshness test, insert a `sec_filings` row with `filing_date` within the last 90 days and assert the phase skips that symbol.

---

## New Table: `sec_filings`

Add the following DDL to `src/quantstack/data/_schema.py`:

```python
@dataclass
class SECFiling:
    accession_number: str    # PK — SEC's unique filing identifier
    symbol: str
    form_type: str           # '10-K', '10-Q', '8-K'
    filing_date: date
    period_of_report: date | None
    primary_doc_url: str | None
    fetched_at: datetime
```

The SQL DDL should create the table with `accession_number` as the primary key, an index on `(symbol, form_type)`, and an index on `filing_date` for freshness queries.

---

## Acquisition Phase 15: SEC Filing Metadata

Add a new phase to `src/quantstack/data/acquisition_pipeline.py`.

**What it does:**
- For each symbol in the universe, call `registry.fetch("sec_filings", symbol, form_types=["10-K", "10-Q", "8-K"])` via the provider registry (or directly via `EDGARProvider.fetch_sec_filings()` if the registry integration from section-10 is not yet wired).
- Normalize the response into `sec_filings` table rows.
- Upsert using `ON CONFLICT (accession_number) DO UPDATE` to maintain idempotency.
- Update `data_metadata` for `(symbol, "sec_filings")` with the current timestamp after successful fetch.

**Freshness logic:** Skip a symbol if a `sec_filings` row exists with `filing_date` within the last 90 days. This avoids redundant EDGAR calls for symbols whose quarterly filings are already captured.

**Rate limiting:** EDGAR allows 10 req/sec. With ~50 universe symbols at 1 call per symbol, the full phase completes in approximately 5 seconds. The EDGAR provider handles throttling internally.

---

## Acquisition Phase 16: EDGAR Insider and Institutional Data

Add a second new phase to `src/quantstack/data/acquisition_pipeline.py`.

**What it does:**
- **Form 4 (insider transactions):** For each symbol, call `EDGARProvider.fetch_insider_transactions(symbol)`. Normalize the response into the existing `insider_trades` table schema.
- **13F (institutional holdings):** For each symbol, call `EDGARProvider.fetch_institutional_holdings(symbol)`. Normalize into the existing `institutional_ownership` table schema.
- Upsert both using `ON CONFLICT DO UPDATE` for idempotency.
- Update `data_metadata` for each data type after successful fetch.

**Freshness logic:**
- Insider transactions (Form 4): Fetch if `fetched_at` in `data_metadata` is older than 7 days.
- Institutional holdings (13F): Fetch if `fetched_at` is older than 90 days (13F filings are quarterly).

---

## Data Normalization

EDGAR returns data in different field names than the existing table schemas. The EDGAR provider (from section-08) handles this mapping, but the acquisition phases must validate the output matches what the tables expect.

**Insider transactions mapping (EDGAR -> `insider_trades` table):**

| EDGAR Field | Table Column | Notes |
|-------------|-------------|-------|
| `transactionDate` | `transaction_date` | Date type |
| `ownerName` | `owner_name` | String |
| `transactionShares` | `shares` | Integer |
| `pricePerShare` | `price_per_share` | Decimal |
| `transactionAcquiredDisposedCode` | `transaction_type` | 'A' -> 'buy', 'D' -> 'sell' |
| `isDirector` / `isOfficer` / `isTenPercentOwner` | (informational) | Used to enrich but not mapped to a specific column |

**SEC filings mapping:** EDGAR filing metadata maps directly to the `sec_filings` table columns. The `accession_number` serves as a natural primary key since it is globally unique across all SEC filings.

---

## Files to Create/Modify

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/data/_schema.py` | Modify | Add `sec_filings` table DDL (CREATE TABLE with accession_number PK) |
| `src/quantstack/data/acquisition_pipeline.py` | Modify | Add Phase 15 (SEC filings) and Phase 16 (EDGAR insider/institutional) |
| `tests/unit/test_sec_filings.py` | Create | All tests listed above |

The EDGAR provider itself (`src/quantstack/data/providers/edgar.py`) is created in section-08 and is a prerequisite, not modified here.

---

## Implementation Checklist

1. Write all six test stubs in `tests/unit/test_sec_filings.py` with descriptive docstrings.
2. Add the `sec_filings` table DDL to `_schema.py`.
3. Implement Phase 15 in `acquisition_pipeline.py` — SEC filing metadata fetch, normalize, upsert, update `data_metadata`.
4. Implement Phase 16 in `acquisition_pipeline.py` — Form 4 insider transactions and 13F institutional holdings fetch, normalize, upsert, update `data_metadata`.
5. Verify upsert idempotency by running the phase twice against a test database.
6. Verify freshness skip logic prevents redundant EDGAR calls.
7. Run the full test suite and confirm all six tests pass.
