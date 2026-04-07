# Section 03: Corporate Actions Monitor

## Overview

This section implements automated detection and handling of corporate actions (dividends, splits, M&A events) for all universe symbols. It covers three data collectors (AV dividends, AV splits, EDGAR 8-K), split auto-adjustment with broker reconciliation, M&A thesis flagging, and a daily scheduled job orchestrating the pipeline.

**Dependencies:** Section 01 (DB schema for `corporate_actions` and `split_adjustments` tables) and Section 02 (system alerts for `emit_system_alert()` helper).

**New files:**
- `src/quantstack/data/corporate_actions.py` — collectors, split adjustment, CIK mapping
- `tests/unit/test_corporate_actions.py`
- `tests/integration/test_corporate_actions_integration.py`

**Modified files:**
- `src/quantstack/data/scheduled_refresh.py` — add `refresh_corporate_actions()` entry point
- `pyproject.toml` — add `edgartools` dependency

---

## Tests (Write First)

### Unit Tests — `tests/unit/test_corporate_actions.py`

```python
"""Unit tests for corporate actions monitor."""
import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, patch, MagicMock


class TestFetchAVDividends:
    """Tests for fetch_av_dividends()."""

    # Test: parses AV response into list[CorporateAction] with correct fields
    # - Mock AV HTTP response with 2 dividend records
    # - Assert returned list has 2 items, each with event_type="dividend",
    #   source="alpha_vantage", correct symbol, effective_date, raw_payload

    # Test: handles empty response (no dividends) gracefully
    # - Mock AV response with empty data array
    # - Assert returns empty list, no exception

    # Test: handles "None" string values in declaration/record dates
    # - Mock AV response where declaration_date and record_date are literal "None" strings
    # - Assert CorporateAction.announcement_date is Python None (not string "None")


class TestFetchAVSplits:
    """Tests for fetch_av_splits()."""

    # Test: parses split response with correct split_ratio
    # - Mock AV response with a 4:1 split record
    # - Assert CorporateAction has event_type="split", raw_payload contains split_ratio=4.0


class TestFetchEdgar8KEvents:
    """Tests for fetch_edgar_8k_events()."""

    # Test: parses 8-K items 1.01, 2.01, 3.03, 5.01 into CorporateAction
    # - Mock EDGAR submissions API response with filings containing these item codes
    # - Assert each parsed into CorporateAction with event_type matching item semantics
    #   (1.01 -> "merger_signing", 2.01 -> "acquisition_complete", etc.)

    # Test: skips 8-K items we don't care about (e.g., 5.07)
    # - Mock response with item 5.07 only
    # - Assert returns empty list

    # Test: handles missing CIK gracefully (logs warning, returns empty)
    # - Call with cik=None or empty string
    # - Assert returns empty list, no exception raised


class TestCIKMapping:
    """Tests for CIK -> ticker resolution."""

    # Test: loads from company_tickers.json format and resolves ticker -> CIK
    # - Mock HTTP response matching SEC company_tickers.json structure
    # - Assert lookup("AAPL") returns correct CIK string

    # Test: returns None for unknown ticker (doesn't crash)
    # - Assert lookup("ZZZZZ") returns None


class TestApplySplitAdjustment:
    """Tests for apply_split_adjustment()."""

    # Test: computes correct new_qty and new_cost for 4:1 split
    # - old_qty=10, old_cost=200.0, ratio=4.0
    # - Assert new_qty=40, new_cost=50.0

    # Test: computes correct values for 1:10 reverse split (ratio=0.1)
    # - old_qty=100, old_cost=5.0, ratio=0.1
    # - Assert new_qty=10, new_cost=50.0

    # Test: asserts invariant: old_qty * old_cost == new_qty * new_cost
    # - For any ratio, assert total_cost is preserved within floating point tolerance

    # Test: is idempotent -- second call for same (symbol, date) is no-op
    # - Mock split_adjustments table already containing a row for this symbol+date
    # - Assert function returns None (no-op), no new DB write

    # Test: handles fractional shares on reverse split (rounds down)
    # - old_qty=15, ratio=0.1 -> new_qty=1 (floor), note cash-out in metadata

    # Test: skips if broker already adjusted (qty reconciliation)
    # - Mock Alpaca position qty already matching post-split value
    # - Assert function syncs DB to broker state and returns without manual adjustment


class TestRefreshCorporateActions:
    """Tests for the orchestration function."""

    # Test: deduplicates on insert (unique constraint, no error)
    # - Insert a corporate action, then call refresh which returns the same action
    # - Assert no IntegrityError, row count unchanged
```

### Integration Tests — `tests/integration/test_corporate_actions_integration.py`

```python
"""Integration tests for corporate actions monitor."""
import pytest


class TestRefreshCorporateActionsE2E:
    """End-to-end tests with mocked external APIs but real DB."""

    # Test: refresh_corporate_actions end-to-end with mocked AV + EDGAR responses
    # - Mock AV dividends, AV splits, and EDGAR 8-K responses
    # - Call refresh_corporate_actions(["AAPL", "TSLA"])
    # - Assert rows written to corporate_actions table
    # - Assert summary dict has correct counts

    # Test: split auto-adjustment updates position in DB and writes audit row
    # - Seed a position in DB (symbol="AAPL", qty=10, cost_basis=200.0)
    # - Mock AV split response with 4:1 split for AAPL
    # - Run refresh_corporate_actions
    # - Assert position updated: qty=40, cost_basis=50.0
    # - Assert split_adjustments row written with correct old/new values

    # Test: M&A detection creates system alert with correct category and severity
    # - Mock EDGAR 8-K response with item 1.01 for a held symbol
    # - Run refresh_corporate_actions
    # - Assert system_alerts table has row with category='thesis_review', severity='critical'
```

---

## Implementation Details

### Data Model

The `corporate_actions` and `split_adjustments` tables are created in Section 01. For reference, the key structures used by this module:

**CorporateAction** fields: `symbol`, `event_type` (dividend/split/merger/acquisition/delisting), `source` (alpha_vantage/edgar_8k), `effective_date`, `announcement_date` (nullable), `raw_payload` (JSONB), `processed` (bool), `created_at`.

Unique constraint: `(symbol, event_type, effective_date, source)` enables idempotent inserts.

**SplitAdjustment** fields: `symbol`, `effective_date`, `split_ratio`, `old_quantity`, `new_quantity`, `old_cost_basis`, `new_cost_basis`, `applied_at`.

Unique constraint: `(symbol, effective_date, event_type)` prevents double-adjustment.

**Invariant:** `old_quantity * old_cost_basis == new_quantity * new_cost_basis` (total cost basis preserved through any split).

### New Module: `src/quantstack/data/corporate_actions.py`

This module contains all corporate actions logic: collectors, adjustment, and CIK mapping.

#### Alpha Vantage Collectors

```python
async def fetch_av_dividends(symbol: str) -> list[CorporateAction]:
    """Fetch dividend history from AV DIVIDENDS endpoint.
    
    Uses the existing AV rate limiter from src/quantstack/data/fetcher.py.
    Parses response into CorporateAction objects. Handles empty responses
    and "None" string sentinel values in date fields (AV returns literal
    string "None" for missing dates instead of null/omitting the field).
    """

async def fetch_av_splits(symbol: str) -> list[CorporateAction]:
    """Fetch split history from AV SPLITS endpoint.
    
    Parses split_ratio from AV format (e.g., "4:1" string) into float (4.0).
    Reverse splits appear as "1:10" -> ratio 0.1.
    """
```

Both follow the existing AV call pattern in `src/quantstack/data/fetcher.py` -- use the same HTTP client, rate limiter, and error handling. Poll daily for all universe symbols. Deduplicate via INSERT ... ON CONFLICT DO NOTHING using the unique constraint.

#### EDGAR 8-K Collector

```python
async def fetch_edgar_8k_events(symbol: str, cik: str) -> list[CorporateAction]:
    """Fetch recent 8-K filings from EDGAR submissions API.
    
    Target items and their event_type mapping:
    - Item 1.01 (Entry into Material Agreement) -> "merger_signing"
    - Item 2.01 (Completion of Acquisition/Disposition) -> "acquisition_complete"  
    - Item 3.03 (Material Modification of Rights) -> "rights_modification"
    - Item 5.01 (Changes in Control) -> "change_of_control"
    
    All other 8-K items are ignored.
    
    SEC rate limit: 10 requests/second with proper User-Agent header
    (SEC requires: "CompanyName AdminEmail" format).
    
    If CIK is None or empty, logs a warning and returns empty list.
    """
```

Requires the `edgartools` library (add to `pyproject.toml` dependencies). If `edgartools` does not natively parse 8-K items, fall back to the EDGAR submissions API directly for filing metadata and flag for manual review rather than automated item extraction.

#### CIK Mapping

```python
class CIKMapper:
    """Resolves ticker symbols to SEC CIK numbers.
    
    Uses SEC's company_tickers.json endpoint:
    https://www.sec.gov/files/company_tickers.json
    
    Loaded at startup, cached in memory. Refreshed weekly via supervisor
    scheduled task. Returns None for unknown tickers (does not raise).
    """
    
    async def load(self) -> None:
        """Fetch and parse company_tickers.json from SEC."""
    
    def lookup(self, ticker: str) -> str | None:
        """Return CIK for ticker, or None if not found."""
```

#### Split Auto-Adjustment

```python
async def apply_split_adjustment(
    symbol: str, split_ratio: float, effective_date: date
) -> SplitAdjustment | None:
    """Auto-adjust cost basis and quantity for a stock split.
    
    Steps:
    1. Check split_adjustments table -- if row exists for (symbol, effective_date),
       return None (idempotent, already applied).
    2. Query Alpaca broker for current position qty. Compare to DB qty.
       If broker qty already reflects post-split value, sync DB to broker
       state and skip manual adjustment (Alpaca auto-adjusts).
    3. Read current position from portfolio_state DB.
    4. Compute: new_qty = old_qty * split_ratio
               new_cost = old_cost / split_ratio
    5. Assert invariant: abs(old_qty * old_cost - new_qty * new_cost) < 0.01
    6. Update position in DB via db_conn() context manager.
    7. Write audit row to split_adjustments table.
    8. Call emit_system_alert() with info about the applied split.
    
    For reverse splits (ratio < 1): floor the new_qty to handle fractional
    shares. Record the fractional remainder and estimated cash-out amount
    in the split_adjustments metadata.
    
    Returns the SplitAdjustment record, or None if skipped (idempotent/broker-adjusted).
    """
```

The Alpaca reconciliation check is critical to avoid double-adjustment. Alpaca automatically adjusts positions for splits, so if the broker has already applied the adjustment, we just sync the DB to match broker state.

#### M&A Thesis Flagging

When `fetch_edgar_8k_events()` returns an item 1.01 (merger signing) or 2.01 (acquisition complete) for a symbol that is currently held:

1. Call `emit_system_alert()` (from Section 02) with:
   - `category="thesis_review"`
   - `severity="critical"`
   - `title=f"M&A event detected: {symbol} - {item_description}"`
   - `detail=` filing date, item type, brief description from filing
   - `metadata={"symbol": symbol, "filing_date": ..., "item_code": ...}`
2. The supervisor graph's `health_check` node picks up critical alerts automatically.

### Scheduled Job: `refresh_corporate_actions()`

Add to `src/quantstack/data/scheduled_refresh.py`:

```python
async def refresh_corporate_actions(symbols: list[str]) -> dict:
    """Daily corporate actions check for all holdings.
    
    Called by supervisor graph's scheduled_tasks node.
    
    Pipeline:
    1. Fetch AV dividends for all symbols (rate-limited at 75/min)
    2. Fetch AV splits for all symbols (same rate limiter)
    3. Load CIK mapping, fetch EDGAR 8-K events for all symbols (10 req/s)
    4. Insert new events via INSERT ... ON CONFLICT DO NOTHING (dedup)
    5. Find unprocessed splits (processed=False), apply each via apply_split_adjustment()
    6. Find M&A events for held symbols, create thesis_review alerts
    7. Mark all processed events as processed=True
    
    Returns: {
        "new_dividends": int,
        "new_splits": int, 
        "new_ma_events": int,
        "splits_applied": int,
        "errors": list[str]  # any per-symbol failures (non-fatal)
    }
    """
```

**Rate limiting budget:** For a 50-symbol universe, this requires ~100 AV calls (dividends + splits) and ~50 EDGAR calls. At 75 AV calls/min, that is ~1.5 minutes for AV. EDGAR at 10 req/s finishes in ~5 seconds. Well within daily budget.

**Error handling:** Per-symbol failures are caught and logged but do not abort the entire refresh. If EDGAR is down or a CIK is missing, AV data still covers dividends and splits. The function returns an `errors` list for observability.

### Dependency: `edgartools`

Add `edgartools` to `pyproject.toml` under `[project.dependencies]`. If `edgartools` proves inadequate for 8-K item parsing at implementation time, the fallback is to hit the EDGAR submissions API directly (`https://efts.sec.gov/LATEST/search-index?q=...`) and parse the JSON response. The key requirement is extracting item codes (1.01, 2.01, 3.03, 5.01) from 8-K filings -- the exact library is a means, not a constraint.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| EDGAR rate limiting or parsing failures | Graceful degradation: if EDGAR fails, AV data still covers splits/dividends. Log warning, do not block the pipeline. |
| Alpaca auto-adjusts splits causing double adjustment | Reconciliation check: compare broker position qty vs DB qty before applying. If broker already adjusted, sync DB to broker and skip manual adjustment. |
| `edgartools` library may not parse 8-K items natively | Verify during implementation. Fallback: use EDGAR submissions API directly for filing metadata. |
| AV returns stale or missing data for recent corporate actions | Cross-reference: splits detected by AV should match broker-reported adjustments. Dividends are informational (no position adjustment needed). |
| Fractional shares after reverse split | Round down new_qty, record cash-out estimate in metadata. This matches broker behavior (Alpaca pays cash for fractional shares on reverse splits). |
