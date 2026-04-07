# Section 01: Staleness Helper

## Overview

This section creates the foundational freshness-checking infrastructure that all 22 signal collectors (except `ml_signal`) will use to reject stale data before computing signals. A collector running RSI on 3-week-old prices produces a confident but meaningless signal. The fix is a shared helper that checks `data_metadata.last_timestamp` and returns a boolean verdict before any computation begins.

This section delivers two things:
1. The `check_freshness()` function in a new module `src/quantstack/signal_engine/staleness.py`
2. Extended `data_metadata` coverage across all data source types (currently only OHLCV phases populate it)

Section 05 (staleness-collectors) depends on this section and wires the helper into all 22 collectors. This section is strictly the helper and its prerequisite data coverage.

---

## Background

### data_metadata Table

The `data_metadata` table already exists in the schema (defined in `src/quantstack/db.py`):

```sql
CREATE TABLE IF NOT EXISTS data_metadata (
    symbol          VARCHAR     NOT NULL,
    timeframe       VARCHAR     NOT NULL,
    first_timestamp TIMESTAMPTZ,
    last_timestamp  TIMESTAMPTZ,
    row_count       INTEGER,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe)
)
```

The `timeframe` column currently stores OHLCV timeframe values (e.g., `"1d"`, `"5min"`, `"1h"`). For non-OHLCV data sources, we repurpose this column as a data source identifier (e.g., `"macro_indicators"`, `"news_sentiment"`, `"options_chains"`). The column name is slightly misleading for non-OHLCV use, but adding a new table or renaming the column would be a larger migration with no functional benefit.

### DataStore

`src/quantstack/data/storage.py` defines `DataStore(PgDataStore)`. Collectors instantiate it with `DataStore(read_only=True)` to load data. The staleness helper needs a database connection to query `data_metadata` — it accepts a `DataStore` instance to stay consistent with how collectors already access data.

### Collector Pattern

Each collector (e.g., `src/quantstack/signal_engine/collectors/technical.py`) loads data from `DataStore`, computes indicators, and returns a flat dict. If a collector cannot compute (no data, error), it returns `{}`. The synthesis engine already handles empty dicts by redistributing weight across remaining collectors.

---

## Tests (Write First)

Create `tests/unit/test_staleness_helper.py`:

```python
"""Tests for src/quantstack/signal_engine/staleness.py"""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch


class TestCheckFreshness:
    """Tests for the check_freshness() function."""

    def test_returns_true_when_data_within_max_days(self):
        """data_metadata.last_timestamp is 1 day old, max_days=4 → True."""
        # Arrange: mock DataStore that returns last_timestamp = 1 day ago
        # Act: call check_freshness(store, "AAPL", "ohlcv", max_days=4)
        # Assert: returns True

    def test_returns_false_when_data_exceeds_max_days(self):
        """data_metadata.last_timestamp is 5 days old, max_days=4 → False."""
        # Arrange: mock DataStore that returns last_timestamp = 5 days ago
        # Act: call check_freshness(store, "AAPL", "ohlcv", max_days=4)
        # Assert: returns False

    def test_returns_false_when_no_metadata_row_exists(self):
        """No data_metadata row for symbol/table → False (missing = stale)."""
        # Arrange: mock DataStore that returns None for the metadata query
        # Act: call check_freshness(store, "AAPL", "ohlcv", max_days=4)
        # Assert: returns False

    def test_uses_calendar_days_not_trading_days(self):
        """4 calendar days covers a 3-day weekend (Fri close → Tue open)."""
        # Arrange: last_timestamp = Friday, now = Tuesday (4 calendar days)
        # Act: call check_freshness(store, "AAPL", "ohlcv", max_days=4)
        # Assert: returns True (exactly at threshold)

    def test_logs_warning_when_stale(self):
        """When stale, logs warning with symbol, table, actual age, threshold."""
        # Arrange: stale data
        # Act: call check_freshness(...)
        # Assert: logger.warning called with structured message containing
        #   symbol, table name, actual age in days, and max_days threshold

    def test_logs_warning_when_missing(self):
        """When no metadata row, logs warning identifying the gap."""
        # Arrange: no metadata row
        # Act: call check_freshness(...)
        # Assert: logger.warning called

    def test_does_not_log_when_fresh(self):
        """Fresh data produces no warning log."""
        # Arrange: fresh data
        # Act: call check_freshness(...)
        # Assert: logger.warning NOT called
```

---

## Implementation

### New File: `src/quantstack/signal_engine/staleness.py`

```python
"""Staleness checking for signal engine collectors.

Each collector calls check_freshness() before computing signals.
If the underlying data is too old, the collector should return {}
and let the synthesis engine redistribute weight.

Freshness is determined by querying data_metadata.last_timestamp.
The table column is (symbol, timeframe) where 'timeframe' doubles
as a data source identifier for non-OHLCV tables (e.g., "macro_indicators").
"""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from quantstack.db import db_conn


def check_freshness(
    symbol: str,
    table: str,
    max_days: int,
) -> bool:
    """Check if the most recent data for symbol in table is within max_days of now.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        table: Data source identifier as stored in data_metadata.timeframe
               (e.g., "1d" for daily OHLCV, "macro_indicators" for macro data).
        max_days: Maximum acceptable age in calendar days.

    Returns:
        True if data is fresh enough to use. False if stale or missing.
    """
    ...
```

The implementation queries `data_metadata` for the row matching `(symbol, table)`, compares `last_timestamp` against `datetime.now(timezone.utc) - timedelta(days=max_days)`, and returns the boolean result. Use `db_conn()` context manager for the query (single indexed lookup on the PK).

Key implementation details:
- Use `db_conn()` directly rather than accepting a `DataStore` parameter. The staleness check is a single metadata lookup — it does not need the full DataStore interface. This also avoids forcing collectors to pass their store instance through.
- If the query returns no row, treat as stale (return `False`) and log a warning identifying the missing metadata.
- If `last_timestamp` is `NULL`, treat as stale.
- All timestamps compared in UTC.

### Staleness Threshold Constants

Define a module-level dict mapping collector categories to their thresholds. This centralizes the configuration and makes it easy for section 05 to look up the right threshold per collector:

```python
# Staleness thresholds per data source type (calendar days).
# Rationale for each threshold is documented in the implementation plan.
STALENESS_THRESHOLDS: dict[str, int] = {
    # Price-derived: 2 trading days = up to 4 calendar days (covers 3-day weekends)
    "ohlcv": 4,
    # Options-derived: chains update daily
    "options_chains": 3,
    # News/sentiment: longer relevance window
    "news_sentiment": 7,
    # Fundamentals: quarterly data
    "company_overview": 90,
    # Macro/commodity: monthly indicators
    "macro_indicators": 45,
    # Insider/flow: reported quarterly, some real-time
    "insider_trades": 30,
    # Short interest: bi-monthly FINRA updates
    "short_interest": 14,
    # Sector: weekly rotation signals
    "sector": 7,
    # Events: calendar-based
    "events": 30,
    # EWF: external forecast freshness
    "ewf": 7,
}
```

### Prerequisite: Extend data_metadata Coverage

Currently only OHLCV acquisition phases update `data_metadata`. The staleness helper is useless for non-OHLCV collectors until their acquisition phases also populate metadata rows.

Modify `src/quantstack/data/acquisition_pipeline.py` to upsert a `data_metadata` row after each non-OHLCV acquisition phase completes. The upsert pattern:

```sql
INSERT INTO data_metadata (symbol, timeframe, last_timestamp, updated_at)
VALUES (:symbol, :table_name, :last_ts, NOW())
ON CONFLICT (symbol, timeframe)
DO UPDATE SET last_timestamp = EXCLUDED.last_timestamp,
             updated_at = NOW()
```

Phases that need this addition (the `timeframe` value to use is in parentheses):
- Macro indicators phase (`"macro_indicators"`)
- News sentiment phase (`"news_sentiment"`)
- Options chains phase (`"options_chains"`)
- Insider transactions phase (`"insider_trades"`)
- Institutional ownership phase (`"institutional_ownership"`)
- Company overview / fundamentals phase (`"company_overview"`)
- Earnings history phase (`"earnings_history"`)
- Commodity data phase (`"commodity"`)

For each phase, after the main data insert/upsert succeeds, compute `last_ts` as the max timestamp from the fetched batch and upsert into `data_metadata`. If the phase fetches no data for a symbol, do not update the metadata row (absence of data is not the same as fresh data).

### Test for metadata coverage

Add a validation test to `tests/unit/test_staleness_helper.py`:

```python
class TestMetadataCoverage:
    """Verify that all data source types used by STALENESS_THRESHOLDS
    are populated by the acquisition pipeline."""

    def test_all_threshold_keys_have_acquisition_coverage(self):
        """Every key in STALENESS_THRESHOLDS must correspond to a data source
        that the acquisition pipeline populates in data_metadata.
        
        This is a documentation/contract test — it verifies the keys match
        the known set of acquisition phase identifiers.
        """
        from quantstack.signal_engine.staleness import STALENESS_THRESHOLDS
        expected_sources = {
            "ohlcv", "options_chains", "news_sentiment", "company_overview",
            "macro_indicators", "insider_trades", "short_interest", "sector",
            "events", "ewf",
        }
        assert set(STALENESS_THRESHOLDS.keys()) == expected_sources
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/quantstack/signal_engine/staleness.py` | `check_freshness()` function and `STALENESS_THRESHOLDS` dict |
| `tests/unit/test_staleness_helper.py` | Unit tests for freshness checking |

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/data/acquisition_pipeline.py` | Add `data_metadata` upserts after each non-OHLCV acquisition phase |

## Dependencies

- **Depends on:** Nothing (Batch 1, no prerequisites)
- **Blocks:** Section 05 (staleness-collectors) — which wires `check_freshness()` into all 22 collectors

## Performance Considerations

With `data_metadata` populated universally, `check_freshness()` is a single indexed PK lookup per call. At 22 collectors x 50 symbols = 1,100 calls per signal engine cycle, this adds roughly 100ms total overhead (negligible compared to the 2-6 second collector runtime). If benchmarks in section 05 show overhead exceeding 2 seconds, the mitigation is to batch all freshness checks into a single query at the engine level before dispatching to collectors — but this optimization should not be built preemptively.
