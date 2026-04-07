# Section 02: Signal Cache Auto-Invalidation

## Overview

The signal engine caches `SignalBrief` results with a 1-hour TTL to avoid re-running 22 collectors on repeated lookups. However, the intraday data refresh cycle runs every 5 minutes, writing fresh quotes, OHLCV bars, and news sentiment to the database. Nothing invalidates the cache after these writes, so a trading decision can use a SignalBrief built on data that is up to 55 minutes stale.

The fix: call `cache.invalidate(symbol)` in `scheduled_refresh.py` after each successful data write, and log cache stats after each refresh cycle.

## Dependencies

None. This section is independently implementable. The cache module (`src/quantstack/signal_engine/cache.py`) already exposes `invalidate(symbol)` and `stats()` — both are implemented but never called from the refresh path.

## Files to Modify

- `src/quantstack/data/scheduled_refresh.py` — add cache invalidation calls after data writes, add stats logging

No new files required.

---

## Tests

Write these in `tests/unit/test_cache_invalidation.py` before implementing.

```python
"""Tests for signal cache auto-invalidation after intraday refresh."""

import pytest


def test_intraday_refresh_invalidates_refreshed_symbols():
    """After intraday refresh completes for AAPL, cache.get('AAPL') returns None."""
    # Pre-populate cache with a brief for AAPL
    # Run intraday refresh (mock AlphaVantageClient and PgDataStore)
    # Assert cache.get("AAPL") returns None


def test_invalidation_is_per_symbol():
    """After refresh for [AAPL, MSFT], only those two are invalidated; GOOG remains cached."""
    # Pre-populate cache with briefs for AAPL, MSFT, GOOG
    # Run intraday refresh for AAPL and MSFT only
    # Assert cache.get("AAPL") is None
    # Assert cache.get("MSFT") is None
    # Assert cache.get("GOOG") is not None


def test_eod_refresh_calls_cache_clear():
    """EOD refresh calls cache.clear() — existing behavior preserved."""
    # Verify run_eod_refresh calls cache.clear() at the end


def test_failed_write_does_not_invalidate():
    """If the data write fails for a symbol, that symbol's cache is NOT invalidated."""
    # Pre-populate cache with AAPL brief
    # Mock store.save_ohlcv to raise for AAPL
    # Run intraday refresh
    # Assert cache.get("AAPL") is still the original brief


def test_cache_stats_logged_after_invalidation():
    """Cache stats are logged after each intraday refresh cycle."""
    # Mock logger
    # Run intraday refresh
    # Assert logger.info was called with a message containing cache stats


def test_race_condition_accepted():
    """Document: if signal engine caches during refresh window, brief is re-cached.
    This is accepted behavior — seconds of staleness vs the prior 55-minute window.
    """
    # This is a documentation test, not a behavioral assertion.
    # The test exists to confirm the team acknowledges the narrow race window.
    pass
```

---

## Implementation Details

### Current State

`src/quantstack/data/scheduled_refresh.py` has two async functions:

- `run_intraday_refresh()` — runs every 5 minutes during market hours. Three steps:
  1. Bulk quotes for up to 100 universe symbols (1 API call)
  2. 5-min OHLCV for up to 20 watched symbols (1 call each)
  3. News sentiment for top 10 symbols (batched 5 per call)

- `run_eod_refresh()` — runs once after market close. Fetches daily OHLCV, options chains, fundamentals, and earnings calendar.

`src/quantstack/signal_engine/cache.py` already provides:
- `invalidate(symbol: str)` — removes a single symbol's cached brief
- `clear()` — drops all cached briefs
- `stats()` — returns `{"hits": N, "misses": N, "size": N}`

### Changes to `scheduled_refresh.py`

**Add import at top of file:**

```python
from quantstack.signal_engine import cache as signal_cache
```

**Step 1 — After bulk quotes succeed:** Invalidate all symbols that received fresh quotes. The symbols are those in `all_symbols[:100]` for which the bulk quote returned data. After the `for _, row in quotes_df.iterrows()` loop that counts refreshed symbols, collect the successfully refreshed symbols and invalidate each one.

**Step 2 — After each 5-min OHLCV write succeeds:** Inside the `for sym in watched[:20]` loop, after `store.save_ohlcv` completes successfully, call `signal_cache.invalidate(sym)`. Place the call inside the `if df is not None and not df.empty` block, after the save. Do NOT invalidate in the `except` branch — if the write fails, the stale cache is the best available data.

**Step 3 — After each news sentiment batch succeeds:** After `store.save_news_sentiment` completes, invalidate each symbol in the batch: `for sym in batch: signal_cache.invalidate(sym)`. Place inside the success branch only.

**After all three steps complete:** Before the final `logger.info` line that logs the summary, add:

```python
logger.info(
    "[data_refresh] Cache invalidated. Stats: %s",
    signal_cache.stats(),
)
```

**EOD refresh:** The EOD function should call `signal_cache.clear()` at the end of the function body, before the final summary log. This formalizes the full cache flush after daily data is refreshed. Add the same stats log line.

### Design Rationale

- **Per-symbol invalidation during intraday** (not `clear()`): If the universe has 50 symbols and only 20 are watched, invalidating all 50 causes a thundering herd — every next signal request for any symbol triggers a full 22-collector run. Per-symbol invalidation means only the refreshed symbols pay the recomputation cost.

- **Invalidate after write, not before:** If the write fails, the cached brief (even if somewhat stale) is still built on real data. Invalidating before a failed write leaves the cache empty, forcing a recomputation that will just read the same stale DB data anyway.

- **Known race condition:** A narrow window exists where the signal engine starts computing a brief for AAPL, the refresh cycle writes new AAPL data and invalidates the cache, and then the signal engine finishes and re-caches its (now slightly stale) brief. This window is seconds long. The prior situation was up to 55 minutes of guaranteed staleness. This is strictly better and accepted as-is.

### Collecting Refreshed Symbols for Step 1

The bulk quotes step iterates rows but only increments a counter. To know which symbols to invalidate, collect them into a list. The change: maintain a `refreshed_symbols` list alongside the existing counter, append each symbol as it is counted, then invalidate the list after the loop.

### Error Handling

No new error paths. `signal_cache.invalidate()` is a dict delete — it cannot fail. `signal_cache.stats()` returns a dict — it cannot fail. Both are called only after successful writes, so they do not affect the existing error-handling flow.
