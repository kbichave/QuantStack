# Section 03: Business Calendar Utility

## Purpose

The compliance layer (PDT enforcement, wash sale tracking) and the algo scheduler (TWAP/VWAP market-hours scheduling) need exchange-aware date arithmetic. This section creates `src/quantstack/execution/compliance/calendar.py` -- a thin, compliance-focused facade over the existing `TradingCalendar` in `src/quantstack/core/core/calendar.py`.

The existing `TradingCalendar` already wraps `exchange_calendars` and provides `is_trading_day`, `get_trading_days`, `next_trading_day`, `prev_trading_day`, `add_trading_days`, `trading_days_between`, and `get_session_times`. This section does NOT reimplement any of that. It adds three compliance-specific operations that downstream sections need:

1. **Rolling business-day window** -- PDT needs "the last 5 business days" from a reference date.
2. **Calendar-day offset with month boundary handling** -- wash sale needs "sell_date + 30 calendar days" that correctly spans across month boundaries (trivial with `timedelta`, but worth a named function for clarity and testability).
3. **Market hours check** -- the algo scheduler needs to know if a given datetime falls within market hours, and the compliance layer needs to determine which trading day a fill belongs to (a fill at 9:31 AM belongs to today; a fill processed at 5:00 PM after-hours still belongs to today's session).

## Dependency

- **Depends on:** section-01-schema-foundation (the `execution/compliance/` package must exist as a directory with `__init__.py`)
- **Blocks:** section-04-sec-compliance (PDTChecker and WashSaleTracker import from this module)

## Existing Code

The existing calendar lives at `src/quantstack/core/core/calendar.py` and exposes:

```python
from quantstack.core.core.calendar import TradingCalendar, get_default_calendar

cal = get_default_calendar()  # cached singleton, default exchange = NYSE
cal.is_trading_day(some_date)
cal.get_trading_days(start, end)
cal.add_trading_days(dt, n)       # add n trading days (negative OK)
cal.trading_days_between(start, end)  # count trading days (exclusive endpoints)
cal.get_session_times(dt)         # returns (open_time, close_time) tuple
cal.next_trading_day(dt)
cal.prev_trading_day(dt)
```

The `exchange_calendars` package is already a project dependency.

## Tests

File: `tests/unit/execution/compliance/test_calendar.py`

These tests validate the compliance calendar facade. They should use the real `exchange_calendars` backend (not mocks) so that holiday data is accurate.

```python
# --- Rolling business day window ---

# Test: rolling_business_day_window(date(2026, 4, 6), n=5) returns 5 trading days
#   ending on or before April 6 (a Monday). The window should skip weekends
#   and any holidays in the range.

# Test: rolling_business_day_window called on a weekend date uses the
#   preceding Friday as the anchor (the window ends on the last trading day
#   at or before the reference date).

# Test: rolling_business_day_window(date(2026, 1, 2), n=5) correctly handles
#   the New Year's Day holiday (Jan 1) -- the window should contain only
#   trading days, meaning it reaches further back in calendar time.

# Test: rolling_business_day_window with n=1 returns a single-element list
#   containing the most recent trading day at or before the reference date.

# --- Calendar day offset ---

# Test: calendar_day_offset(date(2026, 3, 15), 30) returns date(2026, 4, 14)

# Test: calendar_day_offset(date(2026, 1, 31), 30) returns date(2026, 3, 2)
#   -- correctly spans February.

# Test: calendar_day_offset(date(2026, 12, 15), 30) returns date(2027, 1, 14)
#   -- correctly spans year boundary.

# --- Market hours ---

# Test: is_during_market_hours(datetime(2026, 4, 6, 10, 30), "NYSE") returns True
#   (10:30 AM on a Monday, within 9:30-16:00 session)

# Test: is_during_market_hours(datetime(2026, 4, 6, 9, 29), "NYSE") returns False
#   (one minute before open)

# Test: is_during_market_hours(datetime(2026, 4, 6, 16, 0), "NYSE") returns False
#   (market close is exclusive -- 16:00 is not "during" hours)

# Test: is_during_market_hours on a weekend returns False regardless of time

# Test: is_during_market_hours on a holiday returns False regardless of time

# --- Trading day for timestamp ---

# Test: trading_day_for(datetime(2026, 4, 6, 10, 30)) returns date(2026, 4, 6)
#   -- fill during market hours belongs to that day's session.

# Test: trading_day_for(datetime(2026, 4, 6, 17, 0)) returns date(2026, 4, 6)
#   -- after-hours fill still belongs to the same session date.

# Test: trading_day_for(datetime(2026, 4, 6, 4, 0)) returns date(2026, 4, 3)
#   -- pre-market before 4:00 AM (or whatever cutover is chosen) could belong
#   to the previous session. The implementation should define a session cutover
#   hour (e.g., 4:00 AM ET) below which timestamps are attributed to the
#   previous trading day.

# --- 5 business days from Monday = next Monday ---

# Test: add 5 trading days to Monday 2026-04-06 yields Monday 2026-04-13
#   (assuming no holidays in that week). This delegates to
#   TradingCalendar.add_trading_days and is a smoke test for the facade.

# --- 30 calendar days correctly spans months ---

# Test: wash_sale_window_end(date(2026, 3, 1)) returns date(2026, 3, 31)
#   This is just calendar_day_offset(dt, 30) with a domain-specific name.
```

## Implementation

File: `src/quantstack/execution/compliance/calendar.py`

The module should expose the following functions. All functions delegate to the existing `TradingCalendar` singleton from `get_default_calendar()`.

### `rolling_business_day_window(reference_date: date, n: int, exchange: str = "NYSE") -> list[date]`

Returns the last `n` trading days ending on or before `reference_date`. If `reference_date` is not a trading day, the window ends on the most recent trading day before it.

Implementation approach: start from `reference_date`, walk backward using `prev_trading_day` until `n` dates are collected. Return them in chronological order (oldest first).

### `calendar_day_offset(dt: date, days: int) -> date`

Returns `dt + timedelta(days=days)`. This is a trivial wrapper, but it gives a named, testable function for the "30 calendar days" wash sale window and avoids scattering raw `timedelta` arithmetic across compliance code.

### `wash_sale_window_end(sell_date: date) -> date`

Returns `sell_date + 30 calendar days`. A domain alias for `calendar_day_offset(sell_date, 30)`.

### `is_during_market_hours(dt: datetime, exchange: str = "NYSE") -> bool`

Returns True if `dt` falls on a trading day AND the time component is within the exchange's session hours (open inclusive, close exclusive). Uses `TradingCalendar.is_trading_day()` and `TradingCalendar.get_session_times()`.

### `trading_day_for(dt: datetime, exchange: str = "NYSE", session_cutover_hour: int = 4) -> date`

Determines which trading session a timestamp belongs to. If `dt.hour < session_cutover_hour`, the timestamp is attributed to the previous trading day (handles overnight/pre-market fills). Otherwise, it belongs to `dt.date()`'s session. If the resulting date is not a trading day, returns the most recent trading day before it.

This is critical for PDT: a fill must be associated with the correct business day to determine whether an open-and-close happened on the "same day."

### Re-exports

The module should also re-export `get_default_calendar` for convenience, so compliance code can do:

```python
from quantstack.execution.compliance.calendar import (
    rolling_business_day_window,
    wash_sale_window_end,
    trading_day_for,
    get_default_calendar,
)
```

## Package Setup

Section 01 (schema-foundation) creates the `src/quantstack/execution/compliance/` directory and its `__init__.py`. This section adds `calendar.py` into that package. If implementing before section 01 is complete, create the directory and `__init__.py` as a prerequisite.

The `__init__.py` for the compliance package should export the calendar functions:

```python
# src/quantstack/execution/compliance/__init__.py
from quantstack.execution.compliance.calendar import (
    rolling_business_day_window,
    calendar_day_offset,
    wash_sale_window_end,
    is_during_market_hours,
    trading_day_for,
)
```

## Design Decisions

**Why a facade instead of using TradingCalendar directly?** Three reasons: (1) compliance code needs domain-named functions (`wash_sale_window_end`, `trading_day_for`) that don't belong on a general-purpose calendar class; (2) the `session_cutover_hour` logic for attributing after-hours fills to trading days is compliance-specific; (3) a single import path (`execution.compliance.calendar`) keeps the compliance module self-contained.

**Why not extend TradingCalendar?** The existing calendar is in `core/core/` and serves the entire system. Adding compliance-specific methods there would violate single responsibility and create a dependency from core on execution-layer concepts.

**Why real exchange_calendars in tests, not mocks?** The whole point of this module is accurate holiday handling. Mocking the calendar backend would test the test, not the behavior. The `exchange_calendars` package is fast enough for unit tests (calendar loading is cached).

**Session cutover at 4:00 AM ET.** This is the conventional pre-market open time. Fills between midnight and 4:00 AM are attributed to the previous trading day. Fills from 4:00 AM onward belong to the current date's session. The cutover hour is configurable per call to support future exchanges with different conventions.
