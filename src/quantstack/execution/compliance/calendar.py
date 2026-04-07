"""
Business calendar utilities for the execution/compliance layer.

Wraps :func:`quantstack.core.core.calendar.get_default_calendar` to provide
rolling-window helpers, wash-sale date math, and market-hours awareness
needed by downstream compliance and TCA modules.
"""

from datetime import date, datetime, timedelta, timezone

from quantstack.core.core.calendar import (
    EXCHANGE_CODES,
    TradingCalendar,
    get_default_calendar,
)

import exchange_calendars as xcals

# Re-export so callers can get the singleton from this module.
__all__ = [
    "get_default_calendar",
    "rolling_business_day_window",
    "calendar_day_offset",
    "wash_sale_window_end",
    "is_during_market_hours",
    "trading_day_for",
]


def rolling_business_day_window(
    reference_date: date,
    n: int,
    exchange: str = "NYSE",
) -> list[date]:
    """Return the last *n* trading days ending on or before *reference_date*.

    If *reference_date* is not a trading day the window ends on the most
    recent trading day before it.  The returned list is in chronological
    order (oldest first).
    """
    cal = get_default_calendar()

    # Find the anchor: reference_date itself if it is a session, else the
    # previous trading day.
    if cal.is_trading_day(reference_date, exchange):
        anchor = reference_date
    else:
        anchor = cal.prev_trading_day(reference_date, exchange)

    # Walk backwards n-1 more trading days from anchor.
    # add_trading_days with a negative offset is cleaner than a manual loop
    # but gives us only the start date; we still need the full list.
    start = cal.add_trading_days(anchor, -(n - 1), exchange)
    return cal.get_trading_days(start, anchor, exchange)


def calendar_day_offset(dt: date, days: int) -> date:
    """Return *dt* shifted by *days* calendar days."""
    return dt + timedelta(days=days)


def wash_sale_window_end(sell_date: date) -> date:
    """Return the last day of the IRS 30-calendar-day wash-sale window."""
    return sell_date + timedelta(days=30)


def is_during_market_hours(
    dt: datetime,
    exchange: str = "NYSE",
) -> bool:
    """Return ``True`` if *dt* falls on a trading day AND within session hours.

    Session boundaries come from ``exchange_calendars`` (UTC-aware).
    The open is inclusive, the close is exclusive.

    *dt* may be naive (assumed UTC) or timezone-aware.
    """
    cal = get_default_calendar()
    check_date = dt.date()

    if not cal.is_trading_day(check_date, exchange):
        return False

    # Get exchange_calendars calendar for precise session times.
    xcal_code = EXCHANGE_CODES[exchange]
    xcal = xcals.get_calendar(xcal_code)

    session_open = xcal.session_open(check_date)   # tz-aware UTC
    session_close = xcal.session_close(check_date)  # tz-aware UTC

    # Ensure dt is tz-aware (treat naive as UTC).
    if dt.tzinfo is None:
        dt_aware = dt.replace(tzinfo=timezone.utc)
    else:
        dt_aware = dt

    return session_open <= dt_aware < session_close


def trading_day_for(
    dt: datetime,
    exchange: str = "NYSE",
    session_cutover_hour: int = 4,
) -> date:
    """Determine which trading session a timestamp belongs to.

    If ``dt.hour < session_cutover_hour`` the timestamp is attributed to
    the *previous* trading day (e.g. 3 AM activity belongs to the prior
    session).  Otherwise, the date component of *dt* is used — if that
    date is not a trading day, the most recent trading day is returned.
    """
    cal = get_default_calendar()

    if dt.hour < session_cutover_hour:
        # Pre-cutover: belongs to the previous trading session.
        candidate = dt.date() - timedelta(days=1)
        if cal.is_trading_day(candidate, exchange):
            return candidate
        return cal.prev_trading_day(candidate, exchange)

    # Normal path: use dt's own date, snapping to most recent session.
    candidate = dt.date()
    if cal.is_trading_day(candidate, exchange):
        return candidate
    return cal.prev_trading_day(candidate, exchange)
