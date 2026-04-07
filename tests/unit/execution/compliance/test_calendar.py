"""Tests for quantstack.execution.compliance.calendar — business calendar utilities.

Uses real exchange_calendars data (no mocks) so assertions match actual NYSE
holiday/session schedules.
"""

from datetime import date, datetime, timezone

import pytest

from quantstack.execution.compliance.calendar import (
    calendar_day_offset,
    get_default_calendar,
    is_during_market_hours,
    rolling_business_day_window,
    trading_day_for,
    wash_sale_window_end,
)


# ---------------------------------------------------------------------------
# rolling_business_day_window
# ---------------------------------------------------------------------------


class TestRollingBusinessDayWindow:
    def test_weekday_returns_n_trading_days(self):
        """2026-04-06 is a Monday — should return 5 trading days ending on it."""
        result = rolling_business_day_window(date(2026, 4, 6), n=5)
        assert len(result) == 5
        assert result[-1] == date(2026, 4, 6)
        # All results should be weekdays and in ascending order
        for i, d in enumerate(result):
            assert d.weekday() < 5
            if i > 0:
                assert d > result[i - 1]

    def test_weekend_uses_preceding_trading_day(self):
        """2026-04-04 is Saturday AND 04-03 is Good Friday (holiday).
        Window should anchor on Thursday 2026-04-02."""
        result = rolling_business_day_window(date(2026, 4, 4), n=3)
        assert len(result) == 3
        assert result[-1] == date(2026, 4, 2)

    def test_new_years_day_holiday(self):
        """2026-01-01 is a holiday — window should not include it."""
        result = rolling_business_day_window(date(2026, 1, 1), n=3)
        assert date(2026, 1, 1) not in result
        assert len(result) == 3
        # Anchor should be the last trading day of 2025 (Dec 31, 2025 is a Wed)
        assert result[-1] == date(2025, 12, 31)

    def test_n_equals_one(self):
        result = rolling_business_day_window(date(2026, 4, 6), n=1)
        assert result == [date(2026, 4, 6)]


# ---------------------------------------------------------------------------
# calendar_day_offset
# ---------------------------------------------------------------------------


class TestCalendarDayOffset:
    def test_basic_offset(self):
        assert calendar_day_offset(date(2026, 3, 15), 30) == date(2026, 4, 14)

    def test_spans_february(self):
        assert calendar_day_offset(date(2026, 1, 31), 30) == date(2026, 3, 2)

    def test_spans_year_boundary(self):
        assert calendar_day_offset(date(2026, 12, 15), 30) == date(2027, 1, 14)


# ---------------------------------------------------------------------------
# is_during_market_hours
# ---------------------------------------------------------------------------


class TestIsDuringMarketHours:
    def test_within_session(self):
        """Monday 10:30 ET → within NYSE session."""
        # 10:30 ET = 14:30 UTC
        dt = datetime(2026, 4, 6, 14, 30, tzinfo=timezone.utc)
        assert is_during_market_hours(dt) is True

    def test_before_open(self):
        """Monday 9:29 ET → before NYSE open."""
        # 9:29 ET = 13:29 UTC
        dt = datetime(2026, 4, 6, 13, 29, tzinfo=timezone.utc)
        assert is_during_market_hours(dt) is False

    def test_close_is_exclusive(self):
        """Monday 16:00 ET → close boundary is exclusive."""
        # 16:00 ET = 20:00 UTC
        dt = datetime(2026, 4, 6, 20, 0, tzinfo=timezone.utc)
        assert is_during_market_hours(dt) is False

    def test_weekend_is_false(self):
        """Saturday → always False regardless of time."""
        dt = datetime(2026, 4, 4, 14, 30, tzinfo=timezone.utc)
        assert is_during_market_hours(dt) is False

    def test_holiday_is_false(self):
        """New Year's Day 2026 → False."""
        dt = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
        assert is_during_market_hours(dt) is False


# ---------------------------------------------------------------------------
# trading_day_for
# ---------------------------------------------------------------------------


class TestTradingDayFor:
    def test_during_market_hours(self):
        """Midday Monday → same date."""
        dt = datetime(2026, 4, 6, 10, 30)
        assert trading_day_for(dt) == date(2026, 4, 6)

    def test_after_hours_same_day(self):
        """After close Monday → still Monday's session."""
        dt = datetime(2026, 4, 6, 17, 0)
        assert trading_day_for(dt) == date(2026, 4, 6)

    def test_pre_cutover_maps_to_previous_session(self):
        """Monday 3 AM (before cutover) → previous trading day.
        2026-04-03 is Good Friday so previous session is Thursday 04-02."""
        dt = datetime(2026, 4, 6, 3, 0)
        assert trading_day_for(dt) == date(2026, 4, 2)

    def test_weekend_after_cutover(self):
        """Saturday 10 AM → most recent trading day.
        2026-04-03 is Good Friday so previous session is Thursday 04-02."""
        dt = datetime(2026, 4, 4, 10, 0)
        assert trading_day_for(dt) == date(2026, 4, 2)


# ---------------------------------------------------------------------------
# wash_sale_window_end
# ---------------------------------------------------------------------------


class TestWashSaleWindowEnd:
    def test_basic(self):
        assert wash_sale_window_end(date(2026, 3, 1)) == date(2026, 3, 31)

    def test_year_boundary(self):
        assert wash_sale_window_end(date(2026, 12, 15)) == date(2027, 1, 14)


# ---------------------------------------------------------------------------
# get_default_calendar re-export
# ---------------------------------------------------------------------------


def test_get_default_calendar_returns_trading_calendar():
    cal = get_default_calendar()
    assert isinstance(cal, type(get_default_calendar()))
    assert hasattr(cal, "is_trading_day")
