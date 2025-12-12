"""
Tests for TradingCalendar.

Verifies:
1. Correct holiday detection for US markets
2. Trading day navigation
3. Session time retrieval
4. Edge cases and error handling
"""

from datetime import date, timedelta
import pytest

from quantcore.core.calendar import TradingCalendar, SUPPORTED_EXCHANGES
from quantcore.core.errors import CalendarError


class TestTradingCalendarBasics:
    """Test basic calendar functionality."""

    def test_create_calendar_default(self):
        """Can create calendar with default exchange."""
        cal = TradingCalendar()
        assert cal.default_exchange == "NYSE"

    def test_create_calendar_with_exchange(self):
        """Can create calendar with specified exchange."""
        cal = TradingCalendar(default_exchange="CME")
        assert cal.default_exchange == "CME"

    def test_create_calendar_unsupported_exchange(self):
        """Raises error for unsupported exchange."""
        with pytest.raises(CalendarError):
            TradingCalendar(default_exchange="INVALID")

    def test_supported_exchanges(self):
        """Can get list of supported exchanges."""
        exchanges = TradingCalendar.get_supported_exchanges()
        assert "NYSE" in exchanges
        assert "NASDAQ" in exchanges
        assert "CME" in exchanges
        assert "ICE" in exchanges


class TestTradingDayDetection:
    """Test trading day detection."""

    @pytest.fixture
    def calendar(self):
        return TradingCalendar()

    def test_weekday_is_trading_day(self, calendar):
        """Regular weekday is a trading day."""
        # A regular Tuesday in 2024
        assert calendar.is_trading_day(date(2024, 3, 5)) is True

    def test_saturday_not_trading_day(self, calendar):
        """Saturday is not a trading day."""
        assert calendar.is_trading_day(date(2024, 3, 9)) is False

    def test_sunday_not_trading_day(self, calendar):
        """Sunday is not a trading day."""
        assert calendar.is_trading_day(date(2024, 3, 10)) is False

    def test_christmas_not_trading_day(self, calendar):
        """Christmas is not a trading day."""
        assert calendar.is_trading_day(date(2024, 12, 25)) is False

    def test_new_years_not_trading_day(self, calendar):
        """New Year's Day is not a trading day."""
        assert calendar.is_trading_day(date(2024, 1, 1)) is False

    def test_thanksgiving_not_trading_day(self, calendar):
        """Thanksgiving is not a trading day."""
        assert calendar.is_trading_day(date(2024, 11, 28)) is False

    def test_july_4th_not_trading_day(self, calendar):
        """Independence Day is not a trading day."""
        assert calendar.is_trading_day(date(2024, 7, 4)) is False

    def test_mlk_day_not_trading_day(self, calendar):
        """MLK Day is not a trading day."""
        assert calendar.is_trading_day(date(2024, 1, 15)) is False

    def test_day_after_christmas_is_trading_day(self, calendar):
        """December 26 is typically a trading day."""
        assert calendar.is_trading_day(date(2024, 12, 26)) is True


class TestTradingDayNavigation:
    """Test next/previous trading day navigation."""

    @pytest.fixture
    def calendar(self):
        return TradingCalendar()

    def test_next_trading_day_from_friday(self, calendar):
        """Next trading day from Friday is Monday."""
        friday = date(2024, 3, 8)
        next_day = calendar.next_trading_day(friday)
        assert next_day == date(2024, 3, 11)  # Monday

    def test_next_trading_day_from_weekday(self, calendar):
        """Next trading day from Tuesday is Wednesday."""
        tuesday = date(2024, 3, 5)
        next_day = calendar.next_trading_day(tuesday)
        assert next_day == date(2024, 3, 6)  # Wednesday

    def test_next_trading_day_from_holiday(self, calendar):
        """Next trading day from holiday skips the holiday."""
        christmas_eve = date(2024, 12, 24)
        next_day = calendar.next_trading_day(christmas_eve)
        assert next_day == date(2024, 12, 26)  # Skip Christmas

    def test_prev_trading_day_from_monday(self, calendar):
        """Previous trading day from Monday is Friday."""
        monday = date(2024, 3, 11)
        prev_day = calendar.prev_trading_day(monday)
        assert prev_day == date(2024, 3, 8)  # Friday

    def test_prev_trading_day_from_weekday(self, calendar):
        """Previous trading day from Wednesday is Tuesday."""
        wednesday = date(2024, 3, 6)
        prev_day = calendar.prev_trading_day(wednesday)
        assert prev_day == date(2024, 3, 5)  # Tuesday

    def test_add_trading_days_positive(self, calendar):
        """Add positive trading days."""
        start = date(2024, 3, 4)  # Monday
        result = calendar.add_trading_days(start, 5)
        assert result == date(2024, 3, 11)  # Next Monday

    def test_add_trading_days_negative(self, calendar):
        """Add negative trading days (subtract)."""
        start = date(2024, 3, 11)  # Monday
        result = calendar.add_trading_days(start, -5)
        assert result == date(2024, 3, 4)  # Previous Monday

    def test_add_trading_days_zero(self, calendar):
        """Adding zero days returns same date."""
        start = date(2024, 3, 5)
        result = calendar.add_trading_days(start, 0)
        assert result == start


class TestTradingDaysRange:
    """Test getting trading days in a range."""

    @pytest.fixture
    def calendar(self):
        return TradingCalendar()

    def test_get_trading_days_one_week(self, calendar):
        """Get trading days for one week."""
        start = date(2024, 3, 4)  # Monday
        end = date(2024, 3, 8)  # Friday
        days = calendar.get_trading_days(start, end)

        assert len(days) == 5
        assert days[0] == date(2024, 3, 4)
        assert days[-1] == date(2024, 3, 8)

    def test_get_trading_days_excludes_weekend(self, calendar):
        """Trading days excludes weekend."""
        start = date(2024, 3, 8)  # Friday
        end = date(2024, 3, 11)  # Monday
        days = calendar.get_trading_days(start, end)

        assert len(days) == 2
        assert date(2024, 3, 9) not in days  # Saturday
        assert date(2024, 3, 10) not in days  # Sunday

    def test_get_trading_days_excludes_holidays(self, calendar):
        """Trading days excludes holidays."""
        start = date(2024, 12, 23)
        end = date(2024, 12, 27)
        days = calendar.get_trading_days(start, end)

        assert date(2024, 12, 25) not in days  # Christmas

    def test_get_trading_days_invalid_range(self, calendar):
        """Raises error for invalid date range."""
        with pytest.raises(CalendarError):
            calendar.get_trading_days(date(2024, 3, 10), date(2024, 3, 5))

    def test_trading_days_between(self, calendar):
        """Count trading days between dates."""
        start = date(2024, 3, 4)
        end = date(2024, 3, 11)
        count = calendar.trading_days_between(start, end)

        # Between Monday and next Monday (exclusive both ends)
        assert count == 4


class TestSessionTimes:
    """Test session time retrieval."""

    @pytest.fixture
    def calendar(self):
        return TradingCalendar()

    def test_get_nyse_session_times(self, calendar):
        """Get NYSE standard session times."""
        open_time, close_time = calendar.get_session_times(date(2024, 3, 5), "NYSE")

        assert open_time.hour == 9
        assert open_time.minute == 30
        assert close_time.hour == 16
        assert close_time.minute == 0

    def test_get_cme_session_times(self, calendar):
        """Get CME session times."""
        open_time, close_time = calendar.get_session_times(date(2024, 3, 5), "CME")

        # CME has different hours
        assert open_time is not None
        assert close_time is not None


class TestMultipleExchanges:
    """Test calendar with different exchanges."""

    def test_nyse_and_nasdaq_same_schedule(self):
        """NYSE and NASDAQ have same holidays."""
        cal = TradingCalendar()

        test_dates = [
            date(2024, 1, 1),  # New Year
            date(2024, 12, 25),  # Christmas
            date(2024, 7, 4),  # July 4th
        ]

        for dt in test_dates:
            assert cal.is_trading_day(dt, "NYSE") == cal.is_trading_day(dt, "NASDAQ")

    def test_validate_exchange(self):
        """Validation catches unsupported exchange."""
        cal = TradingCalendar()

        with pytest.raises(CalendarError):
            cal.is_trading_day(date(2024, 3, 5), "INVALID")
