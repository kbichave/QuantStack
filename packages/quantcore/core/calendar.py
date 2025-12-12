"""
Trading calendar system for QuantCore.

Provides exchange-aware calendar functionality for handling trading days,
holidays, and session times.

Uses the `exchange_calendars` package when available, falls back to
basic implementation otherwise.

Supported exchanges:
- NYSE (New York Stock Exchange)
- NASDAQ
- CME (Chicago Mercantile Exchange) - Energy futures
- ICE (Intercontinental Exchange) - Brent futures

Usage:
    from quantcore.core.calendar import TradingCalendar

    cal = TradingCalendar()

    # Check if a date is a trading day
    if cal.is_trading_day(date(2024, 1, 1), "NYSE"):
        print("Market is open")

    # Get all trading days in a range
    days = cal.get_trading_days(date(2024, 1, 1), date(2024, 12, 31), "NYSE")
"""

from datetime import date, datetime, time, timedelta
from typing import List, Optional, Tuple, Dict, Set
from functools import lru_cache

from loguru import logger

from quantcore.core.errors import CalendarError


# Try to import exchange_calendars, fall back to basic implementation
try:
    import exchange_calendars as xcals

    XCALS_AVAILABLE = True
except ImportError:
    XCALS_AVAILABLE = False
    logger.warning(
        "exchange_calendars not installed. Using basic calendar implementation. "
        "Install with: pip install exchange_calendars"
    )


# Exchange name mappings to exchange_calendars codes
EXCHANGE_CODES: Dict[str, str] = {
    "NYSE": "XNYS",
    "NASDAQ": "XNAS",
    "CME": "CMES",  # CME equity futures
    "ICE": "IEPA",  # ICE Europe
    "LSE": "XLON",
    "TSE": "XTKS",
}

# Supported exchanges
SUPPORTED_EXCHANGES: Set[str] = {"NYSE", "NASDAQ", "CME", "ICE"}

# US market holidays (basic fallback)
US_HOLIDAYS_2024: Set[date] = {
    date(2024, 1, 1),  # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),  # Independence Day
    date(2024, 9, 2),  # Labor Day
    date(2024, 11, 28),  # Thanksgiving
    date(2024, 12, 25),  # Christmas
}

US_HOLIDAYS_2025: Set[date] = {
    date(2025, 1, 1),  # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),  # Independence Day
    date(2025, 9, 1),  # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
}

# Basic US holidays (static fallback)
BASIC_US_HOLIDAYS: Set[date] = US_HOLIDAYS_2024 | US_HOLIDAYS_2025

# Standard session times by exchange
SESSION_TIMES: Dict[str, Tuple[time, time]] = {
    "NYSE": (time(9, 30), time(16, 0)),
    "NASDAQ": (time(9, 30), time(16, 0)),
    "CME": (time(17, 0), time(16, 0)),  # 23-hour session
    "ICE": (time(8, 0), time(18, 0)),  # ICE Europe hours (London time)
}


class TradingCalendar:
    """
    Trading calendar for exchange-aware date operations.

    Provides methods to:
    - Check if a date is a trading day
    - Get lists of trading days in a range
    - Navigate to next/previous trading day
    - Get session times for an exchange

    Uses `exchange_calendars` package when available for accurate
    holiday data, with a basic fallback for US markets.
    """

    def __init__(self, default_exchange: str = "NYSE"):
        """
        Initialize trading calendar.

        Args:
            default_exchange: Default exchange to use when not specified
        """
        if default_exchange not in SUPPORTED_EXCHANGES:
            raise CalendarError(
                f"Unsupported exchange: {default_exchange}",
                supported=list(SUPPORTED_EXCHANGES),
            )

        self.default_exchange = default_exchange
        self._calendars: Dict[str, any] = {}

        # Load calendars if exchange_calendars is available
        if XCALS_AVAILABLE:
            self._load_calendars()

    def _load_calendars(self) -> None:
        """Load exchange calendars from exchange_calendars package."""
        for exchange, code in EXCHANGE_CODES.items():
            if exchange in SUPPORTED_EXCHANGES:
                try:
                    self._calendars[exchange] = xcals.get_calendar(code)
                except Exception as e:
                    logger.warning(f"Could not load calendar for {exchange}: {e}")

    def is_trading_day(
        self,
        dt: date,
        exchange: Optional[str] = None,
    ) -> bool:
        """
        Check if a date is a trading day.

        Args:
            dt: Date to check
            exchange: Exchange name (default: self.default_exchange)

        Returns:
            True if the exchange is open on this date
        """
        exchange = exchange or self.default_exchange
        self._validate_exchange(exchange)

        # Convert datetime to date if needed
        if isinstance(dt, datetime):
            dt = dt.date()

        # Weekend check (applies to all exchanges)
        if dt.weekday() >= 5:
            return False

        # Use exchange_calendars if available
        if XCALS_AVAILABLE and exchange in self._calendars:
            cal = self._calendars[exchange]
            try:
                return cal.is_session(dt)
            except Exception:
                # Fall through to basic implementation
                pass

        # Basic fallback: check US holidays
        if exchange in ("NYSE", "NASDAQ"):
            return dt not in BASIC_US_HOLIDAYS

        # CME and ICE have similar holidays to NYSE
        return dt not in BASIC_US_HOLIDAYS

    def get_trading_days(
        self,
        start: date,
        end: date,
        exchange: Optional[str] = None,
    ) -> List[date]:
        """
        Get all trading days in a date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            exchange: Exchange name (default: self.default_exchange)

        Returns:
            List of trading dates in the range
        """
        exchange = exchange or self.default_exchange
        self._validate_exchange(exchange)

        # Convert datetime to date if needed
        if isinstance(start, datetime):
            start = start.date()
        if isinstance(end, datetime):
            end = end.date()

        if start > end:
            raise CalendarError(
                "Start date must be before or equal to end date",
                start=str(start),
                end=str(end),
            )

        # Use exchange_calendars if available
        if XCALS_AVAILABLE and exchange in self._calendars:
            cal = self._calendars[exchange]
            try:
                sessions = cal.sessions_in_range(start, end)
                return [s.date() for s in sessions]
            except Exception as e:
                logger.debug(f"exchange_calendars error, using fallback: {e}")

        # Basic fallback
        trading_days = []
        current = start
        while current <= end:
            if self.is_trading_day(current, exchange):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

    def next_trading_day(
        self,
        dt: date,
        exchange: Optional[str] = None,
    ) -> date:
        """
        Get the next trading day after a given date.

        Args:
            dt: Reference date
            exchange: Exchange name (default: self.default_exchange)

        Returns:
            Next trading day
        """
        exchange = exchange or self.default_exchange
        self._validate_exchange(exchange)

        if isinstance(dt, datetime):
            dt = dt.date()

        # Use exchange_calendars if available
        if XCALS_AVAILABLE and exchange in self._calendars:
            cal = self._calendars[exchange]
            try:
                next_session = cal.next_session(dt)
                return next_session.date()
            except Exception:
                pass

        # Basic fallback
        next_day = dt + timedelta(days=1)
        max_lookforward = 10  # Prevent infinite loop
        for _ in range(max_lookforward):
            if self.is_trading_day(next_day, exchange):
                return next_day
            next_day += timedelta(days=1)

        raise CalendarError(
            f"Could not find next trading day within {max_lookforward} days",
            from_date=str(dt),
            exchange=exchange,
        )

    def prev_trading_day(
        self,
        dt: date,
        exchange: Optional[str] = None,
    ) -> date:
        """
        Get the previous trading day before a given date.

        Args:
            dt: Reference date
            exchange: Exchange name (default: self.default_exchange)

        Returns:
            Previous trading day
        """
        exchange = exchange or self.default_exchange
        self._validate_exchange(exchange)

        if isinstance(dt, datetime):
            dt = dt.date()

        # Use exchange_calendars if available
        if XCALS_AVAILABLE and exchange in self._calendars:
            cal = self._calendars[exchange]
            try:
                prev_session = cal.previous_session(dt)
                return prev_session.date()
            except Exception:
                pass

        # Basic fallback
        prev_day = dt - timedelta(days=1)
        max_lookback = 10  # Prevent infinite loop
        for _ in range(max_lookback):
            if self.is_trading_day(prev_day, exchange):
                return prev_day
            prev_day -= timedelta(days=1)

        raise CalendarError(
            f"Could not find previous trading day within {max_lookback} days",
            from_date=str(dt),
            exchange=exchange,
        )

    def get_session_times(
        self,
        dt: date,
        exchange: Optional[str] = None,
    ) -> Tuple[time, time]:
        """
        Get trading session open and close times.

        Args:
            dt: Date (for potential early close detection)
            exchange: Exchange name (default: self.default_exchange)

        Returns:
            Tuple of (open_time, close_time) in exchange local time
        """
        exchange = exchange or self.default_exchange
        self._validate_exchange(exchange)

        # Use exchange_calendars for early close detection if available
        if XCALS_AVAILABLE and exchange in self._calendars:
            cal = self._calendars[exchange]
            try:
                if isinstance(dt, datetime):
                    dt = dt.date()

                if cal.is_session(dt):
                    open_dt = cal.session_open(dt)
                    close_dt = cal.session_close(dt)
                    return (open_dt.time(), close_dt.time())
            except Exception:
                pass

        # Return standard session times
        return SESSION_TIMES.get(exchange, SESSION_TIMES["NYSE"])

    def trading_days_between(
        self,
        start: date,
        end: date,
        exchange: Optional[str] = None,
    ) -> int:
        """
        Count trading days between two dates.

        Args:
            start: Start date (exclusive)
            end: End date (exclusive)
            exchange: Exchange name

        Returns:
            Number of trading days between start and end
        """
        if start >= end:
            return 0

        # Get trading days and count (excluding endpoints)
        trading_days = self.get_trading_days(
            start + timedelta(days=1), end - timedelta(days=1), exchange
        )
        return len(trading_days)

    def add_trading_days(
        self,
        dt: date,
        days: int,
        exchange: Optional[str] = None,
    ) -> date:
        """
        Add a number of trading days to a date.

        Args:
            dt: Starting date
            days: Number of trading days to add (can be negative)
            exchange: Exchange name

        Returns:
            Resulting date after adding trading days
        """
        exchange = exchange or self.default_exchange

        if isinstance(dt, datetime):
            dt = dt.date()

        if days == 0:
            return dt

        direction = 1 if days > 0 else -1
        remaining = abs(days)
        current = dt

        while remaining > 0:
            current = current + timedelta(days=direction)
            if self.is_trading_day(current, exchange):
                remaining -= 1

        return current

    def _validate_exchange(self, exchange: str) -> None:
        """Validate exchange is supported."""
        if exchange not in SUPPORTED_EXCHANGES:
            raise CalendarError(
                f"Unsupported exchange: {exchange}",
                supported=list(SUPPORTED_EXCHANGES),
            )

    @staticmethod
    def get_supported_exchanges() -> List[str]:
        """Get list of supported exchanges."""
        return list(SUPPORTED_EXCHANGES)


# Module-level convenience function
@lru_cache(maxsize=1)
def get_default_calendar() -> TradingCalendar:
    """Get a cached default trading calendar instance."""
    return TradingCalendar()
