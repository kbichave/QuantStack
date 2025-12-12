"""
Market event calendar for trading restrictions.

Avoids trading around high-impact events that destroy MR edge.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import List, Optional, Set
from enum import Enum
import pandas as pd
from loguru import logger


class EventType(Enum):
    """Types of market events."""

    FOMC = "FOMC"
    CPI = "CPI"
    NFP = "NFP"  # Non-Farm Payrolls
    PPI = "PPI"
    GDP = "GDP"
    EARNINGS = "EARNINGS"
    OPEX = "OPEX"  # Options expiration
    QUAD_WITCH = "QUAD_WITCH"
    MARKET_OPEN = "MARKET_OPEN"
    MARKET_CLOSE = "MARKET_CLOSE"
    AUCTION = "AUCTION"
    HOLIDAY = "HOLIDAY"


@dataclass
class MarketEvent:
    """A specific market event."""

    event_type: EventType
    timestamp: datetime
    symbol: Optional[str] = None  # None = affects all symbols
    impact: str = "HIGH"  # HIGH, MEDIUM, LOW
    blackout_hours_before: int = 1
    blackout_hours_after: int = 2

    def is_in_blackout(self, check_time: datetime) -> bool:
        """Check if a time falls within the event blackout window."""
        blackout_start = self.timestamp - timedelta(hours=self.blackout_hours_before)
        blackout_end = self.timestamp + timedelta(hours=self.blackout_hours_after)
        return blackout_start <= check_time <= blackout_end


class EventCalendar:
    """
    Economic and market event calendar.

    Provides:
    - Event lookup by date
    - Blackout window checking
    - Earnings calendar integration
    """

    # Standard blackout rules by event type
    BLACKOUT_RULES = {
        EventType.FOMC: {"before": 2, "after": 4},
        EventType.CPI: {"before": 1, "after": 2},
        EventType.NFP: {"before": 1, "after": 2},
        EventType.PPI: {"before": 1, "after": 1},
        EventType.GDP: {"before": 1, "after": 1},
        EventType.EARNINGS: {"before": 4, "after": 2},  # Typically after close
        EventType.OPEX: {"before": 2, "after": 2},
        EventType.QUAD_WITCH: {"before": 4, "after": 2},
        EventType.MARKET_OPEN: {"before": 0, "after": 1},
        EventType.MARKET_CLOSE: {"before": 1, "after": 0},
    }

    def __init__(self):
        """Initialize event calendar."""
        self._events: List[MarketEvent] = []
        self._load_recurring_events()

    def _load_recurring_events(self) -> None:
        """Load recurring events (open/close auctions)."""
        # Market open and close are handled separately via trading windows
        pass

    def add_event(
        self,
        event_type: EventType,
        timestamp: datetime,
        symbol: Optional[str] = None,
        impact: str = "HIGH",
    ) -> None:
        """
        Add an event to the calendar.

        Args:
            event_type: Type of event
            timestamp: Event time
            symbol: Affected symbol (None = all)
            impact: Impact level
        """
        rules = self.BLACKOUT_RULES.get(event_type, {"before": 1, "after": 1})

        event = MarketEvent(
            event_type=event_type,
            timestamp=timestamp,
            symbol=symbol,
            impact=impact,
            blackout_hours_before=rules["before"],
            blackout_hours_after=rules["after"],
        )

        self._events.append(event)

    def add_fomc_dates(self, dates: List[datetime]) -> None:
        """Add FOMC meeting dates."""
        for date in dates:
            # FOMC announcements are typically at 2:00 PM ET
            timestamp = datetime.combine(date.date(), time(14, 0))
            self.add_event(EventType.FOMC, timestamp)

    def add_cpi_dates(self, dates: List[datetime]) -> None:
        """Add CPI release dates."""
        for date in dates:
            # CPI is released at 8:30 AM ET
            timestamp = datetime.combine(date.date(), time(8, 30))
            self.add_event(EventType.CPI, timestamp)

    def add_nfp_dates(self, dates: List[datetime]) -> None:
        """Add Non-Farm Payrolls dates (first Friday of month)."""
        for date in dates:
            timestamp = datetime.combine(date.date(), time(8, 30))
            self.add_event(EventType.NFP, timestamp)

    def add_earnings(
        self,
        symbol: str,
        date: datetime,
        is_before_open: bool = False,
    ) -> None:
        """Add earnings announcement."""
        if is_before_open:
            timestamp = datetime.combine(date.date(), time(8, 0))
        else:
            timestamp = datetime.combine(date.date(), time(16, 30))

        self.add_event(EventType.EARNINGS, timestamp, symbol=symbol)

    def add_opex_dates(self, year: int) -> None:
        """Add options expiration dates (third Friday of each month)."""
        for month in range(1, 13):
            # Find third Friday
            first_day = datetime(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)

            timestamp = datetime.combine(third_friday.date(), time(16, 0))

            # Check for quad witching (March, June, Sept, Dec)
            if month in [3, 6, 9, 12]:
                self.add_event(EventType.QUAD_WITCH, timestamp)
            else:
                self.add_event(EventType.OPEX, timestamp)

    def is_blackout(
        self,
        check_time: datetime,
        symbol: Optional[str] = None,
    ) -> tuple[bool, Optional[MarketEvent]]:
        """
        Check if a time is in a blackout window.

        Args:
            check_time: Time to check
            symbol: Symbol to check (checks general events if None)

        Returns:
            Tuple of (is_blackout, causing_event)
        """
        for event in self._events:
            # Check symbol-specific events
            if event.symbol is not None and event.symbol != symbol:
                continue

            if event.is_in_blackout(check_time):
                return True, event

        return False, None

    def get_events_for_date(self, date: datetime) -> List[MarketEvent]:
        """Get all events for a specific date."""
        target_date = date.date()
        return [e for e in self._events if e.timestamp.date() == target_date]

    def get_upcoming_events(
        self,
        from_time: datetime,
        hours_ahead: int = 24,
    ) -> List[MarketEvent]:
        """Get events within the next N hours."""
        end_time = from_time + timedelta(hours=hours_ahead)
        return [e for e in self._events if from_time <= e.timestamp <= end_time]

    def get_blackout_series(
        self,
        index: pd.DatetimeIndex,
        symbol: Optional[str] = None,
    ) -> pd.Series:
        """
        Get blackout status for a datetime index.

        Args:
            index: DatetimeIndex to check
            symbol: Symbol to check

        Returns:
            Boolean Series (True = in blackout)
        """
        blackouts = []

        for ts in index:
            is_bo, _ = self.is_blackout(ts.to_pydatetime(), symbol)
            blackouts.append(is_bo)

        return pd.Series(blackouts, index=index, name="is_blackout")


class TradingCalendar:
    """
    Trading calendar for market hours and holidays.
    """

    # US Market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    # Pre/post market
    PRE_MARKET_OPEN = time(4, 0)
    POST_MARKET_CLOSE = time(20, 0)

    # US Market holidays (approximate - should use proper calendar)
    HOLIDAYS_2024 = {
        datetime(2024, 1, 1),  # New Year's Day
        datetime(2024, 1, 15),  # MLK Day
        datetime(2024, 2, 19),  # Presidents Day
        datetime(2024, 3, 29),  # Good Friday
        datetime(2024, 5, 27),  # Memorial Day
        datetime(2024, 6, 19),  # Juneteenth
        datetime(2024, 7, 4),  # Independence Day
        datetime(2024, 9, 2),  # Labor Day
        datetime(2024, 11, 28),  # Thanksgiving
        datetime(2024, 12, 25),  # Christmas
    }

    def __init__(self):
        """Initialize trading calendar."""
        self._holidays: Set[datetime] = set()
        self._load_holidays()

    def _load_holidays(self) -> None:
        """Load market holidays."""
        self._holidays.update(self.HOLIDAYS_2024)

    def is_trading_day(self, date: datetime) -> bool:
        """Check if a date is a trading day."""
        # Check weekend
        if date.weekday() >= 5:
            return False

        # Check holiday
        if date.date() in {h.date() for h in self._holidays}:
            return False

        return True

    def is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during regular market hours."""
        if not self.is_trading_day(timestamp):
            return False

        t = timestamp.time()
        return self.MARKET_OPEN <= t < self.MARKET_CLOSE

    def get_trading_days(
        self,
        start: datetime,
        end: datetime,
    ) -> List[datetime]:
        """Get list of trading days between two dates."""
        days = []
        current = start

        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)

        return days
