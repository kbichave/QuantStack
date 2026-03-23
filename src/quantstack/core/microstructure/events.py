"""
Market event calendar for trading restrictions.

Avoids trading around high-impact events that destroy MR edge.

EventType, MarketEvent, and EventCalendar now live in shared.event_calendar (L0)
so lower layers can import them without upward dependency violations.
This module re-exports them for backward compatibility.
"""

from datetime import datetime, time, timedelta

import pandas as pd

# Re-export canonical types from shared (L0)
from quantstack.shared.event_calendar import (  # noqa: F401
    EventCalendar,
    EventType,
    MarketEvent,
)


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
        self._holidays: set[datetime] = set()
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
    ) -> list[datetime]:
        """Get list of trading days between two dates."""
        days = []
        current = start

        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)

        return days
