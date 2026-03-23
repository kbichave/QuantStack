# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Market event calendar — zero intra-project dependencies.

Moved from core.microstructure.events so that lower layers (data) can
import EventCalendar without upward dependency violations. The original
module re-exports from here for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum

import pandas as pd


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
    symbol: str | None = None  # None = affects all symbols
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
        self._events: list[MarketEvent] = []
        self._load_recurring_events()

    def _load_recurring_events(self) -> None:
        """Load recurring events (open/close auctions)."""
        # Market open and close are handled separately via trading windows
        pass

    def add_event(
        self,
        event_type: EventType,
        timestamp: datetime,
        symbol: str | None = None,
        impact: str = "HIGH",
    ) -> None:
        """Add an event to the calendar."""
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

    def add_fomc_dates(self, dates: list[datetime]) -> None:
        """Add FOMC meeting dates."""
        for date in dates:
            timestamp = datetime.combine(date.date(), time(14, 0))
            self.add_event(EventType.FOMC, timestamp)

    def add_cpi_dates(self, dates: list[datetime]) -> None:
        """Add CPI release dates."""
        for date in dates:
            timestamp = datetime.combine(date.date(), time(8, 30))
            self.add_event(EventType.CPI, timestamp)

    def add_nfp_dates(self, dates: list[datetime]) -> None:
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
            first_day = datetime(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)

            timestamp = datetime.combine(third_friday.date(), time(16, 0))

            if month in [3, 6, 9, 12]:
                self.add_event(EventType.QUAD_WITCH, timestamp)
            else:
                self.add_event(EventType.OPEX, timestamp)

    def is_blackout(
        self,
        check_time: datetime,
        symbol: str | None = None,
    ) -> tuple[bool, MarketEvent | None]:
        """Check if a time is in a blackout window."""
        for event in self._events:
            if event.symbol is not None and event.symbol != symbol:
                continue

            if event.is_in_blackout(check_time):
                return True, event

        return False, None

    def get_events_for_date(self, date: datetime) -> list[MarketEvent]:
        """Get all events for a specific date."""
        target_date = date.date()
        return [e for e in self._events if e.timestamp.date() == target_date]

    def get_upcoming_events(
        self,
        from_time: datetime,
        hours_ahead: int = 24,
    ) -> list[MarketEvent]:
        """Get events within the next N hours."""
        end_time = from_time + timedelta(hours=hours_ahead)
        return [e for e in self._events if from_time <= e.timestamp <= end_time]

    def get_blackout_series(
        self,
        index: pd.DatetimeIndex,
        symbol: str | None = None,
    ) -> pd.Series:
        """Get blackout status for a datetime index."""
        blackouts = []

        for ts in index:
            is_bo, _ = self.is_blackout(ts.to_pydatetime(), symbol)
            blackouts.append(is_bo)

        return pd.Series(blackouts, index=index, name="is_blackout")
