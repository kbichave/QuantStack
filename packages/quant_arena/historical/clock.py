# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Historical clock for simulation date management.

Provides a clock abstraction that:
- Steps through trading days chronologically
- Tracks simulation progress
- Provides lookback window helpers
- Determines policy update timing
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator, List, Optional, Tuple

from loguru import logger


@dataclass
class DateRange:
    """A range of dates."""

    start: date
    end: date

    @property
    def days(self) -> int:
        """Number of calendar days in range."""
        return (self.end - self.start).days + 1

    def __contains__(self, d: date) -> bool:
        """Check if date is in range."""
        return self.start <= d <= self.end

    def __repr__(self) -> str:
        return f"DateRange({self.start} to {self.end})"


class HistoricalClock:
    """
    Manages simulation time for historical backtesting.

    The clock steps through a list of trading days and provides
    utilities for:
    - Date iteration
    - Progress tracking
    - Lookback windows
    - Policy update timing

    Usage:
        clock = HistoricalClock(trading_days)

        for current_date in clock.iterate():
            # Process day
            print(f"Day {clock.day_index}: {current_date}")

            # Check for monthly boundary
            if clock.is_month_end():
                # Update policy
                pass
    """

    def __init__(
        self,
        trading_days: List[date],
        warmup_days: int = 20,
    ):
        """
        Initialize historical clock.

        Args:
            trading_days: List of trading days in chronological order
            warmup_days: Number of days to skip for indicator warmup
        """
        if not trading_days:
            raise ValueError("trading_days cannot be empty")

        self._trading_days = sorted(trading_days)
        self._warmup_days = warmup_days

        # Current position
        self._current_index = warmup_days  # Start after warmup
        self._current_date: Optional[date] = None

        # Cache for efficiency
        self._date_to_index = {d: i for i, d in enumerate(self._trading_days)}

        logger.info(
            f"HistoricalClock initialized: {len(self._trading_days)} days, "
            f"warmup={warmup_days}"
        )

    @property
    def current_date(self) -> Optional[date]:
        """Get current simulation date."""
        return self._current_date

    @property
    def day_index(self) -> int:
        """Get current day index (0-based)."""
        return self._current_index

    @property
    def total_days(self) -> int:
        """Total number of trading days."""
        return len(self._trading_days)

    @property
    def tradable_days(self) -> int:
        """Number of days available for trading (after warmup)."""
        return max(0, len(self._trading_days) - self._warmup_days)

    @property
    def progress(self) -> float:
        """Simulation progress as fraction (0.0 to 1.0)."""
        if self.tradable_days == 0:
            return 0.0
        completed = self._current_index - self._warmup_days
        return min(1.0, max(0.0, completed / self.tradable_days))

    @property
    def start_date(self) -> date:
        """First trading day."""
        return self._trading_days[0]

    @property
    def end_date(self) -> date:
        """Last trading day."""
        return self._trading_days[-1]

    @property
    def effective_start_date(self) -> date:
        """First tradable day (after warmup)."""
        return self._trading_days[min(self._warmup_days, len(self._trading_days) - 1)]

    def iterate(self) -> Iterator[date]:
        """
        Iterate through trading days.

        Yields:
            Each trading day starting after warmup period
        """
        for i in range(self._warmup_days, len(self._trading_days)):
            self._current_index = i
            self._current_date = self._trading_days[i]
            yield self._current_date

    def advance(self) -> bool:
        """
        Advance to next trading day.

        Returns:
            True if successfully advanced, False if at end
        """
        if self._current_index >= len(self._trading_days) - 1:
            return False

        self._current_index += 1
        self._current_date = self._trading_days[self._current_index]
        return True

    def reset(self) -> None:
        """Reset clock to beginning (after warmup)."""
        self._current_index = self._warmup_days
        self._current_date = None

    def get_lookback(self, days: int) -> DateRange:
        """
        Get lookback date range from current date.

        Args:
            days: Number of trading days to look back

        Returns:
            DateRange covering the lookback period
        """
        if self._current_date is None:
            raise RuntimeError("Clock not started. Call iterate() or advance() first.")

        start_idx = max(0, self._current_index - days)
        return DateRange(
            start=self._trading_days[start_idx],
            end=self._current_date,
        )

    def get_lookback_dates(self, days: int) -> List[date]:
        """
        Get list of lookback trading dates.

        Args:
            days: Number of trading days to look back

        Returns:
            List of dates in chronological order
        """
        start_idx = max(0, self._current_index - days)
        return self._trading_days[start_idx : self._current_index]

    def get_previous_date(self, n: int = 1) -> Optional[date]:
        """
        Get date n trading days ago.

        Args:
            n: Number of trading days back (default 1)

        Returns:
            The date or None if out of range
        """
        idx = self._current_index - n
        if idx < 0 or idx >= len(self._trading_days):
            return None
        return self._trading_days[idx]

    def is_trading_day(self, d: date) -> bool:
        """Check if a date is a trading day."""
        return d in self._date_to_index

    def is_month_end(self) -> bool:
        """
        Check if current date is the last trading day of the month.

        Returns:
            True if current date is month-end
        """
        if self._current_date is None:
            return False

        next_idx = self._current_index + 1
        if next_idx >= len(self._trading_days):
            return True  # End of data is also month end

        next_date = self._trading_days[next_idx]
        return next_date.month != self._current_date.month

    def is_quarter_end(self) -> bool:
        """
        Check if current date is the last trading day of the quarter.

        Returns:
            True if current date is quarter-end
        """
        if self._current_date is None:
            return False

        next_idx = self._current_index + 1
        if next_idx >= len(self._trading_days):
            return True

        next_date = self._trading_days[next_idx]
        current_quarter = (self._current_date.month - 1) // 3
        next_quarter = (next_date.month - 1) // 3
        return (
            next_quarter != current_quarter or next_date.year != self._current_date.year
        )

    def is_year_end(self) -> bool:
        """
        Check if current date is the last trading day of the year.

        Returns:
            True if current date is year-end
        """
        if self._current_date is None:
            return False

        next_idx = self._current_index + 1
        if next_idx >= len(self._trading_days):
            return True

        next_date = self._trading_days[next_idx]
        return next_date.year != self._current_date.year

    def should_update_policy(self, frequency: str) -> bool:
        """
        Check if policy should be updated based on frequency.

        Args:
            frequency: "monthly", "quarterly", or "never"

        Returns:
            True if policy should be updated
        """
        if frequency == "never":
            return False
        elif frequency == "monthly":
            return self.is_month_end()
        elif frequency == "quarterly":
            return self.is_quarter_end()
        else:
            logger.warning(f"Unknown policy frequency: {frequency}")
            return False

    def days_since(self, reference_date: date) -> int:
        """
        Get number of trading days since a reference date.

        Args:
            reference_date: Date to count from

        Returns:
            Number of trading days (negative if reference is in future)
        """
        if reference_date not in self._date_to_index:
            # Find closest date
            ref_idx = 0
            for i, d in enumerate(self._trading_days):
                if d >= reference_date:
                    ref_idx = i
                    break
        else:
            ref_idx = self._date_to_index[reference_date]

        return self._current_index - ref_idx

    def get_dates_between(self, start: date, end: date) -> List[date]:
        """Get all trading days between two dates (inclusive)."""
        return [d for d in self._trading_days if start <= d <= end]

    def __len__(self) -> int:
        """Number of trading days."""
        return len(self._trading_days)

    def __repr__(self) -> str:
        current = (
            self._current_date.isoformat() if self._current_date else "not started"
        )
        return (
            f"HistoricalClock(current={current}, "
            f"progress={self.progress:.1%}, days={len(self._trading_days)})"
        )
