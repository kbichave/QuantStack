"""
Microstructure-aware trading filters.

Prevents trading during unfavorable microstructure conditions.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.microstructure.liquidity import LiquidityAnalyzer, LiquidityFeatures
from quantcore.microstructure.events import EventCalendar, TradingCalendar


@dataclass
class FilterDecision:
    """Result of a filter check."""

    passed: bool
    reason: str
    severity: str = "INFO"  # INFO, WARNING, BLOCK


class TradingWindowFilter:
    """
    Filter trades based on time of day.

    Avoids:
    - First hour (opening auction noise)
    - Last hour (closing auction + positioning)
    - Pre/post market
    """

    def __init__(
        self,
        skip_first_minutes: int = 60,
        skip_last_minutes: int = 60,
        only_regular_hours: bool = True,
    ):
        """
        Initialize trading window filter.

        Args:
            skip_first_minutes: Minutes after open to skip
            skip_last_minutes: Minutes before close to skip
            only_regular_hours: Only allow regular hours trading
        """
        self.skip_first_minutes = skip_first_minutes
        self.skip_last_minutes = skip_last_minutes
        self.only_regular_hours = only_regular_hours

        self.calendar = TradingCalendar()

    def check(self, timestamp: datetime) -> FilterDecision:
        """
        Check if timestamp is in valid trading window.

        Args:
            timestamp: Time to check

        Returns:
            FilterDecision
        """
        # Check if trading day
        if not self.calendar.is_trading_day(timestamp):
            return FilterDecision(
                passed=False,
                reason="Not a trading day",
                severity="BLOCK",
            )

        t = timestamp.time()

        # Regular hours check
        if self.only_regular_hours:
            if not self.calendar.is_market_hours(timestamp):
                return FilterDecision(
                    passed=False,
                    reason="Outside regular hours",
                    severity="BLOCK",
                )

        # First hour
        open_threshold = time(
            self.calendar.MARKET_OPEN.hour,
            self.calendar.MARKET_OPEN.minute + self.skip_first_minutes,
        )
        if t < open_threshold:
            return FilterDecision(
                passed=False,
                reason=f"Within first {self.skip_first_minutes} minutes",
                severity="WARNING",
            )

        # Last hour
        close_time = self.calendar.MARKET_CLOSE
        close_threshold_minutes = (
            close_time.hour * 60 + close_time.minute - self.skip_last_minutes
        )
        close_threshold = time(
            close_threshold_minutes // 60, close_threshold_minutes % 60
        )

        if t >= close_threshold:
            return FilterDecision(
                passed=False,
                reason=f"Within last {self.skip_last_minutes} minutes",
                severity="WARNING",
            )

        return FilterDecision(
            passed=True,
            reason="Valid trading window",
        )

    def get_valid_windows_series(
        self,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Get valid trading window flags for an index.

        Args:
            index: DatetimeIndex

        Returns:
            Boolean Series
        """
        valid = []
        for ts in index:
            decision = self.check(ts.to_pydatetime())
            valid.append(decision.passed)

        return pd.Series(valid, index=index, name="valid_window")


class LiquidityFilter:
    """
    Filter trades based on liquidity conditions.

    Blocks trades when:
    - Volume too low
    - Spread too wide
    - Vol-of-vol spiking
    """

    def __init__(
        self,
        min_liquidity_score: float = 0.4,
        min_volume_ratio: float = 0.3,
        max_spread_bps: float = 15.0,
        max_vol_of_vol_zscore: float = 2.5,
    ):
        """
        Initialize liquidity filter.

        Args:
            min_liquidity_score: Minimum composite score
            min_volume_ratio: Minimum volume vs average
            max_spread_bps: Maximum spread in bps
            max_vol_of_vol_zscore: Max vol-of-vol z-score
        """
        self.min_liquidity_score = min_liquidity_score
        self.min_volume_ratio = min_volume_ratio
        self.max_spread_bps = max_spread_bps
        self.max_vol_of_vol_zscore = max_vol_of_vol_zscore

    def check(self, liquidity: LiquidityFeatures) -> FilterDecision:
        """
        Check if liquidity conditions are acceptable.

        Args:
            liquidity: LiquidityFeatures for the bar

        Returns:
            FilterDecision
        """
        reasons = []

        # Check liquidity score
        if liquidity.liquidity_score < self.min_liquidity_score:
            reasons.append(f"Low liquidity score: {liquidity.liquidity_score:.2f}")

        # Check volume
        if liquidity.volume_vs_avg < self.min_volume_ratio:
            reasons.append(f"Low volume: {liquidity.volume_vs_avg:.2f}x avg")

        # Check spread
        if liquidity.estimated_spread_bps > self.max_spread_bps:
            reasons.append(f"Wide spread: {liquidity.estimated_spread_bps:.1f} bps")

        # Check vol-of-vol
        if abs(liquidity.volatility_of_volatility) > self.max_vol_of_vol_zscore:
            reasons.append(
                f"Vol burst: {liquidity.volatility_of_volatility:.2f} z-score"
            )

        if reasons:
            return FilterDecision(
                passed=False,
                reason="; ".join(reasons),
                severity="WARNING" if len(reasons) == 1 else "BLOCK",
            )

        return FilterDecision(
            passed=True,
            reason="Liquidity OK",
        )


class EventFilter:
    """
    Filter trades based on economic/market events.

    Blocks trading around high-impact events.
    """

    def __init__(self, event_calendar: Optional[EventCalendar] = None):
        """
        Initialize event filter.

        Args:
            event_calendar: Event calendar instance
        """
        self.calendar = event_calendar or EventCalendar()

    def check(
        self,
        timestamp: datetime,
        symbol: Optional[str] = None,
    ) -> FilterDecision:
        """
        Check if timestamp is in an event blackout.

        Args:
            timestamp: Time to check
            symbol: Symbol to check

        Returns:
            FilterDecision
        """
        is_blackout, event = self.calendar.is_blackout(timestamp, symbol)

        if is_blackout:
            return FilterDecision(
                passed=False,
                reason=f"Event blackout: {event.event_type.value}",
                severity="BLOCK",
            )

        # Check for upcoming events
        upcoming = self.calendar.get_upcoming_events(timestamp, hours_ahead=2)
        if upcoming:
            return FilterDecision(
                passed=True,
                reason=f"Event in {len(upcoming)} hours: {upcoming[0].event_type.value}",
                severity="INFO",
            )

        return FilterDecision(
            passed=True,
            reason="No event restrictions",
        )


class MicrostructureFilter:
    """
    Combined microstructure filter.

    Integrates all microstructure checks:
    - Trading window
    - Liquidity
    - Events
    """

    def __init__(
        self,
        trading_window_filter: Optional[TradingWindowFilter] = None,
        liquidity_filter: Optional[LiquidityFilter] = None,
        event_filter: Optional[EventFilter] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize combined filter.

        Args:
            trading_window_filter: Trading window filter
            liquidity_filter: Liquidity filter
            event_filter: Event filter
            strict_mode: If True, any warning blocks trade
        """
        self.window_filter = trading_window_filter or TradingWindowFilter()
        self.liquidity_filter = liquidity_filter or LiquidityFilter()
        self.event_filter = event_filter or EventFilter()
        self.strict_mode = strict_mode

    def check(
        self,
        timestamp: datetime,
        liquidity: LiquidityFeatures,
        symbol: Optional[str] = None,
    ) -> Tuple[bool, List[FilterDecision]]:
        """
        Run all microstructure checks.

        Args:
            timestamp: Time to check
            liquidity: Liquidity features
            symbol: Symbol to check

        Returns:
            Tuple of (passed, list of decisions)
        """
        decisions = []

        # Trading window
        window_decision = self.window_filter.check(timestamp)
        decisions.append(window_decision)

        # Liquidity
        liquidity_decision = self.liquidity_filter.check(liquidity)
        decisions.append(liquidity_decision)

        # Events
        event_decision = self.event_filter.check(timestamp, symbol)
        decisions.append(event_decision)

        # Determine overall pass/fail
        has_block = any(d.severity == "BLOCK" for d in decisions if not d.passed)
        has_warning = any(d.severity == "WARNING" for d in decisions if not d.passed)

        if has_block:
            passed = False
        elif has_warning and self.strict_mode:
            passed = False
        else:
            passed = all(d.passed for d in decisions if d.severity != "INFO")

        return passed, decisions

    def get_filter_series(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get filter results for entire DataFrame.

        Args:
            df: DataFrame with liquidity features
            symbol: Symbol

        Returns:
            DataFrame with filter columns
        """
        result = df.copy()

        # Get trading window
        result["filter_window"] = self.window_filter.get_valid_windows_series(df.index)

        # Get event blackouts
        result["filter_event"] = ~self.event_filter.calendar.get_blackout_series(
            df.index, symbol
        )

        # Liquidity filter (requires computed features)
        if "liquidity_score" in df.columns:
            result["filter_liquidity"] = (
                df["liquidity_score"] >= self.liquidity_filter.min_liquidity_score
            ) & (df["volume_vs_avg"] >= self.liquidity_filter.min_volume_ratio)
        else:
            result["filter_liquidity"] = True

        # Combined filter
        result["filter_microstructure"] = (
            result["filter_window"]
            & result["filter_event"]
            & result["filter_liquidity"]
        ).astype(int)

        return result
