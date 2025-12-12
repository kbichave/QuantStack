# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Adaptive Holding Period Management.

Determines optimal holding periods based on:
1. Signal source timeframe (Weekly, Daily, 4H, 1H)
2. MTF alignment score
3. Current market conditions
4. Trade performance

The hierarchy follows:
- Weekly signals → Position trades (weeks)
- Daily signals → Swing trades (3-10 days)
- 4H signals → Short swings (1-5 days)
- 1H signals → Intraday (hours)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from loguru import logger


class HoldingType(Enum):
    """Types of holding periods."""

    INTRADAY = "intraday"  # Hours, close same day
    SHORT_SWING = "short_swing"  # 1-5 days
    SWING = "swing"  # 3-10 days
    POSITION = "position"  # Weeks to months


@dataclass
class HoldingPeriodConfig:
    """Configuration for a holding period type."""

    type: HoldingType
    min_bars: int  # Minimum bars to hold
    max_bars: int  # Maximum bars before forced exit
    target_atr_multiple: float  # Take profit in ATR multiples
    stop_atr_multiple: float  # Stop loss in ATR multiples
    trailing_stop: bool  # Use trailing stop
    scale_out: bool  # Scale out of position


# Default configurations for each holding type
HOLDING_CONFIGS: Dict[HoldingType, HoldingPeriodConfig] = {
    HoldingType.INTRADAY: HoldingPeriodConfig(
        type=HoldingType.INTRADAY,
        min_bars=1,
        max_bars=6,  # 6 hourly bars = one trading day
        target_atr_multiple=1.5,
        stop_atr_multiple=1.0,
        trailing_stop=False,
        scale_out=False,
    ),
    HoldingType.SHORT_SWING: HoldingPeriodConfig(
        type=HoldingType.SHORT_SWING,
        min_bars=4,  # 4 4H bars = 1 day
        max_bars=30,  # ~5 days in 4H bars
        target_atr_multiple=2.0,
        stop_atr_multiple=1.0,
        trailing_stop=True,
        scale_out=False,
    ),
    HoldingType.SWING: HoldingPeriodConfig(
        type=HoldingType.SWING,
        min_bars=3,  # 3 days minimum
        max_bars=10,  # 10 days maximum
        target_atr_multiple=2.5,
        stop_atr_multiple=1.5,
        trailing_stop=True,
        scale_out=True,
    ),
    HoldingType.POSITION: HoldingPeriodConfig(
        type=HoldingType.POSITION,
        min_bars=5,  # 1 week minimum
        max_bars=40,  # ~2 months
        target_atr_multiple=3.0,
        stop_atr_multiple=2.0,
        trailing_stop=True,
        scale_out=True,
    ),
}


@dataclass
class HoldingDecision:
    """Decision about holding period for a trade."""

    holding_type: HoldingType
    config: HoldingPeriodConfig

    # Entry details
    entry_date: date
    entry_price: float

    # Target and stop
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None

    # Expected exit
    expected_exit_date: Optional[date] = None
    max_exit_date: Optional[date] = None

    # Current state
    bars_held: int = 0
    highest_price: float = 0.0
    lowest_price: float = float("inf")

    # Reasoning
    reasoning: str = ""

    def should_exit(
        self,
        current_price: float,
        current_date: date,
        current_bar: int,
    ) -> tuple[bool, str]:
        """
        Check if position should be exited.

        Returns:
            Tuple of (should_exit, reason)
        """
        # Update tracking
        self.bars_held = current_bar
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)

        # Check stop loss
        if self.stop_price and current_price <= self.stop_price:
            return True, f"Stop loss hit at ${current_price:.2f}"

        # Check trailing stop
        if self.config.trailing_stop and self.trailing_stop_price:
            if current_price <= self.trailing_stop_price:
                return True, f"Trailing stop hit at ${current_price:.2f}"

        # Check target
        if self.target_price and current_price >= self.target_price:
            return True, f"Target reached at ${current_price:.2f}"

        # Check max bars
        if self.bars_held >= self.config.max_bars:
            return True, f"Max holding period ({self.config.max_bars} bars) reached"

        # Check max exit date
        if self.max_exit_date and current_date >= self.max_exit_date:
            return True, f"Max exit date {self.max_exit_date} reached"

        return False, ""

    def update_trailing_stop(self, current_price: float, atr: float) -> None:
        """Update trailing stop based on current price."""
        if not self.config.trailing_stop:
            return

        # Only trail if profitable
        if current_price > self.entry_price:
            new_stop = current_price - (atr * self.config.stop_atr_multiple)

            if self.trailing_stop_price is None:
                self.trailing_stop_price = new_stop
            else:
                # Ratchet up only
                self.trailing_stop_price = max(self.trailing_stop_price, new_stop)

    def get_scale_out_levels(self) -> List[tuple[float, float]]:
        """
        Get scale-out levels if enabled.

        Returns:
            List of (price, fraction_to_sell) tuples
        """
        if not self.config.scale_out or not self.target_price:
            return []

        # Scale out at 50% and 75% of target move
        move = self.target_price - self.entry_price

        return [
            (self.entry_price + move * 0.5, 0.33),  # Sell 33% at halfway
            (self.entry_price + move * 0.75, 0.33),  # Sell 33% at 75%
            # Final 34% at full target or trail
        ]


class HoldingPeriodManager:
    """
    Manager for adaptive holding periods.

    Determines holding period based on signal source and MTF alignment,
    then tracks positions for exit decisions.
    """

    def __init__(self):
        """Initialize manager."""
        self._active_holdings: Dict[str, HoldingDecision] = {}  # symbol -> decision

    def determine_holding_period(
        self,
        symbol: str,
        entry_price: float,
        entry_date: date,
        signal_source: str,
        mtf_alignment: float,
        atr: float,
        mtf_context: Optional[Any] = None,
    ) -> HoldingDecision:
        """
        Determine appropriate holding period for a trade.

        Args:
            symbol: Symbol being traded
            entry_price: Entry price
            entry_date: Entry date
            signal_source: Source timeframe of signal ("weekly", "daily", "4h", "1h")
            mtf_alignment: MTF alignment score (0-1)
            atr: Current ATR for stop/target calculation
            mtf_context: Optional MTFContext for additional context

        Returns:
            HoldingDecision with all details
        """
        # Map signal source to holding type
        holding_type = self._map_signal_to_holding(signal_source, mtf_alignment)
        config = HOLDING_CONFIGS[holding_type]

        # Calculate target and stop
        is_long = True  # Assume long, would need trade direction in production

        target_price = entry_price + (atr * config.target_atr_multiple)
        stop_price = entry_price - (atr * config.stop_atr_multiple)

        # Calculate expected exit dates
        if holding_type == HoldingType.INTRADAY:
            expected_exit = entry_date  # Same day
            max_exit = entry_date + timedelta(days=1)
        elif holding_type == HoldingType.SHORT_SWING:
            expected_exit = entry_date + timedelta(days=3)
            max_exit = entry_date + timedelta(days=5)
        elif holding_type == HoldingType.SWING:
            expected_exit = entry_date + timedelta(days=5)
            max_exit = entry_date + timedelta(days=10)
        else:  # POSITION
            expected_exit = entry_date + timedelta(days=15)
            max_exit = entry_date + timedelta(days=60)

        # Build reasoning
        reasoning = self._build_reasoning(
            signal_source, mtf_alignment, holding_type, config, mtf_context
        )

        decision = HoldingDecision(
            holding_type=holding_type,
            config=config,
            entry_date=entry_date,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            trailing_stop_price=stop_price if config.trailing_stop else None,
            expected_exit_date=expected_exit,
            max_exit_date=max_exit,
            highest_price=entry_price,
            lowest_price=entry_price,
            reasoning=reasoning,
        )

        # Track the holding
        self._active_holdings[symbol] = decision

        logger.info(
            f"Holding period set for {symbol}: {holding_type.value}, "
            f"target=${target_price:.2f}, stop=${stop_price:.2f}, "
            f"max_exit={max_exit}"
        )

        return decision

    def _map_signal_to_holding(
        self,
        signal_source: str,
        mtf_alignment: float,
    ) -> HoldingType:
        """Map signal source to holding type, adjusted by alignment."""
        source_lower = signal_source.lower()

        # Base mapping
        if "weekly" in source_lower or "position" in source_lower:
            base_type = HoldingType.POSITION
        elif "daily" in source_lower or "swing" in source_lower:
            base_type = HoldingType.SWING
        elif "4h" in source_lower or "short" in source_lower:
            base_type = HoldingType.SHORT_SWING
        else:  # 1h, intraday
            base_type = HoldingType.INTRADAY

        # Adjust based on alignment
        # Low alignment = shorter holding (more conservative)
        if mtf_alignment < 0.5:
            if base_type == HoldingType.POSITION:
                return HoldingType.SWING
            elif base_type == HoldingType.SWING:
                return HoldingType.SHORT_SWING
        # High alignment = can extend
        elif mtf_alignment > 0.8:
            if base_type == HoldingType.SWING:
                return HoldingType.POSITION
            elif base_type == HoldingType.SHORT_SWING:
                return HoldingType.SWING

        return base_type

    def _build_reasoning(
        self,
        signal_source: str,
        mtf_alignment: float,
        holding_type: HoldingType,
        config: HoldingPeriodConfig,
        mtf_context: Optional[Any],
    ) -> str:
        """Build reasoning string for holding decision."""
        parts = [
            f"Signal from {signal_source} timeframe",
            f"MTF alignment: {mtf_alignment:.0%}",
            f"Holding type: {holding_type.value}",
            f"Expected hold: {config.min_bars}-{config.max_bars} bars",
        ]

        if mtf_context:
            if hasattr(mtf_context, "weekly_regime"):
                parts.append(f"Weekly regime: {mtf_context.weekly_regime}")
            if hasattr(mtf_context, "suggested_holding_period"):
                parts.append(f"MTF suggested: {mtf_context.suggested_holding_period}")

        if config.trailing_stop:
            parts.append("Trailing stop enabled")
        if config.scale_out:
            parts.append("Scale-out enabled")

        return " | ".join(parts)

    def get_holding(self, symbol: str) -> Optional[HoldingDecision]:
        """Get active holding decision for a symbol."""
        return self._active_holdings.get(symbol)

    def check_exit(
        self,
        symbol: str,
        current_price: float,
        current_date: date,
        atr: float,
    ) -> tuple[bool, str]:
        """
        Check if a position should be exited.

        Args:
            symbol: Symbol to check
            current_price: Current price
            current_date: Current date
            atr: Current ATR for trailing stop update

        Returns:
            Tuple of (should_exit, reason)
        """
        decision = self._active_holdings.get(symbol)
        if not decision:
            return False, "No active holding"

        # Update trailing stop
        decision.update_trailing_stop(current_price, atr)

        # Check exit conditions
        decision.bars_held += 1
        return decision.should_exit(current_price, current_date, decision.bars_held)

    def close_holding(self, symbol: str) -> None:
        """Remove a symbol from active holdings."""
        if symbol in self._active_holdings:
            del self._active_holdings[symbol]
            logger.info(f"Closed holding tracking for {symbol}")

    def get_all_holdings(self) -> Dict[str, HoldingDecision]:
        """Get all active holdings."""
        return self._active_holdings.copy()


# =============================================================================
# SINGLETON
# =============================================================================

_manager_instance: Optional[HoldingPeriodManager] = None


def get_holding_manager() -> HoldingPeriodManager:
    """Get singleton HoldingPeriodManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = HoldingPeriodManager()
    return _manager_instance
