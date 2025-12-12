"""
Mean-reversion entry and exit rules.

Implements the core MR logic: Z-score stretch + reversion confirmation.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.config.timeframes import Timeframe, TIMEFRAME_PARAMS


@dataclass
class EntrySignal:
    """Entry signal from MR rules."""

    triggered: bool
    direction: Literal["LONG", "SHORT", "NONE"]
    zscore: float
    reversion_confirmed: bool
    stretch_magnitude: float  # How far from mean
    price: float
    atr: float

    @property
    def strength(self) -> float:
        """Signal strength based on stretch magnitude."""
        return min(abs(self.stretch_magnitude) / 3.0, 1.0)


class MeanReversionRules:
    """
    Core mean-reversion entry rules.

    Entry conditions for LONG:
    1. Z-score stretch: z_{t-1} < -threshold (oversold)
    2. Reversion confirmation: z_t > z_{t-1} + delta (turning up)
    3. Price confirmation: close > close_{t-1} (positive bar)

    Entry conditions for SHORT:
    1. Z-score stretch: z_{t-1} > +threshold (overbought)
    2. Reversion confirmation: z_t < z_{t-1} - delta (turning down)
    3. Price confirmation: close < close_{t-1} (negative bar)
    """

    def __init__(
        self,
        timeframe: Timeframe,
        zscore_threshold: Optional[float] = None,
        reversion_delta: float = 0.2,
        require_price_confirmation: bool = True,
    ):
        """
        Initialize MR rules.

        Args:
            timeframe: Trading timeframe
            zscore_threshold: Override z-score threshold
            reversion_delta: Minimum z-score change for confirmation
            require_price_confirmation: Require price bar confirmation
        """
        self.timeframe = timeframe
        self.params = TIMEFRAME_PARAMS[timeframe]

        self.zscore_threshold = zscore_threshold or self.params.zscore_entry_threshold
        self.reversion_delta = reversion_delta
        self.require_price_confirmation = require_price_confirmation

    def check_entry(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> EntrySignal:
        """
        Check if entry conditions are met at given index.

        Args:
            df: DataFrame with features (must have zscore_price, close, atr)
            idx: Index to check (default: last bar)

        Returns:
            EntrySignal with entry details
        """
        if len(df) < 2:
            return self._no_signal(df)

        # Get current and previous bar
        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        # Required features
        if "zscore_price" not in df.columns:
            logger.warning("zscore_price not in DataFrame")
            return self._no_signal(df)

        z_current = current.get("zscore_price", 0)
        z_previous = previous.get("zscore_price", 0)

        if pd.isna(z_current) or pd.isna(z_previous):
            return self._no_signal(df)

        close_current = current["close"]
        close_previous = previous["close"]
        atr = current.get("atr", close_current * 0.01)

        # Check LONG conditions
        long_stretch = z_previous < -self.zscore_threshold
        long_reversion = z_current > z_previous + self.reversion_delta
        long_price_confirm = (
            close_current > close_previous if self.require_price_confirmation else True
        )

        long_triggered = long_stretch and long_reversion and long_price_confirm

        # Check SHORT conditions
        short_stretch = z_previous > self.zscore_threshold
        short_reversion = z_current < z_previous - self.reversion_delta
        short_price_confirm = (
            close_current < close_previous if self.require_price_confirmation else True
        )

        short_triggered = short_stretch and short_reversion and short_price_confirm

        # Determine direction
        if long_triggered and not short_triggered:
            direction = "LONG"
            triggered = True
            stretch_magnitude = z_previous
            reversion_confirmed = long_reversion
        elif short_triggered and not long_triggered:
            direction = "SHORT"
            triggered = True
            stretch_magnitude = z_previous
            reversion_confirmed = short_reversion
        else:
            direction = "NONE"
            triggered = False
            stretch_magnitude = z_previous
            reversion_confirmed = False

        return EntrySignal(
            triggered=triggered,
            direction=direction,
            zscore=z_current,
            reversion_confirmed=reversion_confirmed,
            stretch_magnitude=stretch_magnitude,
            price=close_current,
            atr=atr,
        )

    def check_exit(
        self,
        df: pd.DataFrame,
        entry_direction: Literal["LONG", "SHORT"],
        entry_price: float,
        tp_price: float,
        sl_price: float,
        idx: int = -1,
    ) -> tuple[bool, str, float]:
        """
        Check exit conditions.

        Args:
            df: DataFrame with OHLCV
            entry_direction: Original trade direction
            entry_price: Entry price
            tp_price: Take profit price
            sl_price: Stop loss price
            idx: Bar index to check

        Returns:
            Tuple of (should_exit, exit_reason, exit_price)
        """
        current = df.iloc[idx]

        high = current["high"]
        low = current["low"]
        close = current["close"]

        if entry_direction == "LONG":
            # Check SL (hit if low <= sl_price)
            if low <= sl_price:
                return True, "SL", sl_price
            # Check TP (hit if high >= tp_price)
            if high >= tp_price:
                return True, "TP", tp_price
            # Check z-score exit (optional - exit when back to mean)
            z_current = current.get("zscore_price", 0)
            if z_current >= -self.params.zscore_exit_threshold:
                return True, "MEAN_REVERT", close
        else:  # SHORT
            # Check SL
            if high >= sl_price:
                return True, "SL", sl_price
            # Check TP
            if low <= tp_price:
                return True, "TP", tp_price
            # Check z-score exit
            z_current = current.get("zscore_price", 0)
            if z_current <= self.params.zscore_exit_threshold:
                return True, "MEAN_REVERT", close

        return False, "", 0.0

    def _no_signal(self, df: pd.DataFrame) -> EntrySignal:
        """Return empty signal."""
        if len(df) > 0:
            current = df.iloc[-1]
            price = current["close"]
            atr = current.get("atr", price * 0.01)
        else:
            price = 0.0
            atr = 0.0

        return EntrySignal(
            triggered=False,
            direction="NONE",
            zscore=0.0,
            reversion_confirmed=False,
            stretch_magnitude=0.0,
            price=price,
            atr=atr,
        )

    def scan_for_signals(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scan entire DataFrame for entry signals.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with signal columns added
        """
        result = df.copy()

        # Initialize signal columns
        result["mr_signal"] = 0
        result["mr_direction"] = "NONE"
        result["mr_strength"] = 0.0

        for i in range(1, len(result)):
            signal = self.check_entry(result, idx=i)

            if signal.triggered:
                result.iloc[i, result.columns.get_loc("mr_signal")] = 1
                result.iloc[i, result.columns.get_loc("mr_direction")] = (
                    signal.direction
                )
                result.iloc[i, result.columns.get_loc("mr_strength")] = signal.strength

        return result
