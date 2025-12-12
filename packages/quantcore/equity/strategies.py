"""
Equity trading strategies.

Provides signal generation strategies for equity trading:
- MeanReversionStrategy: Z-score stretch + reversion confirmation
- MomentumStrategy: RSI + MACD based signals
- TrendFollowingStrategy: MA crossover signals
- RRGStrategy: Relative Rotation Graph quadrant signals
- CompositeStrategy: Voting ensemble of multiple strategies
"""

from typing import List
from enum import Enum

import numpy as np
import pandas as pd


class Signal(Enum):
    """Trading signal."""

    LONG = 1
    SHORT = -1
    FLAT = 0


class EquityStrategy:
    """Base class for equity strategies."""

    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate signals for all bars.

        Args:
            features: DataFrame with feature columns

        Returns:
            Series of signals (1=LONG, -1=SHORT, 0=FLAT)
        """
        raise NotImplementedError


class MeanReversionStrategy(EquityStrategy):
    """
    Mean reversion strategy using z-score stretch + reversion confirmation.

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
        zscore_threshold: float = 2.0,
        reversion_delta: float = 0.2,
    ):
        super().__init__("MeanReversion")
        self.zscore_threshold = zscore_threshold
        self.reversion_delta = reversion_delta

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate signals using MeanReversionRules logic."""
        signals = pd.Series(0, index=features.index)

        if "zscore_price" not in features.columns:
            return signals

        zscore = features["zscore_price"]
        zscore_prev = zscore.shift(1)
        close = features["close"]
        close_prev = close.shift(1)

        # Long: oversold + reverting up
        long_stretch = zscore_prev < -self.zscore_threshold
        long_reversion = zscore > zscore_prev + self.reversion_delta
        long_price = close > close_prev
        long_signal = long_stretch & long_reversion & long_price

        # Short: overbought + reverting down
        short_stretch = zscore_prev > self.zscore_threshold
        short_reversion = zscore < zscore_prev - self.reversion_delta
        short_price = close < close_prev
        short_signal = short_stretch & short_reversion & short_price

        signals[long_signal] = 1
        signals[short_signal] = -1

        return signals


class MomentumStrategy(EquityStrategy):
    """
    Momentum strategy using RSI + MACD.

    LONG: RSI oversold + MACD bullish crossover
    SHORT: RSI overbought + MACD bearish crossover
    """

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        super().__init__("Momentum")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI and MACD."""
        signals = pd.Series(0, index=features.index)

        # Find RSI column
        rsi_col = None
        for col in ["1H_rsi_14", "rsi_14", "rsi"]:
            if col in features.columns:
                rsi_col = col
                break

        # Find MACD columns
        macd_col = None
        macd_signal_col = None
        for col in ["1H_macd", "macd"]:
            if col in features.columns:
                macd_col = col
                break
        for col in ["1H_macd_signal", "macd_signal"]:
            if col in features.columns:
                macd_signal_col = col
                break

        if rsi_col is None:
            return signals

        rsi = features[rsi_col]

        if macd_col and macd_signal_col:
            macd_diff = features[macd_col] - features[macd_signal_col]
            long_signal = (rsi < self.rsi_oversold) & (macd_diff > 0)
            short_signal = (rsi > self.rsi_overbought) & (macd_diff < 0)
        else:
            long_signal = rsi < self.rsi_oversold
            short_signal = rsi > self.rsi_overbought

        signals[long_signal] = 1
        signals[short_signal] = -1

        return signals


class TrendFollowingStrategy(EquityStrategy):
    """
    Trend following strategy using MA crossovers.

    LONG: price > SMA20 > SMA50 (uptrend)
    SHORT: price < SMA20 < SMA50 (downtrend)
    """

    def __init__(self):
        super().__init__("TrendFollowing")

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate signals based on MA crossovers."""
        signals = pd.Series(0, index=features.index)

        # Find SMA columns
        sma20_col = None
        sma50_col = None
        for col in ["1H_sma_20", "sma_20"]:
            if col in features.columns:
                sma20_col = col
                break
        for col in ["1H_sma_50", "sma_50"]:
            if col in features.columns:
                sma50_col = col
                break

        if sma20_col is None or sma50_col is None:
            return signals

        close = features["close"]
        sma20 = features[sma20_col]
        sma50 = features[sma50_col]

        # Long: price > SMA20 > SMA50
        long_signal = (close > sma20) & (sma20 > sma50)
        # Short: price < SMA20 < SMA50
        short_signal = (close < sma20) & (sma20 < sma50)

        signals[long_signal] = 1
        signals[short_signal] = -1

        return signals


class RRGStrategy(EquityStrategy):
    """
    RRG-based rotation strategy.

    LONG: In Leading or Improving quadrant
    SHORT: In Weakening or Lagging quadrant
    """

    def __init__(self):
        super().__init__("RRG")

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate signals based on RRG quadrant."""
        signals = pd.Series(0, index=features.index)

        # Find RRG columns
        leading_col = None
        improving_col = None
        for col in ["1H_rrg_leading", "rrg_leading"]:
            if col in features.columns:
                leading_col = col
                break
        for col in ["1H_rrg_improving", "rrg_improving"]:
            if col in features.columns:
                improving_col = col
                break

        if leading_col is None and improving_col is None:
            return signals

        # Long in Leading or Improving quadrant
        if leading_col:
            signals[features[leading_col] == 1] = 1
        if improving_col:
            signals[features[improving_col] == 1] = 1

        # Short in Weakening or Lagging quadrant
        for col in [
            "1H_rrg_weakening",
            "rrg_weakening",
            "1H_rrg_lagging",
            "rrg_lagging",
        ]:
            if col in features.columns:
                signals[features[col] == 1] = -1

        return signals


class CompositeStrategy(EquityStrategy):
    """
    Composite strategy using voting across multiple strategies.

    Takes average of signals from all child strategies:
    - avg > 0.3 → LONG
    - avg < -0.3 → SHORT
    - otherwise → FLAT
    """

    def __init__(self, strategies: List[EquityStrategy]):
        super().__init__("Composite")
        self.strategies = strategies

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate signals by voting across strategies."""
        all_signals = pd.DataFrame(index=features.index)

        for strategy in self.strategies:
            all_signals[strategy.name] = strategy.generate_signals(features)

        avg_signal = all_signals.mean(axis=1)

        signals = pd.Series(0, index=features.index)
        signals[avg_signal > 0.3] = 1
        signals[avg_signal < -0.3] = -1

        return signals
