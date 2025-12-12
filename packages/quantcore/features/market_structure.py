"""
Market structure features.

Includes:
- Probable swing point detection (no lookahead)
- ZigZag-style swing detection (ATR-based)
- Trend exhaustion indicators
- Higher highs / lower lows patterns
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class MarketStructureFeatures(FeatureBase):
    """
    Market structure indicators.

    Features:
    - Probable swing lows/highs (live-usable, no lookahead)
    - Trend exhaustion counts
    - Higher highs / lower lows patterns
    - Support/resistance proximity
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market structure features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with market structure features added
        """
        result = df.copy()
        close = result["close"]
        high = result["high"]
        low = result["low"]

        # Probable swing points (no lookahead)
        result["probable_swing_low"] = self._detect_probable_swing_low(
            result, lookback=self.params.swing_lookback
        )
        result["probable_swing_high"] = self._detect_probable_swing_high(
            result, lookback=self.params.swing_lookback
        )

        # Bars since last swing
        result["bars_since_swing_low"] = self._bars_since_signal(
            result["probable_swing_low"]
        )
        result["bars_since_swing_high"] = self._bars_since_signal(
            result["probable_swing_high"]
        )

        # Trend exhaustion (consecutive up/down bars)
        result["consecutive_up_bars"] = self._count_consecutive(close.diff() > 0)
        result["consecutive_down_bars"] = self._count_consecutive(close.diff() < 0)

        # Exhaustion signals
        result["uptrend_exhaustion"] = (
            result["consecutive_up_bars"] >= self.params.trend_exhaustion_bars
        ).astype(int)
        result["downtrend_exhaustion"] = (
            result["consecutive_down_bars"] >= self.params.trend_exhaustion_bars
        ).astype(int)

        # Higher highs / lower lows patterns
        result["higher_high"] = (high > high.shift(1)).astype(int)
        result["lower_low"] = (low < low.shift(1)).astype(int)
        result["higher_low"] = (low > low.shift(1)).astype(int)
        result["lower_high"] = (high < high.shift(1)).astype(int)

        # Trend structure score
        result["trend_structure"] = self._compute_trend_structure(result)

        # Recent swing high/low values
        result["recent_swing_high"] = self._get_recent_swing_value(
            high, result["probable_swing_high"], lookback=20
        )
        result["recent_swing_low"] = self._get_recent_swing_value(
            low, result["probable_swing_low"], lookback=20
        )

        # Distance to recent swings (percentage)
        result["dist_to_swing_high"] = (
            (result["recent_swing_high"] - close) / close * 100
        )
        result["dist_to_swing_low"] = (close - result["recent_swing_low"]) / close * 100

        # Support/resistance proximity
        result["near_support"] = (result["dist_to_swing_low"] < 1).astype(int)
        result["near_resistance"] = (result["dist_to_swing_high"] < 1).astype(int)

        # Range position (where price is in recent range)
        recent_high = high.rolling(window=20).max()
        recent_low = low.rolling(window=20).min()
        range_size = recent_high - recent_low
        range_size_safe = range_size.replace(0, np.nan)
        result["range_position"] = (close - recent_low) / range_size_safe * 100

        # Breakout signals
        result["new_20_high"] = (high == recent_high).astype(int)
        result["new_20_low"] = (low == recent_low).astype(int)

        return result

    def _detect_probable_swing_low(
        self,
        df: pd.DataFrame,
        lookback: int,
    ) -> pd.Series:
        """
        Detect probable swing lows without lookahead.

        Conditions for probable swing low:
        1. Last k lows are rising (higher lows)
        2. RSI turning up (if available)
        3. Price closes above prior bar's low
        4. Recent volatility decreasing or stable
        """
        low = df["low"]
        close = df["close"]

        # Condition 1: Rising lows (higher lows)
        rising_lows = pd.Series(True, index=df.index)
        for i in range(1, lookback):
            rising_lows = rising_lows & (low.shift(i - 1) > low.shift(i))

        # Condition 2: Current close above prior bar's low
        close_above_prior_low = close > low.shift(1)

        # Condition 3: Not making new lows
        not_new_low = low > low.rolling(window=lookback * 2).min().shift(1)

        # Condition 4: RSI turning (if available)
        if "rsi" in df.columns:
            rsi_turning = (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"].shift(1) < 40)
        else:
            rsi_turning = pd.Series(True, index=df.index)

        # Combine conditions
        probable_swing = rising_lows & close_above_prior_low & not_new_low & rsi_turning

        return probable_swing.astype(int)

    def _detect_probable_swing_high(
        self,
        df: pd.DataFrame,
        lookback: int,
    ) -> pd.Series:
        """
        Detect probable swing highs without lookahead.

        Conditions for probable swing high:
        1. Last k highs are falling (lower highs)
        2. RSI turning down (if available)
        3. Price closes below prior bar's high
        4. Recent volatility decreasing or stable
        """
        high = df["high"]
        close = df["close"]

        # Condition 1: Falling highs (lower highs)
        falling_highs = pd.Series(True, index=df.index)
        for i in range(1, lookback):
            falling_highs = falling_highs & (high.shift(i - 1) < high.shift(i))

        # Condition 2: Current close below prior bar's high
        close_below_prior_high = close < high.shift(1)

        # Condition 3: Not making new highs
        not_new_high = high < high.rolling(window=lookback * 2).max().shift(1)

        # Condition 4: RSI turning (if available)
        if "rsi" in df.columns:
            rsi_turning = (df["rsi"] < df["rsi"].shift(1)) & (df["rsi"].shift(1) > 60)
        else:
            rsi_turning = pd.Series(True, index=df.index)

        # Combine conditions
        probable_swing = (
            falling_highs & close_below_prior_high & not_new_high & rsi_turning
        )

        return probable_swing.astype(int)

    def _bars_since_signal(self, signal: pd.Series) -> pd.Series:
        """Count bars since last signal."""
        # Create groups that increment on each signal
        groups = signal.cumsum()

        # Count within each group
        counts = signal.groupby(groups).cumcount()

        # Where no signal yet, set to large number
        counts = counts.where(groups > 0, np.nan)

        return counts

    def _count_consecutive(self, condition: pd.Series) -> pd.Series:
        """Count consecutive True values."""
        # Create groups that reset on False
        groups = (~condition).cumsum()

        # Count within each group
        return condition.groupby(groups).cumcount() + condition.astype(int)

    def _compute_trend_structure(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute trend structure score.

        Positive = bullish structure (higher highs, higher lows)
        Negative = bearish structure (lower highs, lower lows)
        """
        # Rolling sum of higher highs vs lower highs
        hh_count = df["higher_high"].rolling(window=10).sum()
        lh_count = df["lower_high"].rolling(window=10).sum()

        # Rolling sum of higher lows vs lower lows
        hl_count = df["higher_low"].rolling(window=10).sum()
        ll_count = df["lower_low"].rolling(window=10).sum()

        # Bullish: HH + HL, Bearish: LH + LL
        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count

        total = bullish_score + bearish_score
        total_safe = total.replace(0, np.nan)

        # Normalize to -1 to +1
        return (bullish_score - bearish_score) / total_safe

    def _get_recent_swing_value(
        self,
        price: pd.Series,
        swing_signal: pd.Series,
        lookback: int,
    ) -> pd.Series:
        """Get the price value at the most recent swing point."""
        # Mask non-swing prices
        swing_prices = price.where(swing_signal == 1, np.nan)

        # Forward fill to propagate swing values
        return swing_prices.fillna(method="ffill", limit=lookback)

    def detect_zigzag_swings(
        self,
        df: pd.DataFrame,
        atr_mult: float = 1.5,
        min_bars: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect ZigZag-style swing points using ATR threshold.

        This is a more robust swing detection than probable_swing_*
        and is used by the wave labeler for pattern detection.

        Args:
            df: DataFrame with OHLCV and ATR
            atr_mult: ATR multiple for reversal threshold
            min_bars: Minimum bars between swings

        Returns:
            Tuple of (swing_high_series, swing_low_series) with 1 at swing points
        """
        n = len(df)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Get ATR
        if "atr" in df.columns:
            atr = df["atr"].values
        else:
            atr = self._compute_atr_array(df)

        swing_high = np.zeros(n, dtype=int)
        swing_low = np.zeros(n, dtype=int)

        if n < 5:
            return pd.Series(swing_high, index=df.index), pd.Series(
                swing_low, index=df.index
            )

        # Initialize
        last_swing_idx = 0
        last_swing_price = closes[0]
        direction: Optional[str] = None

        for i in range(1, n):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            threshold = atr_mult * atr[i]

            if direction is None:
                move = closes[i] - last_swing_price
                if abs(move) >= threshold:
                    direction = "up" if move > 0 else "down"
                    last_swing_idx = 0
                    last_swing_price = closes[0]
                continue

            if direction == "up":
                if highs[i] >= last_swing_price:
                    last_swing_idx = i
                    last_swing_price = highs[i]
                else:
                    move_down = last_swing_price - lows[i]
                    if move_down >= threshold and (i - last_swing_idx) >= min_bars:
                        swing_high[last_swing_idx] = 1
                        direction = "down"
                        last_swing_idx = i
                        last_swing_price = lows[i]
            else:
                if lows[i] <= last_swing_price:
                    last_swing_idx = i
                    last_swing_price = lows[i]
                else:
                    move_up = highs[i] - last_swing_price
                    if move_up >= threshold and (i - last_swing_idx) >= min_bars:
                        swing_low[last_swing_idx] = 1
                        direction = "up"
                        last_swing_idx = i
                        last_swing_price = highs[i]

        return (
            pd.Series(swing_high, index=df.index, name="zigzag_swing_high"),
            pd.Series(swing_low, index=df.index, name="zigzag_swing_low"),
        )

    def _compute_atr_array(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Compute ATR as numpy array."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]

        for i in range(1, len(df)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.zeros(len(df))
        atr[:period] = np.nan
        if period <= len(df):
            atr[period - 1] = np.mean(tr[:period])
            multiplier = 2 / (period + 1)
            for i in range(period, len(df)):
                atr[i] = (tr[i] * multiplier) + (atr[i - 1] * (1 - multiplier))

        return atr

    def get_feature_names(self) -> List[str]:
        """Return list of market structure feature names."""
        return [
            "probable_swing_low",
            "probable_swing_high",
            "bars_since_swing_low",
            "bars_since_swing_high",
            "consecutive_up_bars",
            "consecutive_down_bars",
            "uptrend_exhaustion",
            "downtrend_exhaustion",
            "higher_high",
            "lower_low",
            "higher_low",
            "lower_high",
            "trend_structure",
            "recent_swing_high",
            "recent_swing_low",
            "dist_to_swing_high",
            "dist_to_swing_low",
            "near_support",
            "near_resistance",
            "range_position",
            "new_20_high",
            "new_20_low",
        ]
