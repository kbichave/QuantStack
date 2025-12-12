"""
Gann features for trading analysis.

Implements Gann swing/pivot points and retracement levels:
- ATR-based swing detection (Gann-style)
- Retracement levels (38.2%, 50%, 61.8%)
- Price-time relationships
- Gann angles from swing points
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class GannFeatures(FeatureBase):
    """
    Gann-style technical analysis features.

    Features:
    - Swing high/low detection (ATR-based, no lookahead)
    - Retracement levels (38.2%, 50%, 61.8%)
    - Distance to retracement levels
    - Price-time ratios
    - Gann angles (1x1, 2x1, 1x2)
    """

    def __init__(
        self,
        timeframe: Timeframe,
        atr_mult: float = 1.5,
        min_swing_bars: int = 3,
    ):
        """
        Initialize Gann features calculator.

        Args:
            timeframe: Timeframe for parameter adjustment
            atr_mult: ATR multiple for swing reversal detection
            min_swing_bars: Minimum bars between swings
        """
        super().__init__(timeframe)

        # Adjust parameters based on timeframe
        if timeframe == Timeframe.H1:
            self.atr_mult = atr_mult
            self.min_swing_bars = min_swing_bars
        elif timeframe == Timeframe.H4:
            self.atr_mult = atr_mult * 1.2
            self.min_swing_bars = max(2, min_swing_bars - 1)
        elif timeframe == Timeframe.D1:
            self.atr_mult = atr_mult * 1.5
            self.min_swing_bars = max(2, min_swing_bars - 1)
        else:  # Weekly
            self.atr_mult = atr_mult * 2.0
            self.min_swing_bars = 2

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Gann features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with Gann features added
        """
        result = df.copy()

        if len(df) < 20:
            return self._add_empty_features(result)

        # Compute ATR if not present
        if "atr" not in result.columns:
            result["atr"] = self._compute_atr(result)

        # Detect Gann-style swings
        swing_high, swing_low = self._detect_gann_swings(result)
        result["gann_swing_high"] = swing_high
        result["gann_swing_low"] = swing_low

        # Get recent swing prices
        result["gann_recent_high"] = self._get_recent_swing_price(
            result["high"], swing_high
        )
        result["gann_recent_low"] = self._get_recent_swing_price(
            result["low"], swing_low
        )

        # Compute retracement levels
        result = self._compute_retracement_levels(result)

        # Distance to retracement levels
        result = self._compute_distance_to_retracements(result)

        # Bars since last swing
        result["gann_bars_since_swing_high"] = self._bars_since_signal(swing_high)
        result["gann_bars_since_swing_low"] = self._bars_since_signal(swing_low)

        # Time ratios
        result = self._compute_time_ratios(result)

        # Price-time square
        result = self._compute_price_time_square(result)

        # Gann angles
        result = self._compute_gann_angles(result)

        # Near retracement signals
        result = self._compute_near_retracement_signals(result)

        return result

    def _detect_gann_swings(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect Gann-style swing points using ATR threshold.

        No lookahead - swings are confirmed only after reversal.
        """
        n = len(df)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atr = df["atr"].values

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

            threshold = self.atr_mult * atr[i]

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
                    if (
                        move_down >= threshold
                        and (i - last_swing_idx) >= self.min_swing_bars
                    ):
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
                    if (
                        move_up >= threshold
                        and (i - last_swing_idx) >= self.min_swing_bars
                    ):
                        swing_low[last_swing_idx] = 1
                        direction = "up"
                        last_swing_idx = i
                        last_swing_price = highs[i]

        return (
            pd.Series(swing_high, index=df.index),
            pd.Series(swing_low, index=df.index),
        )

    def _get_recent_swing_price(
        self,
        price: pd.Series,
        swing_signal: pd.Series,
        lookback: int = 50,
    ) -> pd.Series:
        """Get the price value at the most recent swing point."""
        swing_prices = price.where(swing_signal == 1, np.nan)
        return swing_prices.ffill(limit=lookback)

    def _compute_retracement_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Gann/Fibonacci retracement levels.

        Uses recent swing high and low to calculate levels.
        """
        result = df.copy()

        high = result["gann_recent_high"]
        low = result["gann_recent_low"]
        swing_range = high - low

        # Key Gann retracement levels
        result["gann_retracement_382"] = low + swing_range * 0.382
        result["gann_retracement_500"] = low + swing_range * 0.500
        result["gann_retracement_618"] = low + swing_range * 0.618

        # Additional Gann levels
        result["gann_retracement_250"] = low + swing_range * 0.250
        result["gann_retracement_750"] = low + swing_range * 0.750

        return result

    def _compute_distance_to_retracements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute distance from current price to retracement levels."""
        result = df.copy()
        close = result["close"]

        # Distance to each level (percentage)
        result["gann_dist_to_382"] = (
            (close - result["gann_retracement_382"]) / close * 100
        )
        result["gann_dist_to_500"] = (
            (close - result["gann_retracement_500"]) / close * 100
        )
        result["gann_dist_to_618"] = (
            (close - result["gann_retracement_618"]) / close * 100
        )

        # Distance to nearest retracement level
        distances = pd.concat(
            [
                result["gann_dist_to_382"].abs(),
                result["gann_dist_to_500"].abs(),
                result["gann_dist_to_618"].abs(),
            ],
            axis=1,
        )
        result["gann_dist_to_nearest"] = distances.min(axis=1)

        # Position relative to swing range (0 = at low, 100 = at high)
        swing_range = result["gann_recent_high"] - result["gann_recent_low"]
        swing_range_safe = swing_range.replace(0, np.nan)
        result["gann_range_position"] = (
            (close - result["gann_recent_low"]) / swing_range_safe * 100
        )

        return result

    def _bars_since_signal(self, signal: pd.Series) -> pd.Series:
        """Count bars since last signal."""
        groups = signal.cumsum()
        counts = signal.groupby(groups).cumcount()
        counts = counts.where(groups > 0, np.nan)
        return counts

    def _compute_time_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Gann time ratios.

        Time ratios measure the relationship between time elapsed
        and typical swing durations.
        """
        result = df.copy()

        bars_high = result["gann_bars_since_swing_high"]
        bars_low = result["gann_bars_since_swing_low"]

        # Average swing duration (rolling)
        avg_swing_duration = (
            bars_high.rolling(20).mean() + bars_low.rolling(20).mean()
        ) / 2
        avg_swing_duration_safe = avg_swing_duration.replace(0, np.nan)

        # Time ratio: current bars / average swing duration
        result["gann_time_ratio_high"] = bars_high / avg_swing_duration_safe
        result["gann_time_ratio_low"] = bars_low / avg_swing_duration_safe

        # Time exhaustion signals (>1.5 means extended)
        result["gann_time_extended_high"] = (
            result["gann_time_ratio_high"] > 1.5
        ).astype(int)
        result["gann_time_extended_low"] = (result["gann_time_ratio_low"] > 1.5).astype(
            int
        )

        return result

    def _compute_price_time_square(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Gann price-time square.

        The price-time square measures balance between price movement
        and time elapsed since last swing.
        """
        result = df.copy()
        close = result["close"]

        # Price change since swing high
        price_change_from_high = (result["gann_recent_high"] - close).abs()
        bars_from_high = result["gann_bars_since_swing_high"]

        # Price change since swing low
        price_change_from_low = (close - result["gann_recent_low"]).abs()
        bars_from_low = result["gann_bars_since_swing_low"]

        # ATR-normalized price change
        atr = result["atr"]
        atr_safe = atr.replace(0, np.nan)

        price_units_high = price_change_from_high / atr_safe
        price_units_low = price_change_from_low / atr_safe

        # Price-time square balance (1.0 = perfectly balanced)
        # When price_units == bars, price and time are in balance
        bars_high_safe = bars_from_high.replace(0, np.nan)
        bars_low_safe = bars_from_low.replace(0, np.nan)

        result["gann_pt_square_high"] = price_units_high / bars_high_safe
        result["gann_pt_square_low"] = price_units_low / bars_low_safe

        # Combined balance indicator
        result["gann_pt_balance"] = (
            result["gann_pt_square_high"] + result["gann_pt_square_low"]
        ) / 2

        return result

    def _compute_gann_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Gann angles from swing points.

        Gann angles represent price-time relationships:
        - 1x1 (45°): 1 price unit per 1 time unit
        - 2x1 (63.75°): 2 price units per 1 time unit (strong uptrend)
        - 1x2 (26.25°): 1 price unit per 2 time units (weak uptrend)
        """
        result = df.copy()
        close = result["close"]
        atr = result["atr"]
        atr_safe = atr.replace(0, np.nan)

        # Calculate angles from swing low (upward)
        bars_from_low = result["gann_bars_since_swing_low"]
        bars_from_low_safe = bars_from_low.replace(0, np.nan)

        # Price per bar (in ATR units) from swing low
        price_change_from_low = close - result["gann_recent_low"]
        price_per_bar_low = (price_change_from_low / atr_safe) / bars_from_low_safe

        # 1x1 angle: expected price at current time if 1:1 ratio
        result["gann_1x1_from_low"] = result["gann_recent_low"] + (bars_from_low * atr)
        result["gann_2x1_from_low"] = result["gann_recent_low"] + (
            bars_from_low * atr * 2
        )
        result["gann_1x2_from_low"] = result["gann_recent_low"] + (
            bars_from_low * atr * 0.5
        )

        # Price relative to Gann angles (positive = above angle)
        result["gann_vs_1x1_low"] = (close - result["gann_1x1_from_low"]) / atr_safe
        result["gann_vs_2x1_low"] = (close - result["gann_2x1_from_low"]) / atr_safe
        result["gann_vs_1x2_low"] = (close - result["gann_1x2_from_low"]) / atr_safe

        # Calculate angles from swing high (downward)
        bars_from_high = result["gann_bars_since_swing_high"]
        bars_from_high_safe = bars_from_high.replace(0, np.nan)

        result["gann_1x1_from_high"] = result["gann_recent_high"] - (
            bars_from_high * atr
        )
        result["gann_2x1_from_high"] = result["gann_recent_high"] - (
            bars_from_high * atr * 2
        )
        result["gann_1x2_from_high"] = result["gann_recent_high"] - (
            bars_from_high * atr * 0.5
        )

        result["gann_vs_1x1_high"] = (close - result["gann_1x1_from_high"]) / atr_safe
        result["gann_vs_2x1_high"] = (close - result["gann_2x1_from_high"]) / atr_safe
        result["gann_vs_1x2_high"] = (close - result["gann_1x2_from_high"]) / atr_safe

        # Angle strength indicator (how steep is current move)
        result["gann_angle_strength"] = price_per_bar_low

        return result

    def _compute_near_retracement_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute signals for price near retracement levels."""
        result = df.copy()

        # Near retracement level (within 1% of price)
        threshold = 1.0  # 1%

        result["gann_near_382"] = (result["gann_dist_to_382"].abs() < threshold).astype(
            int
        )
        result["gann_near_500"] = (result["gann_dist_to_500"].abs() < threshold).astype(
            int
        )
        result["gann_near_618"] = (result["gann_dist_to_618"].abs() < threshold).astype(
            int
        )

        # Any retracement level
        result["gann_near_any_level"] = (
            result["gann_near_382"] | result["gann_near_500"] | result["gann_near_618"]
        ).astype(int)

        # Oversold (below 38.2%) or overbought (above 61.8%)
        result["gann_oversold"] = (result["gann_range_position"] < 38.2).astype(int)
        result["gann_overbought"] = (result["gann_range_position"] > 61.8).astype(int)

        return result

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()

    def _add_empty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty Gann features for small DataFrames."""
        result = df.copy()
        for name in self.get_feature_names():
            result[name] = np.nan
        return result

    def get_feature_names(self) -> List[str]:
        """Return list of Gann feature names."""
        return [
            # Swing detection
            "gann_swing_high",
            "gann_swing_low",
            "gann_recent_high",
            "gann_recent_low",
            "gann_bars_since_swing_high",
            "gann_bars_since_swing_low",
            # Retracement levels
            "gann_retracement_382",
            "gann_retracement_500",
            "gann_retracement_618",
            "gann_retracement_250",
            "gann_retracement_750",
            # Distance to retracements
            "gann_dist_to_382",
            "gann_dist_to_500",
            "gann_dist_to_618",
            "gann_dist_to_nearest",
            "gann_range_position",
            # Time ratios
            "gann_time_ratio_high",
            "gann_time_ratio_low",
            "gann_time_extended_high",
            "gann_time_extended_low",
            # Price-time square
            "gann_pt_square_high",
            "gann_pt_square_low",
            "gann_pt_balance",
            # Gann angles from low
            "gann_1x1_from_low",
            "gann_2x1_from_low",
            "gann_1x2_from_low",
            "gann_vs_1x1_low",
            "gann_vs_2x1_low",
            "gann_vs_1x2_low",
            # Gann angles from high
            "gann_1x1_from_high",
            "gann_2x1_from_high",
            "gann_1x2_from_high",
            "gann_vs_1x1_high",
            "gann_vs_2x1_high",
            "gann_vs_1x2_high",
            "gann_angle_strength",
            # Near retracement signals
            "gann_near_382",
            "gann_near_500",
            "gann_near_618",
            "gann_near_any_level",
            "gann_oversold",
            "gann_overbought",
        ]
