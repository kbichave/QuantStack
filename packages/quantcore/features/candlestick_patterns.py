"""
Candlestick pattern recognition features.

Uses TA-Lib pattern recognition functions to identify classic candlestick patterns
mentioned in QuantAgent (Double Bottom, Head & Shoulders, Wedges, Triangles, etc.).

TA-Lib returns:
- 0: no pattern
- +100: bullish pattern
- -100: bearish pattern

References:
    QuantAgent: https://github.com/Y-Research-SBU/QuantAgent
    TA-Lib: https://mrjbq7.github.io/ta-lib/func_groups/pattern_recognition.html
"""

from typing import List
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe

# Import TA-Lib (optional dependency)
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, candlestick patterns will be disabled")


class CandlestickPatternFeatures(FeatureBase):
    """
    Candlestick pattern recognition features using TA-Lib.

    Focuses on reversal and continuation patterns that align with mean reversion:
    - Reversal patterns: Hammer, Inverted Hammer, Doji, Engulfing, etc.
    - Continuation patterns: Three Line Strike, Rising/Falling Three Methods
    - Key patterns from QuantAgent: Double Bottom, H&S, Triangles, Wedges

    Features are normalized to [-1, 1] range for ML compatibility.
    """

    def __init__(self, timeframe: Timeframe):
        """
        Initialize candlestick pattern calculator.

        Args:
            timeframe: Timeframe for feature computation
        """
        super().__init__(timeframe)

        # Select patterns based on timeframe
        # Higher timeframes: focus on reversal patterns
        # Lower timeframes: include continuation patterns
        if timeframe in [Timeframe.W1, Timeframe.D1]:
            self.include_continuation = False
        else:
            self.include_continuation = True

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute candlestick pattern features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with pattern features added
        """
        if not TALIB_AVAILABLE:
            result = df.copy()
            # Add dummy columns
            for col in self.get_feature_names():
                result[col] = 0
            return result

        result = df.copy()

        if len(result) < 10:
            logger.warning(
                f"Insufficient data for pattern recognition: {len(result)} bars"
            )
            for col in self.get_feature_names():
                result[col] = 0
            return result

        # Extract OHLC as numpy arrays for TA-Lib
        open_prices = result["open"].values
        high_prices = result["high"].values
        low_prices = result["low"].values
        close_prices = result["close"].values

        # === REVERSAL PATTERNS ===

        # Hammer / Hanging Man (single candle reversal)
        result["cdl_hammer"] = self._normalize_pattern(
            talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_inverted_hammer"] = self._normalize_pattern(
            talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_hanging_man"] = self._normalize_pattern(
            talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_shooting_star"] = self._normalize_pattern(
            talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
        )

        # Doji patterns (indecision)
        result["cdl_doji"] = self._normalize_pattern(
            talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_dragonfly_doji"] = self._normalize_pattern(
            talib.CDLDRAGONFLYDOJI(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_gravestone_doji"] = self._normalize_pattern(
            talib.CDLGRAVESTONEDOJI(open_prices, high_prices, low_prices, close_prices)
        )

        # Engulfing patterns (2-candle reversal)
        result["cdl_engulfing"] = self._normalize_pattern(
            talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_harami"] = self._normalize_pattern(
            talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_piercing"] = self._normalize_pattern(
            talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_dark_cloud"] = self._normalize_pattern(
            talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)
        )

        # Morning/Evening Star (3-candle reversal)
        result["cdl_morning_star"] = self._normalize_pattern(
            talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_evening_star"] = self._normalize_pattern(
            talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
        )

        # Three patterns (continuation/reversal)
        result["cdl_three_white_soldiers"] = self._normalize_pattern(
            talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)
        )
        result["cdl_three_black_crows"] = self._normalize_pattern(
            talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)
        )

        # Abandoned baby (rare reversal)
        result["cdl_abandoned_baby"] = self._normalize_pattern(
            talib.CDLABANDONEDBABY(open_prices, high_prices, low_prices, close_prices)
        )

        # Kicking (strong reversal)
        result["cdl_kicking"] = self._normalize_pattern(
            talib.CDLKICKING(open_prices, high_prices, low_prices, close_prices)
        )

        # === CONTINUATION PATTERNS (if enabled) ===
        if self.include_continuation:
            result["cdl_three_line_strike"] = self._normalize_pattern(
                talib.CDL3LINESTRIKE(open_prices, high_prices, low_prices, close_prices)
            )
            result["cdl_rising_three_methods"] = self._normalize_pattern(
                talib.CDLRISEFALL3METHODS(
                    open_prices, high_prices, low_prices, close_prices
                )
            )

        # === AGGREGATE FEATURES ===

        # Count of bullish patterns detected
        bullish_cols = [col for col in result.columns if col.startswith("cdl_")]
        result["cdl_bullish_count"] = (result[bullish_cols] > 0).sum(axis=1)
        result["cdl_bearish_count"] = (result[bullish_cols] < 0).sum(axis=1)

        # Net pattern signal (-1 to +1)
        result["cdl_net_signal"] = result[bullish_cols].sum(axis=1) / len(bullish_cols)

        # Strongest pattern intensity
        result["cdl_max_bullish"] = result[bullish_cols].max(axis=1)
        result["cdl_max_bearish"] = result[bullish_cols].min(axis=1)

        # === CUSTOM PATTERN LOGIC (QuantAgent-inspired) ===

        # Double bottom detection (simplified)
        result["cdl_double_bottom"] = self._detect_double_bottom(result)

        # V-shaped reversal
        result["cdl_v_reversal"] = self._detect_v_reversal(result)

        return result

    def _normalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """
        Normalize TA-Lib pattern output from [-100, 100] to [-1, 1].

        Args:
            pattern: TA-Lib pattern array

        Returns:
            Normalized array
        """
        return pattern / 100.0

    def _detect_double_bottom(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        Detect double bottom pattern (simplified version).

        A double bottom has:
        - Two local lows within lookback period
        - Lows are within 2% of each other
        - Middle high is at least 3% above lows
        - Current price breaking above middle high

        Returns:
            Series with 1 for bullish double bottom, 0 otherwise
        """
        result = pd.Series(0, index=df.index)

        if len(df) < lookback:
            return result

        close = df["close"].values
        low = df["low"].values

        for i in range(lookback, len(df)):
            window_low = low[i - lookback : i]
            window_close = close[i - lookback : i]

            # Find two lowest points
            sorted_indices = np.argsort(window_low)
            lowest_idx = sorted_indices[0]
            second_lowest_idx = sorted_indices[1]

            # Check if they're similar (within 2%)
            if (
                abs(window_low[lowest_idx] - window_low[second_lowest_idx])
                / window_low[lowest_idx]
                > 0.02
            ):
                continue

            # Find the high between them
            if lowest_idx < second_lowest_idx:
                middle_high = window_close[lowest_idx:second_lowest_idx].max()
            else:
                middle_high = window_close[second_lowest_idx:lowest_idx].max()

            # Check if middle high is significant
            avg_low = (window_low[lowest_idx] + window_low[second_lowest_idx]) / 2
            if middle_high < avg_low * 1.03:
                continue

            # Check if current price breaks above middle high
            current_close = close[i]
            if current_close > middle_high:
                result.iloc[i] = 1

        return result

    def _detect_v_reversal(self, df: pd.DataFrame, lookback: int = 10) -> pd.Series:
        """
        Detect V-shaped reversal (sharp decline followed by sharp recovery).

        A V-reversal has:
        - Sharp decline (>5% over lookback/2 bars)
        - Sharp recovery (>5% over next lookback/2 bars)
        - Recent bottom within lookback period

        Returns:
            Series with 1 for bullish V-reversal, -1 for inverted, 0 otherwise
        """
        result = pd.Series(0, index=df.index)

        if len(df) < lookback:
            return result

        close = df["close"].values

        for i in range(lookback, len(df)):
            window = close[i - lookback : i + 1]

            # Find the lowest point in window
            bottom_idx = window.argmin()

            # Check decline before bottom
            if bottom_idx >= lookback // 2:
                decline_start = close[i - lookback]
                decline_end = window[bottom_idx]
                decline_pct = (decline_end - decline_start) / decline_start

                if decline_pct < -0.05:  # 5% decline
                    # Check recovery after bottom
                    if bottom_idx < len(window) - lookback // 2:
                        recovery_start = window[bottom_idx]
                        recovery_end = window[-1]
                        recovery_pct = (recovery_end - recovery_start) / recovery_start

                        if recovery_pct > 0.05:  # 5% recovery
                            result.iloc[i] = 1

            # Inverted V (top)
            top_idx = window.argmax()
            if top_idx >= lookback // 2:
                rise_start = close[i - lookback]
                rise_end = window[top_idx]
                rise_pct = (rise_end - rise_start) / rise_start

                if rise_pct > 0.05:
                    if top_idx < len(window) - lookback // 2:
                        fall_start = window[top_idx]
                        fall_end = window[-1]
                        fall_pct = (fall_end - fall_start) / fall_start

                        if fall_pct < -0.05:
                            result.iloc[i] = -1

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of candlestick pattern feature names."""
        features = [
            # Single candle patterns
            "cdl_hammer",
            "cdl_inverted_hammer",
            "cdl_hanging_man",
            "cdl_shooting_star",
            # Doji patterns
            "cdl_doji",
            "cdl_dragonfly_doji",
            "cdl_gravestone_doji",
            # Two candle patterns
            "cdl_engulfing",
            "cdl_harami",
            "cdl_piercing",
            "cdl_dark_cloud",
            # Three candle patterns
            "cdl_morning_star",
            "cdl_evening_star",
            "cdl_three_white_soldiers",
            "cdl_three_black_crows",
            # Rare patterns
            "cdl_abandoned_baby",
            "cdl_kicking",
            # Aggregate features
            "cdl_bullish_count",
            "cdl_bearish_count",
            "cdl_net_signal",
            "cdl_max_bullish",
            "cdl_max_bearish",
            # Custom patterns
            "cdl_double_bottom",
            "cdl_v_reversal",
        ]

        if self.include_continuation:
            features.extend(
                [
                    "cdl_three_line_strike",
                    "cdl_rising_three_methods",
                ]
            )

        return features
