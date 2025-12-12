"""
QuantAgents-style pattern features for price action analysis.

Provides higher-level pattern detection beyond candlestick patterns:
- Pullback detection (retracement in trending market)
- Breakout attempts (testing resistance/support)
- Consolidation detection (range-bound, low volatility)
- Bar sequence patterns (consecutive moves)
- Swing-based pattern roles (if swing data available)
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class QuantAgentsPatternFeatures(FeatureBase):
    """
    QuantAgents-inspired pattern recognition features.

    Features:
    - Pullback detection in uptrend/downtrend
    - Breakout attempt detection
    - Consolidation/range detection
    - Bar sequence patterns (consecutive up/down bars)
    - Swing-based pattern roles (if swing data available)
    """

    def __init__(self, timeframe: Timeframe, lookback_period: int = 20):
        """
        Initialize QuantAgents pattern feature calculator.

        Args:
            timeframe: Timeframe for parameter selection
            lookback_period: Lookback period for pattern detection
        """
        super().__init__(timeframe)
        self.lookback_period = lookback_period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute QuantAgents pattern features.

        Args:
            df: OHLCV DataFrame (optionally with swing data)

        Returns:
            DataFrame with pattern features added
        """
        result = df.copy()
        close = result["close"]
        high = result["high"]
        low = result["low"]

        # Pullback detection (requires trend context)
        result["qa_pattern_is_pullback"] = self._detect_pullback(
            close, high, low, self.lookback_period
        )

        # Breakout attempt detection
        result["qa_pattern_is_breakout"] = self._detect_breakout_attempt(
            close, high, low, self.lookback_period
        )

        # Consolidation detection (range-bound + low volatility)
        result["qa_pattern_consolidation"] = self._detect_consolidation(
            close, high, low, self.lookback_period
        )

        # Bar sequence patterns
        result["qa_pattern_bars_up_streak"] = self._count_consecutive_bars(
            close, direction="up"
        )
        result["qa_pattern_bars_down_streak"] = self._count_consecutive_bars(
            close, direction="down"
        )

        # Range position (where is price within recent range?)
        result["qa_pattern_range_position"] = self._compute_range_position(
            close, high, low, self.lookback_period
        )

        # Volatility regime (high/low volatility)
        if "atr" in result.columns:
            result["qa_pattern_vol_regime"] = self._classify_volatility_regime(
                result["atr"], self.lookback_period
            )
        else:
            result["qa_pattern_vol_regime"] = 0  # Neutral

        # Swing-based patterns (if swing data available)
        if self._has_swing_data(result):
            result["qa_pattern_swing_pullback"] = self._detect_swing_pullback(result)
            result["qa_pattern_swing_bounce"] = self._detect_swing_bounce(result)
        else:
            result["qa_pattern_swing_pullback"] = 0
            result["qa_pattern_swing_bounce"] = 0

        # Mean reversion opportunity (high z-score + consolidation exit)
        if "zscore_price" in result.columns:
            result["qa_pattern_mr_opportunity"] = self._detect_mr_opportunity(
                result["zscore_price"],
                result["qa_pattern_consolidation"],
            )
        else:
            result["qa_pattern_mr_opportunity"] = 0

        return result

    def _detect_pullback(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Detect pullback: price retracement in trending market.

        Pullback criteria:
        - Recent trend exists (price moved significantly)
        - Current bar shows retracement (against trend)
        - But retracement is < 50% of recent move

        Returns:
            Series with values: 1 (pullback in uptrend), -1 (pullback in downtrend), 0 (none)
        """
        pullback = pd.Series(0, index=close.index, dtype=int)

        for i in range(period, len(close)):
            # Look at recent price action
            recent_close = close.iloc[i - period : i + 1]
            recent_high = high.iloc[i - period : i + 1]
            recent_low = low.iloc[i - period : i + 1]

            range_start = recent_low.min()
            range_end = recent_high.max()
            range_size = range_end - range_start

            if range_size == 0:
                continue

            current_price = close.iloc[i]

            # Check for uptrend pullback
            # Price recently made higher highs, now pulling back
            if recent_high.iloc[-1] > recent_high.iloc[-period // 2]:
                # Recent higher high
                recent_peak = recent_high.max()
                pullback_size = recent_peak - current_price

                # Is this a pullback (not a reversal)?
                if 0.1 < (pullback_size / range_size) < 0.5:
                    pullback.iloc[i] = 1

            # Check for downtrend pullback (bounce)
            # For a bounce: we made lower lows recently (downtrend), now bouncing
            # Check if trough was made in the latter part of the period
            elif (
                recent_low.iloc[-period // 2 :].min()
                < recent_low.iloc[: -period // 2].min()
            ):
                # Lower lows in second half = recent downtrend
                recent_trough = recent_low.min()
                pullback_size = current_price - recent_trough

                if 0.1 < (pullback_size / range_size) < 0.5:
                    pullback.iloc[i] = -1

        return pullback

    def _detect_breakout_attempt(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Detect breakout attempt: price testing resistance or support.

        Breakout criteria:
        - Price within 2% of recent high (resistance test) or low (support test)
        - Recent consolidation or range-bound behavior

        Returns:
            Series with values: 1 (testing resistance), -1 (testing support), 0 (none)
        """
        breakout = pd.Series(0, index=close.index, dtype=int)

        for i in range(period, len(close)):
            recent_high = high.iloc[i - period : i].max()
            recent_low = low.iloc[i - period : i].min()
            current_price = close.iloc[i]

            range_size = recent_high - recent_low
            if range_size == 0:
                continue

            # Testing resistance
            if (recent_high - current_price) / range_size < 0.02:
                breakout.iloc[i] = 1

            # Testing support
            elif (current_price - recent_low) / range_size < 0.02:
                breakout.iloc[i] = -1

        return breakout

    def _detect_consolidation(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        period: int,
        range_threshold: float = 0.05,  # 5% range
    ) -> pd.Series:
        """
        Detect consolidation: range-bound price with low volatility.

        Consolidation criteria:
        - Recent range is small relative to price level
        - Price stays within range

        Returns:
            Series with values: 1 (consolidating), 0 (not consolidating)
        """
        consolidation = pd.Series(0, index=close.index, dtype=int)

        for i in range(period, len(close)):
            recent_high = high.iloc[i - period : i + 1].max()
            recent_low = low.iloc[i - period : i + 1].min()
            avg_price = close.iloc[i - period : i + 1].mean()

            range_pct = (recent_high - recent_low) / avg_price

            if range_pct < range_threshold:
                consolidation.iloc[i] = 1

        return consolidation

    def _count_consecutive_bars(
        self,
        close: pd.Series,
        direction: str = "up",
    ) -> pd.Series:
        """
        Count consecutive up/down bars.

        Args:
            close: Close price series
            direction: "up" or "down"

        Returns:
            Series with count of consecutive bars
        """
        bar_change = close.diff()

        streak = pd.Series(0, index=close.index, dtype=int)
        current_streak = 0

        for i in range(1, len(close)):
            change = bar_change.iloc[i]

            if pd.isna(change):
                current_streak = 0
            elif (direction == "up" and change > 0) or (
                direction == "down" and change < 0
            ):
                current_streak += 1
            else:
                current_streak = 0

            streak.iloc[i] = current_streak

        return streak

    def _compute_range_position(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute where price is within recent range.

        Returns:
            Series with values 0-1 (0=at low, 1=at high)
        """
        position = pd.Series(np.nan, index=close.index)

        for i in range(period, len(close)):
            recent_high = high.iloc[i - period : i + 1].max()
            recent_low = low.iloc[i - period : i + 1].min()
            current_price = close.iloc[i]

            range_size = recent_high - recent_low
            if range_size > 0:
                position.iloc[i] = (current_price - recent_low) / range_size

        return position

    def _classify_volatility_regime(
        self,
        atr: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Classify volatility regime relative to recent history.

        Returns:
            Series with values: 1 (high vol), 0 (normal), -1 (low vol)
        """
        regime = pd.Series(0, index=atr.index, dtype=int)

        atr_ma = atr.rolling(window=period).mean()
        atr_std = atr.rolling(window=period).std()

        # High volatility: ATR > mean + 0.5*std
        high_vol_mask = atr > (atr_ma + 0.5 * atr_std)
        regime[high_vol_mask] = 1

        # Low volatility: ATR < mean - 0.5*std
        low_vol_mask = atr < (atr_ma - 0.5 * atr_std)
        regime[low_vol_mask] = -1

        return regime

    def _has_swing_data(self, df: pd.DataFrame) -> bool:
        """Check if swing data is available."""
        return (
            "probable_swing_high" in df.columns and "probable_swing_low" in df.columns
        )

    def _detect_swing_pullback(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect pullback from swing high (potential mean reversion entry).

        Returns:
            Series with values: 1 (pullback from high), 0 (none)
        """
        pullback = pd.Series(0, index=df.index, dtype=int)

        if not self._has_swing_data(df):
            return pullback

        close = df["close"]
        high = df["high"]
        swing_high = df["probable_swing_high"]

        # Track the most recent swing high within lookback window
        lookback = min(
            self.lookback_period, 10
        )  # Use shorter lookback for swing detection

        for i in range(1, len(df)):
            # Look for recent swing high within lookback window
            start_idx = max(0, i - lookback)
            recent_swing_highs = swing_high.iloc[start_idx:i]

            if recent_swing_highs.sum() > 0:
                # Find the most recent swing high
                swing_idx = recent_swing_highs[recent_swing_highs == 1].index[-1]
                swing_pos = df.index.get_loc(swing_idx)
                high_price = high.iloc[swing_pos]
                current_price = close.iloc[i]

                if current_price < high_price * 0.98:  # 2% pullback
                    pullback.iloc[i] = 1

        return pullback

    def _detect_swing_bounce(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect bounce from swing low (potential mean reversion entry).

        Returns:
            Series with values: 1 (bounce from low), 0 (none)
        """
        bounce = pd.Series(0, index=df.index, dtype=int)

        if not self._has_swing_data(df):
            return bounce

        close = df["close"]
        swing_low = df["probable_swing_low"]

        for i in range(1, len(df)):
            # Recent swing low occurred
            if swing_low.iloc[i - 1] == 1:
                # Check if price is bouncing
                low_price = df["low"].iloc[i - 1]
                current_price = close.iloc[i]

                if current_price > low_price * 1.02:  # 2% bounce
                    bounce.iloc[i] = 1

        return bounce

    def _detect_mr_opportunity(
        self,
        zscore: pd.Series,
        consolidation: pd.Series,
    ) -> pd.Series:
        """
        Detect mean reversion opportunity.

        MR opportunity = extreme z-score + exiting consolidation

        Returns:
            Series with values: 1 (long opportunity), -1 (short opportunity), 0 (none)
        """
        opportunity = pd.Series(0, index=zscore.index, dtype=int)

        # Long opportunity: high negative z-score (oversold) + consolidation
        long_mask = (zscore < -2.0) & (consolidation == 1)
        opportunity[long_mask] = 1

        # Short opportunity: high positive z-score (overbought) + consolidation
        short_mask = (zscore > 2.0) & (consolidation == 1)
        opportunity[short_mask] = -1

        return opportunity

    def get_feature_names(self) -> List[str]:
        """Return list of QuantAgents pattern feature names."""
        return [
            "qa_pattern_is_pullback",
            "qa_pattern_is_breakout",
            "qa_pattern_consolidation",
            "qa_pattern_bars_up_streak",
            "qa_pattern_bars_down_streak",
            "qa_pattern_range_position",
            "qa_pattern_vol_regime",
            "qa_pattern_swing_pullback",
            "qa_pattern_swing_bounce",
            "qa_pattern_mr_opportunity",
        ]
