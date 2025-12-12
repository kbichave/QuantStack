"""
QuantAgents-style trend features for multi-horizon trend analysis.

Provides higher-level trend abstractions beyond basic EMAs:
- Multi-horizon regression slopes (short/medium/long term)
- Trend regime classification (down/sideways/up)
- Trend quality metrics (R² for trend linearity)
- Trend strength (slope normalized by volatility)
"""

from typing import List
import pandas as pd
import numpy as np
from scipy import stats

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class QuantAgentsTrendFeatures(FeatureBase):
    """
    QuantAgents-inspired trend analysis features.

    Features:
    - Multi-horizon trend slopes (short, medium, long)
    - Trend regime classification (-1/0/+1)
    - Trend strength (normalized by ATR)
    - Trend quality (R² of linear regression)
    - Trend consistency (% of bars aligned with trend)
    """

    def __init__(self, timeframe: Timeframe):
        """
        Initialize QuantAgents trend feature calculator.

        Args:
            timeframe: Timeframe for parameter selection
        """
        super().__init__(timeframe)

        # Multi-horizon windows (scaled by timeframe)
        if timeframe == Timeframe.H1:
            self.window_short = 10  # ~10 hours
            self.window_med = 30  # ~30 hours
            self.window_long = 100  # ~100 hours
        elif timeframe == Timeframe.H4:
            self.window_short = 8  # ~32 hours
            self.window_med = 20  # ~80 hours
            self.window_long = 60  # ~240 hours
        elif timeframe == Timeframe.D1:
            self.window_short = 5  # ~1 week
            self.window_med = 20  # ~1 month
            self.window_long = 60  # ~3 months
        else:  # Weekly
            self.window_short = 4  # ~1 month
            self.window_med = 13  # ~3 months
            self.window_long = 26  # ~6 months

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute QuantAgents trend features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with trend features added
        """
        result = df.copy()
        close = result["close"]

        # Multi-horizon regression slopes
        result["qa_trend_slope_short"] = self._compute_normalized_slope(
            close, self.window_short
        )
        result["qa_trend_slope_med"] = self._compute_normalized_slope(
            close, self.window_med
        )
        result["qa_trend_slope_long"] = self._compute_normalized_slope(
            close, self.window_long
        )

        # Trend quality (R² for each horizon)
        result["qa_trend_quality_short"] = self._compute_r_squared(
            close, self.window_short
        )
        result["qa_trend_quality_med"] = self._compute_r_squared(close, self.window_med)
        result["qa_trend_quality_long"] = self._compute_r_squared(
            close, self.window_long
        )

        # Trend regime classification (using medium window)
        result["qa_trend_regime"] = self._classify_trend_regime(
            result["qa_trend_slope_med"],
            result["qa_trend_quality_med"],
        )

        # Trend strength (slope normalized by ATR if available)
        if "atr" in result.columns:
            result["qa_trend_strength_short"] = self._compute_trend_strength(
                result["qa_trend_slope_short"], result["atr"]
            )
            result["qa_trend_strength_med"] = self._compute_trend_strength(
                result["qa_trend_slope_med"], result["atr"]
            )
            result["qa_trend_strength_long"] = self._compute_trend_strength(
                result["qa_trend_slope_long"], result["atr"]
            )
        else:
            # Fallback to absolute slope if ATR not available
            result["qa_trend_strength_short"] = result["qa_trend_slope_short"].abs()
            result["qa_trend_strength_med"] = result["qa_trend_slope_med"].abs()
            result["qa_trend_strength_long"] = result["qa_trend_slope_long"].abs()

        # Trend consistency (% of recent bars aligned with trend)
        result["qa_trend_consistency"] = self._compute_trend_consistency(
            close, result["qa_trend_slope_med"], self.window_short
        )

        # Trend alignment across horizons (all 3 agree?)
        result["qa_trend_alignment_score"] = self._compute_multi_horizon_alignment(
            result["qa_trend_slope_short"],
            result["qa_trend_slope_med"],
            result["qa_trend_slope_long"],
        )

        return result

    def _compute_normalized_slope(
        self,
        series: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute rolling linear regression slope normalized by price.

        Args:
            series: Price series
            period: Lookback period

        Returns:
            Normalized slope series (% per bar)
        """

        def calc_slope(window):
            if len(window) < period // 2:
                return np.nan
            x = np.arange(len(window))
            try:
                slope, _, _, _, _ = stats.linregress(x, window)
                # Normalize by mean price to get % change per bar
                return slope / window.mean() * 100
            except (ValueError, RuntimeWarning):
                return np.nan

        return series.rolling(window=period).apply(calc_slope, raw=False)

    def _compute_r_squared(
        self,
        series: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute R² of linear regression to measure trend quality.

        Higher R² = more linear trend, lower R² = choppy/non-trending

        Args:
            series: Price series
            period: Lookback period

        Returns:
            R² series (0 to 1)
        """

        def calc_r2(window):
            if len(window) < period // 2:
                return np.nan
            x = np.arange(len(window))
            try:
                _, _, r_value, _, _ = stats.linregress(x, window)
                return r_value**2
            except (ValueError, RuntimeWarning):
                return np.nan

        return series.rolling(window=period).apply(calc_r2, raw=False)

    def _classify_trend_regime(
        self,
        slope: pd.Series,
        quality: pd.Series,
        slope_threshold: float = 0.02,  # 2% per bar
        quality_threshold: float = 0.3,
    ) -> pd.Series:
        """
        Classify trend regime based on slope and quality.

        Args:
            slope: Normalized slope series
            quality: R² series
            slope_threshold: Minimum absolute slope for trending
            quality_threshold: Minimum R² for valid trend

        Returns:
            Series with values: -1 (downtrend), 0 (sideways), +1 (uptrend)
        """
        regime = pd.Series(0, index=slope.index, dtype=int)

        # Uptrend: positive slope + decent quality
        uptrend_mask = (slope > slope_threshold) & (quality > quality_threshold)
        regime[uptrend_mask] = 1

        # Downtrend: negative slope + decent quality
        downtrend_mask = (slope < -slope_threshold) & (quality > quality_threshold)
        regime[downtrend_mask] = -1

        # Sideways: everything else (low slope or low quality)

        return regime

    def _compute_trend_strength(
        self,
        slope: pd.Series,
        atr: pd.Series,
    ) -> pd.Series:
        """
        Compute trend strength as slope normalized by ATR.

        Args:
            slope: Normalized slope (% per bar)
            atr: ATR series

        Returns:
            Trend strength series
        """
        # Convert ATR to percentage
        # Assume we have access to close for this calculation
        # Since we're in the compute() context, we can't easily get close here
        # So we'll just use the absolute slope as a proxy
        # This will be overridden in compute() if ATR is available

        # Avoid division by zero
        atr_safe = atr.replace(0, np.nan)

        return slope.abs() / atr_safe

    def _compute_trend_consistency(
        self,
        series: pd.Series,
        trend_slope: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute trend consistency: % of bars aligned with trend direction.

        Args:
            series: Price series
            trend_slope: Trend slope series (determines expected direction)
            period: Lookback period for consistency check

        Returns:
            Consistency score (0 to 1)
        """
        # Calculate bar-by-bar price changes
        bar_change = series.pct_change()

        # Determine expected direction from trend slope
        expected_direction = trend_slope.apply(
            lambda x: 1 if x > 0 else -1 if x < 0 else 0
        )

        # Check if bar change aligns with expected direction
        def calc_consistency(idx):
            if idx < period:
                return np.nan

            recent_changes = bar_change.iloc[idx - period + 1 : idx + 1]
            expected_dir = expected_direction.iloc[idx]

            if expected_dir == 0:
                return 0.5  # Neutral

            aligned_bars = sum(
                (recent_changes > 0) if expected_dir > 0 else (recent_changes < 0)
            )
            return aligned_bars / period

        return pd.Series(
            [calc_consistency(i) for i in range(len(series))], index=series.index
        )

    def _compute_multi_horizon_alignment(
        self,
        slope_short: pd.Series,
        slope_med: pd.Series,
        slope_long: pd.Series,
    ) -> pd.Series:
        """
        Compute alignment score across all three horizons.

        Score is higher when all horizons agree on trend direction.

        Args:
            slope_short: Short-term slope
            slope_med: Medium-term slope
            slope_long: Long-term slope

        Returns:
            Alignment score (0 to 1)
        """
        # Convert slopes to directions
        dir_short = slope_short.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        dir_med = slope_med.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        dir_long = slope_long.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

        # Calculate agreement
        def calc_alignment(s, m, l):
            if s == m == l and s != 0:
                return 1.0  # Perfect alignment
            elif (s == m and s != 0) or (m == l and m != 0) or (s == l and s != 0):
                return 0.67  # 2 out of 3 agree
            elif s == 0 or m == 0 or l == 0:
                return 0.33  # At least one is neutral
            else:
                return 0.0  # All disagree

        return pd.Series(
            [calc_alignment(s, m, l) for s, m, l in zip(dir_short, dir_med, dir_long)],
            index=slope_short.index,
        )

    def get_feature_names(self) -> List[str]:
        """Return list of QuantAgents trend feature names."""
        return [
            "qa_trend_slope_short",
            "qa_trend_slope_med",
            "qa_trend_slope_long",
            "qa_trend_quality_short",
            "qa_trend_quality_med",
            "qa_trend_quality_long",
            "qa_trend_regime",
            "qa_trend_strength_short",
            "qa_trend_strength_med",
            "qa_trend_strength_long",
            "qa_trend_consistency",
            "qa_trend_alignment_score",
        ]
