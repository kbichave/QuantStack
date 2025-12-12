"""
Trend and mean-related features.

Includes EMAs, regression slope, z-score, and price distance from mean.
"""

from typing import List
import pandas as pd
import numpy as np
from scipy import stats

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class TrendFeatures(FeatureBase):
    """
    Trend-related technical indicators.

    Features:
    - EMA (fast and slow)
    - Price distance from EMA
    - Z-score of price vs EMA
    - Regression slope
    - EMA alignment (bullish/bearish)
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trend features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with trend features added
        """
        result = df.copy()
        close = result["close"]

        # Exponential Moving Averages
        result["ema_fast"] = self.ema(close, self.params.ema_fast)
        result["ema_slow"] = self.ema(close, self.params.ema_slow)

        # Price distance from EMAs (percentage)
        result["price_dist_ema_fast"] = (
            (close - result["ema_fast"]) / result["ema_fast"] * 100
        )
        result["price_dist_ema_slow"] = (
            (close - result["ema_slow"]) / result["ema_slow"] * 100
        )

        # Z-score of price vs EMA
        result["zscore_price"] = self.zscore(close, self.params.zscore_period)

        # Z-score of price relative to EMA
        deviation_from_ema = close - result["ema_fast"]
        result["zscore_ema_deviation"] = self.zscore(
            deviation_from_ema,
            self.params.zscore_period,
        )

        # Regression slope (normalized)
        result["regression_slope"] = self._compute_regression_slope(
            close,
            period=self.params.ema_slow,
        )

        # EMA alignment: 1 if bullish (fast > slow), -1 if bearish
        result["ema_alignment"] = np.where(
            result["ema_fast"] > result["ema_slow"],
            1,
            np.where(result["ema_fast"] < result["ema_slow"], -1, 0),
        )

        # EMA slope (rate of change of EMA)
        result["ema_fast_slope"] = result["ema_fast"].pct_change(5) * 100
        result["ema_slow_slope"] = result["ema_slow"].pct_change(5) * 100

        # Distance between EMAs (convergence/divergence)
        result["ema_spread"] = (
            (result["ema_fast"] - result["ema_slow"]) / result["ema_slow"] * 100
        )

        # Price position: above/below both EMAs
        result["price_above_fast_ema"] = (close > result["ema_fast"]).astype(int)
        result["price_above_slow_ema"] = (close > result["ema_slow"]).astype(int)

        # Trend strength: distance from mean in ATR terms (if ATR available)
        if "atr" in result.columns:
            result["trend_strength"] = deviation_from_ema / result["atr"]

        return result

    def _compute_regression_slope(
        self,
        series: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute rolling linear regression slope.

        The slope is normalized by the mean price to make it comparable
        across different price levels.

        Args:
            series: Price series
            period: Lookback period

        Returns:
            Normalized slope series
        """

        def calc_slope(window):
            if len(window) < period // 2:
                return np.nan
            x = np.arange(len(window))
            try:
                slope, _, _, _, _ = stats.linregress(x, window)
                # Normalize by mean price
                return slope / window.mean() * 100
            except (ValueError, RuntimeWarning):
                return np.nan

        return series.rolling(window=period).apply(calc_slope, raw=False)

    def get_feature_names(self) -> List[str]:
        """Return list of trend feature names."""
        return [
            "ema_fast",
            "ema_slow",
            "price_dist_ema_fast",
            "price_dist_ema_slow",
            "zscore_price",
            "zscore_ema_deviation",
            "regression_slope",
            "ema_alignment",
            "ema_fast_slope",
            "ema_slow_slope",
            "ema_spread",
            "price_above_fast_ema",
            "price_above_slow_ema",
        ]
