"""
Relative Rotation Graph (RRG) features.

Measures relative strength and momentum vs a benchmark for cross-sectional analysis.
"""

from typing import List, Literal, Optional
import pandas as pd
import numpy as np

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


# RRG Quadrant definitions
RRGQuadrant = Literal["LEADING", "WEAKENING", "LAGGING", "IMPROVING"]


class RRGFeatures(FeatureBase):
    """
    Relative Rotation Graph (RRG) indicators.

    Features:
    - RS Ratio (relative strength vs benchmark)
    - RS Momentum (rate of change of relative strength)
    - RRG Quadrant classification
    - Rotation direction and speed

    Note: Requires benchmark data to be provided separately.
    """

    # RRG normalization parameters
    RS_RATIO_PERIOD = 10
    RS_MOMENTUM_PERIOD = 10
    NORMALIZE_PERIOD = 52  # Standard RRG uses 52-week normalization

    def __init__(self, timeframe: Timeframe):
        """Initialize RRG features."""
        super().__init__(timeframe)
        # Adjust normalization period based on timeframe
        tf_multipliers = {
            Timeframe.W1: 1,  # Weekly: 52 weeks
            Timeframe.D1: 5,  # Daily: ~52 weeks in days / 5
            Timeframe.H4: 30,  # 4H: scale appropriately
            Timeframe.H1: 120,  # 1H: scale appropriately
        }
        self.normalize_period = self.NORMALIZE_PERIOD * tf_multipliers.get(timeframe, 1)

    def compute(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute RRG features.

        Args:
            df: OHLCV DataFrame for the stock
            benchmark_df: OHLCV DataFrame for the benchmark (e.g., SPY)
                          If None, RRG features will be NaN

        Returns:
            DataFrame with RRG features added
        """
        result = df.copy()

        if benchmark_df is None or benchmark_df.empty:
            # Return empty RRG features if no benchmark
            for col in self.get_feature_names():
                result[col] = np.nan
            return result

        # Align dataframes
        aligned_benchmark = benchmark_df.reindex(df.index, method="ffill")

        stock_close = result["close"]
        benchmark_close = aligned_benchmark["close"]

        # RS Ratio: Stock / Benchmark
        rs_ratio_raw = stock_close / benchmark_close

        # Normalize RS Ratio to oscillate around 100
        rs_ratio = self._normalize_rs(rs_ratio_raw, self.normalize_period)
        result["rs_ratio"] = rs_ratio

        # RS Momentum: Rate of change of RS Ratio
        rs_momentum_raw = rs_ratio - rs_ratio.shift(self.RS_MOMENTUM_PERIOD)
        rs_momentum = self._normalize_rs_momentum(
            rs_momentum_raw, self.normalize_period
        )
        result["rs_momentum"] = rs_momentum

        # RRG Quadrant classification
        result["rrg_quadrant"] = self._classify_quadrant(rs_ratio, rs_momentum)

        # One-hot encode quadrants for ML
        result["rrg_leading"] = (result["rrg_quadrant"] == "LEADING").astype(int)
        result["rrg_weakening"] = (result["rrg_quadrant"] == "WEAKENING").astype(int)
        result["rrg_lagging"] = (result["rrg_quadrant"] == "LAGGING").astype(int)
        result["rrg_improving"] = (result["rrg_quadrant"] == "IMPROVING").astype(int)

        # Rotation metrics
        result["rrg_distance"] = self._compute_distance(rs_ratio, rs_momentum)
        result["rrg_angle"] = self._compute_angle(rs_ratio, rs_momentum)
        result["rrg_rotation_speed"] = self._compute_rotation_speed(
            rs_ratio, rs_momentum
        )

        # RS Ratio trend
        result["rs_ratio_trend"] = np.where(
            rs_ratio > rs_ratio.shift(5),
            1,
            np.where(rs_ratio < rs_ratio.shift(5), -1, 0),
        )

        # RS Momentum trend
        result["rs_momentum_trend"] = np.where(
            rs_momentum > rs_momentum.shift(3),
            1,
            np.where(rs_momentum < rs_momentum.shift(3), -1, 0),
        )

        # Quadrant transition (is quadrant changing?)
        prev_quadrant = result["rrg_quadrant"].shift(1)
        result["rrg_quadrant_change"] = (
            result["rrg_quadrant"] != prev_quadrant
        ).astype(int)

        # Favorable quadrant for long trades
        result["rrg_long_favorable"] = (
            result["rrg_leading"] | result["rrg_improving"]
        ).astype(int)

        return result

    def _normalize_rs(
        self,
        rs_ratio: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Normalize RS Ratio to oscillate around 100.

        Uses rolling z-score normalization.
        """
        rolling_mean = rs_ratio.rolling(window=period, min_periods=period // 2).mean()
        rolling_std = rs_ratio.rolling(window=period, min_periods=period // 2).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        # Z-score, then scale to oscillate around 100
        zscore = (rs_ratio - rolling_mean) / rolling_std
        normalized = 100 + (zscore * 2)  # Scale factor of 2 for visibility

        return normalized

    def _normalize_rs_momentum(
        self,
        rs_momentum: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Normalize RS Momentum to oscillate around 100.
        """
        rolling_mean = rs_momentum.rolling(
            window=period, min_periods=period // 2
        ).mean()
        rolling_std = rs_momentum.rolling(window=period, min_periods=period // 2).std()

        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (rs_momentum - rolling_mean) / rolling_std
        normalized = 100 + (zscore * 2)

        return normalized

    def _classify_quadrant(
        self,
        rs_ratio: pd.Series,
        rs_momentum: pd.Series,
    ) -> pd.Series:
        """
        Classify into RRG quadrant based on RS Ratio and RS Momentum.

        Quadrants (centered at 100, 100):
        - LEADING: RS Ratio > 100, RS Momentum > 100
        - WEAKENING: RS Ratio > 100, RS Momentum < 100
        - LAGGING: RS Ratio < 100, RS Momentum < 100
        - IMPROVING: RS Ratio < 100, RS Momentum > 100
        """
        conditions = [
            (rs_ratio > 100) & (rs_momentum > 100),  # Leading
            (rs_ratio > 100) & (rs_momentum <= 100),  # Weakening
            (rs_ratio <= 100) & (rs_momentum <= 100),  # Lagging
            (rs_ratio <= 100) & (rs_momentum > 100),  # Improving
        ]

        choices = ["LEADING", "WEAKENING", "LAGGING", "IMPROVING"]

        return pd.Series(
            np.select(conditions, choices, default="LAGGING"),
            index=rs_ratio.index,
        )

    def _compute_distance(
        self,
        rs_ratio: pd.Series,
        rs_momentum: pd.Series,
    ) -> pd.Series:
        """
        Compute distance from RRG center (100, 100).

        Greater distance = stronger relative trend.
        """
        return np.sqrt((rs_ratio - 100) ** 2 + (rs_momentum - 100) ** 2)

    def _compute_angle(
        self,
        rs_ratio: pd.Series,
        rs_momentum: pd.Series,
    ) -> pd.Series:
        """
        Compute angle in RRG space (degrees from horizontal).

        0° = moving right (RS improving, momentum flat)
        90° = moving up (RS flat, momentum improving)
        etc.
        """
        return np.degrees(np.arctan2(rs_momentum - 100, rs_ratio - 100))

    def _compute_rotation_speed(
        self,
        rs_ratio: pd.Series,
        rs_momentum: pd.Series,
    ) -> pd.Series:
        """
        Compute rotation speed (angular velocity).

        Positive = clockwise rotation (typical RRG rotation)
        Negative = counter-clockwise
        """
        angle = self._compute_angle(rs_ratio, rs_momentum)
        return angle.diff(5)  # 5-period angular change

    def get_feature_names(self) -> List[str]:
        """Return list of RRG feature names."""
        return [
            "rs_ratio",
            "rs_momentum",
            "rrg_quadrant",
            "rrg_leading",
            "rrg_weakening",
            "rrg_lagging",
            "rrg_improving",
            "rrg_distance",
            "rrg_angle",
            "rrg_rotation_speed",
            "rs_ratio_trend",
            "rs_momentum_trend",
            "rrg_quadrant_change",
            "rrg_long_favorable",
        ]
