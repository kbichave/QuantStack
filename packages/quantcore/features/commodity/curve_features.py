"""
Futures curve features for commodity trading.

Computes features from term structure, contango/backwardation, and roll yield.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class CurveFeatures(FeatureBase):
    """
    Futures curve features for commodity trading.

    Features:
    - Contango/backwardation indicator
    - Curve slope (1st vs 2nd month)
    - Roll yield estimate
    - Term structure shape
    - Curve steepness metrics
    """

    def __init__(
        self,
        timeframe: Timeframe,
        lookback: int = 20,
    ):
        """
        Initialize curve feature calculator.

        Args:
            timeframe: Timeframe for parameter adjustment
            lookback: Lookback for curve statistics
        """
        super().__init__(timeframe)

        # Adjust lookback based on timeframe
        if timeframe == Timeframe.H1:
            self.lookback = lookback
        elif timeframe == Timeframe.H4:
            self.lookback = max(10, lookback // 2)
        elif timeframe == Timeframe.D1:
            self.lookback = max(5, lookback // 4)
        else:  # Weekly
            self.lookback = max(4, lookback // 5)

    def compute(
        self,
        df: pd.DataFrame,
        curve_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Compute curve features.

        Args:
            df: Main OHLCV DataFrame
            curve_data: Dictionary with futures curve data (from adapter)

        Returns:
            DataFrame with curve features added
        """
        result = df.copy()

        if curve_data is not None and "curve_info" in curve_data:
            result = self._compute_from_curve_data(result, curve_data)
        else:
            # Estimate curve features from price behavior
            result = self._compute_estimated_curve_features(result)

        return result

    def _compute_from_curve_data(
        self,
        df: pd.DataFrame,
        curve_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute features from actual curve data."""
        result = df.copy()
        curve_info = curve_data.get("curve_info", pd.DataFrame())

        if curve_info.empty:
            return self._compute_estimated_curve_features(result)

        common_idx = df.index.intersection(curve_info.index)

        if len(common_idx) == 0:
            return self._compute_estimated_curve_features(result)

        # Contango/backwardation indicator
        if "is_contango" in curve_info.columns:
            is_contango = curve_info.loc[common_idx, "is_contango"]
            result["is_contango"] = result.index.map(
                lambda x: int(is_contango.loc[x]) if x in is_contango.index else np.nan
            )
            result["is_backwardation"] = 1 - result["is_contango"]

        # Curve slope
        if "estimated_curve_slope" in curve_info.columns:
            slope = curve_info.loc[common_idx, "estimated_curve_slope"]
            result["curve_slope"] = result.index.map(
                lambda x: slope.loc[x] if x in slope.index else np.nan
            )

            # Curve slope z-score
            result["curve_slope_zscore"] = self._compute_zscore(
                result["curve_slope"], self.lookback
            )

            # Curve slope momentum
            result["curve_slope_change"] = result["curve_slope"].diff(5)

        # Roll yield estimate (annualized)
        result["roll_yield"] = result["curve_slope"] * 12  # Monthly to annual
        result["roll_yield_zscore"] = self._compute_zscore(
            result["roll_yield"], self.lookback
        )

        # Carry trade signal
        result["carry_signal"] = np.where(
            result["curve_slope"] < -0.02,  # Backwardation > 2% annual
            1,  # Long carry
            np.where(
                result["curve_slope"] > 0.02,  # Contango > 2% annual
                -1,  # Short carry
                0,  # Neutral
            ),
        )

        # Curve regime
        result["curve_regime"] = np.where(
            result["curve_slope"] > 0.01,
            "CONTANGO",
            np.where(result["curve_slope"] < -0.01, "BACKWARDATION", "FLAT"),
        )

        # Curve regime duration
        result["curve_regime_duration"] = self._compute_regime_duration(
            result["is_contango"]
        )

        return result

    def _compute_estimated_curve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate curve features from price behavior when no curve data available.

        Uses momentum and mean reversion characteristics to estimate curve shape.
        """
        result = df.copy()

        # Estimate curve slope from roll return pattern
        # In contango, spot rises relative to futures (roll cost)
        # In backwardation, spot falls relative to futures (roll benefit)

        # Use momentum differential as curve proxy
        mom_short = result["close"].pct_change(5)
        mom_medium = result["close"].pct_change(20)

        # Curve slope estimate: negative momentum differential suggests contango
        result["curve_slope_est"] = (
            -(mom_medium - mom_short).rolling(self.lookback).mean()
        )
        result["curve_slope_est_zscore"] = self._compute_zscore(
            result["curve_slope_est"], self.lookback * 2
        )

        # Contango/backwardation estimate
        result["is_contango_est"] = (result["curve_slope_est"] > 0).astype(int)
        result["is_backwardation_est"] = (result["curve_slope_est"] < 0).astype(int)

        # Roll yield estimate
        result["roll_yield_est"] = result["curve_slope_est"] * 12

        # Volatility term structure proxy
        vol_short = result["close"].pct_change().rolling(5).std() * np.sqrt(252)
        vol_long = result["close"].pct_change().rolling(20).std() * np.sqrt(252)
        result["vol_term_structure"] = vol_short / vol_long.replace(0, np.nan)
        result["vol_term_structure_zscore"] = self._compute_zscore(
            result["vol_term_structure"], self.lookback
        )

        # Carry trade signal
        result["carry_signal"] = np.where(
            result["curve_slope_est"] < -0.005,
            1,  # Long carry
            np.where(result["curve_slope_est"] > 0.005, -1, 0),
        )

        # Curve steepness
        result["curve_steepness"] = result["curve_slope_est"].abs()
        result["curve_steepness_percentile"] = (
            result["curve_steepness"]
            .rolling(self.lookback * 2)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )

        # Copy estimated values to main columns for consistency
        result["is_contango"] = result["is_contango_est"]
        result["is_backwardation"] = result["is_backwardation_est"]
        result["curve_slope"] = result["curve_slope_est"]
        result["roll_yield"] = result["roll_yield_est"]

        return result

    def _compute_zscore(self, series: pd.Series, lookback: int) -> pd.Series:
        """Compute rolling z-score."""
        mean = series.rolling(lookback).mean()
        std = series.rolling(lookback).std()
        return (series - mean) / std.replace(0, np.nan)

    def _compute_regime_duration(self, regime_series: pd.Series) -> pd.Series:
        """Compute consecutive duration in current regime."""
        # Create groups where regime changes
        regime_change = regime_series != regime_series.shift(1)
        regime_groups = regime_change.cumsum()

        # Count duration within each group
        duration = regime_series.groupby(regime_groups).cumcount() + 1

        return duration

    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this class."""
        return [
            # Core curve features
            "is_contango",
            "is_backwardation",
            "curve_slope",
            "curve_slope_zscore",
            "curve_slope_change",
            "roll_yield",
            "roll_yield_zscore",
            "carry_signal",
            "curve_regime",
            "curve_regime_duration",
            # Estimated curve features
            "curve_slope_est",
            "curve_slope_est_zscore",
            "is_contango_est",
            "is_backwardation_est",
            "roll_yield_est",
            "vol_term_structure",
            "vol_term_structure_zscore",
            "curve_steepness",
            "curve_steepness_percentile",
        ]
