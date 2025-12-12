"""
Spread features for commodity trading.

Computes features from WTI-Brent spread, crack spreads, and other spread relationships.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class SpreadFeatures(FeatureBase):
    """
    Spread-based features for commodity trading.

    Features:
    - WTI-Brent spread and z-score
    - Crack spread (3-2-1) and z-score
    - Spread momentum and trend
    - Spread mean reversion signals
    - Spread volatility metrics
    """

    def __init__(
        self,
        timeframe: Timeframe,
        spread_lookback: int = 20,
        zscore_lookback: int = 60,
    ):
        """
        Initialize spread feature calculator.

        Args:
            timeframe: Timeframe for parameter adjustment
            spread_lookback: Lookback for spread statistics
            zscore_lookback: Lookback for z-score calculation
        """
        super().__init__(timeframe)

        # Adjust lookback based on timeframe
        if timeframe == Timeframe.H1:
            self.spread_lookback = spread_lookback
            self.zscore_lookback = zscore_lookback
        elif timeframe == Timeframe.H4:
            self.spread_lookback = max(10, spread_lookback // 2)
            self.zscore_lookback = max(30, zscore_lookback // 2)
        elif timeframe == Timeframe.D1:
            self.spread_lookback = max(5, spread_lookback // 4)
            self.zscore_lookback = max(20, zscore_lookback // 3)
        else:  # Weekly
            self.spread_lookback = max(4, spread_lookback // 5)
            self.zscore_lookback = max(13, zscore_lookback // 5)

    def compute(
        self,
        df: pd.DataFrame,
        spread_data: Optional[pd.DataFrame] = None,
        crack_spread_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute spread features.

        Args:
            df: Main OHLCV DataFrame (e.g., WTI)
            spread_data: Pre-computed WTI-Brent spread data
            crack_spread_data: Pre-computed crack spread data

        Returns:
            DataFrame with spread features added
        """
        result = df.copy()

        # If spread data provided, compute features from it
        if spread_data is not None:
            result = self._compute_wti_brent_features(result, spread_data)

        if crack_spread_data is not None:
            result = self._compute_crack_spread_features(result, crack_spread_data)

        # If no spread data, compute synthetic spread features from price alone
        if spread_data is None and crack_spread_data is None:
            result = self._compute_synthetic_spread_features(result)

        return result

    def _compute_wti_brent_features(
        self,
        df: pd.DataFrame,
        spread_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute WTI-Brent spread features."""
        result = df.copy()

        # Align spread data to main DataFrame
        common_idx = df.index.intersection(spread_data.index)

        if len(common_idx) == 0:
            logger.warning("No common indices between main data and spread data")
            return self._add_empty_spread_features(result, "wti_brent")

        # Core spread values
        spread = (
            spread_data.loc[common_idx, "spread"]
            if "spread" in spread_data.columns
            else pd.Series(index=common_idx)
        )

        # Map to result DataFrame
        result["wti_brent_spread"] = result.index.map(
            lambda x: spread.loc[x] if x in spread.index else np.nan
        )

        # Z-score (from spread data or computed)
        if "spread_zscore" in spread_data.columns:
            zscore = spread_data.loc[common_idx, "spread_zscore"]
            result["wti_brent_zscore"] = result.index.map(
                lambda x: zscore.loc[x] if x in zscore.index else np.nan
            )
        else:
            result["wti_brent_zscore"] = self._compute_zscore(
                result["wti_brent_spread"], self.zscore_lookback
            )

        # Spread momentum
        result["wti_brent_spread_roc_5"] = (
            result["wti_brent_spread"].pct_change(5) * 100
        )
        result["wti_brent_spread_roc_10"] = (
            result["wti_brent_spread"].pct_change(10) * 100
        )

        # Spread trend (regression slope)
        result["wti_brent_spread_slope"] = self._compute_regression_slope(
            result["wti_brent_spread"], self.spread_lookback
        )

        # Spread volatility
        result["wti_brent_spread_vol"] = (
            result["wti_brent_spread"].rolling(self.spread_lookback).std()
        )

        # Mean reversion signals
        result["wti_brent_oversold"] = (result["wti_brent_zscore"] < -2).astype(int)
        result["wti_brent_overbought"] = (result["wti_brent_zscore"] > 2).astype(int)

        # Spread percentile rank
        result["wti_brent_percentile"] = (
            result["wti_brent_spread"]
            .rolling(self.zscore_lookback)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )

        # Spread acceleration
        spread_change = result["wti_brent_spread"].diff()
        result["wti_brent_acceleration"] = spread_change.diff()

        return result

    def _compute_crack_spread_features(
        self,
        df: pd.DataFrame,
        crack_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute crack spread features."""
        result = df.copy()

        common_idx = df.index.intersection(crack_data.index)

        if len(common_idx) == 0:
            logger.warning("No common indices between main data and crack spread data")
            return self._add_empty_spread_features(result, "crack")

        # Core crack spread
        if "crack_spread" in crack_data.columns:
            crack = crack_data.loc[common_idx, "crack_spread"]
            result["crack_spread"] = result.index.map(
                lambda x: crack.loc[x] if x in crack.index else np.nan
            )

        # Z-score
        if "crack_zscore" in crack_data.columns:
            zscore = crack_data.loc[common_idx, "crack_zscore"]
            result["crack_zscore"] = result.index.map(
                lambda x: zscore.loc[x] if x in zscore.index else np.nan
            )
        else:
            result["crack_zscore"] = self._compute_zscore(
                result["crack_spread"], self.zscore_lookback
            )

        # Crack spread momentum
        result["crack_spread_roc_5"] = result["crack_spread"].pct_change(5) * 100

        # Crack spread trend
        result["crack_spread_slope"] = self._compute_regression_slope(
            result["crack_spread"], self.spread_lookback
        )

        # Crack spread volatility
        result["crack_spread_vol"] = (
            result["crack_spread"].rolling(self.spread_lookback).std()
        )

        # Refining margin signals
        result["crack_high_margin"] = (result["crack_zscore"] > 1.5).astype(int)
        result["crack_low_margin"] = (result["crack_zscore"] < -1.5).astype(int)

        return result

    def _compute_synthetic_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute synthetic spread-like features when no spread data available.

        Uses price momentum and mean reversion characteristics.
        """
        result = df.copy()

        # Price-based spread proxy (distance from moving averages)
        sma_20 = result["close"].rolling(20).mean()
        sma_50 = result["close"].rolling(50).mean()

        result["synthetic_spread"] = sma_20 - sma_50
        result["synthetic_spread_pct"] = (sma_20 - sma_50) / sma_50 * 100
        result["synthetic_spread_zscore"] = self._compute_zscore(
            result["synthetic_spread"], self.zscore_lookback
        )

        # Price momentum spread
        mom_5 = result["close"].pct_change(5)
        mom_20 = result["close"].pct_change(20)
        result["momentum_spread"] = mom_5 - mom_20

        return result

    def _compute_zscore(self, series: pd.Series, lookback: int) -> pd.Series:
        """Compute rolling z-score."""
        mean = series.rolling(lookback).mean()
        std = series.rolling(lookback).std()
        return (series - mean) / std.replace(0, np.nan)

    def _compute_regression_slope(self, series: pd.Series, lookback: int) -> pd.Series:
        """Compute rolling regression slope."""

        def slope(x):
            if len(x) < lookback // 2:
                return np.nan
            y = np.array(x)
            x_arr = np.arange(len(y))

            # Handle NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan

            coef = np.polyfit(x_arr[mask], y[mask], 1)
            return coef[0]

        return series.rolling(lookback).apply(slope, raw=True)

    def _add_empty_spread_features(
        self,
        df: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """Add empty spread feature columns."""
        result = df.copy()

        if prefix == "wti_brent":
            cols = [
                "wti_brent_spread",
                "wti_brent_zscore",
                "wti_brent_spread_roc_5",
                "wti_brent_spread_roc_10",
                "wti_brent_spread_slope",
                "wti_brent_spread_vol",
                "wti_brent_oversold",
                "wti_brent_overbought",
                "wti_brent_percentile",
                "wti_brent_acceleration",
            ]
        else:  # crack
            cols = [
                "crack_spread",
                "crack_zscore",
                "crack_spread_roc_5",
                "crack_spread_slope",
                "crack_spread_vol",
                "crack_high_margin",
                "crack_low_margin",
            ]

        for col in cols:
            result[col] = np.nan

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this class."""
        return [
            # WTI-Brent spread features
            "wti_brent_spread",
            "wti_brent_zscore",
            "wti_brent_spread_roc_5",
            "wti_brent_spread_roc_10",
            "wti_brent_spread_slope",
            "wti_brent_spread_vol",
            "wti_brent_oversold",
            "wti_brent_overbought",
            "wti_brent_percentile",
            "wti_brent_acceleration",
            # Crack spread features
            "crack_spread",
            "crack_zscore",
            "crack_spread_roc_5",
            "crack_spread_slope",
            "crack_spread_vol",
            "crack_high_margin",
            "crack_low_margin",
            # Synthetic spread features
            "synthetic_spread",
            "synthetic_spread_pct",
            "synthetic_spread_zscore",
            "momentum_spread",
        ]
