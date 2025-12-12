"""
Microstructure features for commodity trading.

Computes volume imbalance, VWAP deviation, realized volatility, and gap patterns.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class MicrostructureFeatures(FeatureBase):
    """
    Microstructure features for commodity trading.

    Features:
    - Volume imbalance (buy vs sell pressure)
    - VWAP and deviation from VWAP
    - Realized volatility metrics
    - Overnight gap patterns
    - Intraday range patterns
    - Price/volume divergence
    """

    def __init__(
        self,
        timeframe: Timeframe,
        vol_lookback: int = 20,
        vwap_lookback: int = 10,
    ):
        """
        Initialize microstructure feature calculator.

        Args:
            timeframe: Timeframe for parameter adjustment
            vol_lookback: Lookback for volume statistics
            vwap_lookback: Lookback for VWAP calculations
        """
        super().__init__(timeframe)

        # Adjust lookback based on timeframe
        if timeframe == Timeframe.H1:
            self.vol_lookback = vol_lookback
            self.vwap_lookback = vwap_lookback
        elif timeframe == Timeframe.H4:
            self.vol_lookback = max(10, vol_lookback // 2)
            self.vwap_lookback = max(5, vwap_lookback // 2)
        elif timeframe == Timeframe.D1:
            self.vol_lookback = max(5, vol_lookback // 4)
            self.vwap_lookback = max(3, vwap_lookback // 3)
        else:
            self.vol_lookback = max(4, vol_lookback // 5)
            self.vwap_lookback = max(3, vwap_lookback // 4)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute microstructure features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with microstructure features added
        """
        result = df.copy()

        # Volume imbalance features
        result = self._compute_volume_imbalance(result)

        # VWAP features
        result = self._compute_vwap_features(result)

        # Realized volatility features
        result = self._compute_realized_vol_features(result)

        # Gap features
        result = self._compute_gap_features(result)

        # Range features
        result = self._compute_range_features(result)

        # Price/volume divergence
        result = self._compute_divergence_features(result)

        return result

    def _compute_volume_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume imbalance features."""
        result = df.copy()

        # Classify bars as up/down
        result["is_up_bar"] = (result["close"] > result["open"]).astype(int)
        result["is_down_bar"] = (result["close"] < result["open"]).astype(int)

        # Up/down volume
        result["up_volume"] = result["volume"] * result["is_up_bar"]
        result["down_volume"] = result["volume"] * result["is_down_bar"]

        # Rolling up/down volume
        result["up_volume_sum"] = result["up_volume"].rolling(self.vol_lookback).sum()
        result["down_volume_sum"] = (
            result["down_volume"].rolling(self.vol_lookback).sum()
        )

        # Volume imbalance ratio
        total_vol = result["up_volume_sum"] + result["down_volume_sum"]
        result["volume_imbalance"] = (
            result["up_volume_sum"] - result["down_volume_sum"]
        ) / total_vol.replace(0, np.nan)

        # Volume imbalance z-score
        result["volume_imbalance_zscore"] = self._compute_zscore(
            result["volume_imbalance"], self.vol_lookback * 2
        )

        # Money flow (price-weighted volume)
        typical_price = (result["high"] + result["low"] + result["close"]) / 3
        result["money_flow"] = typical_price * result["volume"]

        # Positive/negative money flow
        price_change = typical_price.diff()
        result["positive_mf"] = result["money_flow"].where(price_change > 0, 0)
        result["negative_mf"] = result["money_flow"].where(price_change < 0, 0)

        # Money flow ratio
        pos_mf_sum = result["positive_mf"].rolling(self.vol_lookback).sum()
        neg_mf_sum = result["negative_mf"].rolling(self.vol_lookback).sum()
        result["money_flow_ratio"] = pos_mf_sum / neg_mf_sum.replace(0, np.nan)

        # Money flow index (MFI-like)
        result["mfi"] = 100 - (100 / (1 + result["money_flow_ratio"]))

        # Volume momentum
        result["volume_roc"] = result["volume"].pct_change(5) * 100

        # Relative volume (vs average)
        vol_avg = result["volume"].rolling(self.vol_lookback).mean()
        result["relative_volume"] = result["volume"] / vol_avg.replace(0, np.nan)
        result["relative_volume_zscore"] = self._compute_zscore(
            result["relative_volume"], self.vol_lookback * 2
        )

        return result

    def _compute_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute VWAP features."""
        result = df.copy()

        # Typical price
        typical_price = (result["high"] + result["low"] + result["close"]) / 3

        # Rolling VWAP
        pv = typical_price * result["volume"]
        result["vwap"] = pv.rolling(self.vwap_lookback).sum() / result[
            "volume"
        ].rolling(self.vwap_lookback).sum().replace(0, np.nan)

        # Price deviation from VWAP
        result["vwap_deviation"] = (
            (result["close"] - result["vwap"]) / result["vwap"] * 100
        )
        result["vwap_deviation_zscore"] = self._compute_zscore(
            result["vwap_deviation"], self.vol_lookback
        )

        # VWAP bands (standard deviation)
        vwap_std = (result["close"] - result["vwap"]).rolling(self.vwap_lookback).std()
        result["vwap_upper_band"] = result["vwap"] + 2 * vwap_std
        result["vwap_lower_band"] = result["vwap"] - 2 * vwap_std

        # Position relative to VWAP bands
        result["vwap_band_position"] = (result["close"] - result["vwap_lower_band"]) / (
            result["vwap_upper_band"] - result["vwap_lower_band"]
        ).replace(0, np.nan)

        # Above/below VWAP
        result["above_vwap"] = (result["close"] > result["vwap"]).astype(int)
        result["below_vwap"] = (result["close"] < result["vwap"]).astype(int)

        # Consecutive bars above/below VWAP
        result["bars_above_vwap"] = (
            result["above_vwap"]
            .groupby((result["above_vwap"] != result["above_vwap"].shift()).cumsum())
            .cumsum()
            * result["above_vwap"]
        )

        result["bars_below_vwap"] = (
            result["below_vwap"]
            .groupby((result["below_vwap"] != result["below_vwap"].shift()).cumsum())
            .cumsum()
            * result["below_vwap"]
        )

        return result

    def _compute_realized_vol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute realized volatility features."""
        result = df.copy()

        # Returns
        returns = result["close"].pct_change()
        log_returns = np.log(result["close"] / result["close"].shift(1))

        # Realized volatility (annualized)
        result["realized_vol_5"] = returns.rolling(5).std() * np.sqrt(252)
        result["realized_vol_10"] = returns.rolling(10).std() * np.sqrt(252)
        result["realized_vol_20"] = returns.rolling(20).std() * np.sqrt(252)

        # Volatility term structure (short vs long)
        result["vol_term_ratio"] = result["realized_vol_5"] / result[
            "realized_vol_20"
        ].replace(0, np.nan)

        # Volatility z-score
        result["vol_zscore"] = self._compute_zscore(
            result["realized_vol_10"], self.vol_lookback * 2
        )

        # Volatility shock (sudden vol increase)
        vol_change = result["realized_vol_10"].pct_change(5)
        result["vol_shock"] = (vol_change > 0.5).astype(int)  # 50% increase

        # Parkinson volatility (high-low based)
        result["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2)))
            * np.log(result["high"] / result["low"]).rolling(self.vol_lookback).mean()
            ** 2
        ) * np.sqrt(252)

        # Garman-Klass volatility (OHLC based)
        hl_term = 0.5 * np.log(result["high"] / result["low"]) ** 2
        co_term = (2 * np.log(2) - 1) * np.log(result["close"] / result["open"]) ** 2
        result["gk_vol"] = np.sqrt(
            (hl_term - co_term).rolling(self.vol_lookback).mean()
        ) * np.sqrt(252)

        # Volatility regime
        vol_percentile = (
            result["realized_vol_20"]
            .rolling(60)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )
        result["vol_regime"] = np.where(
            vol_percentile > 0.8,
            2,  # High vol
            np.where(vol_percentile < 0.2, 0, 1),  # Low vol / Normal
        )

        return result

    def _compute_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute gap features."""
        result = df.copy()

        # Gap (open vs previous close)
        result["gap"] = result["open"] - result["close"].shift(1)
        result["gap_pct"] = result["gap"] / result["close"].shift(1) * 100

        # Gap z-score
        result["gap_zscore"] = self._compute_zscore(
            result["gap_pct"], self.vol_lookback
        )

        # Gap up/down
        result["gap_up"] = (result["gap_pct"] > 0.5).astype(int)  # > 0.5% gap up
        result["gap_down"] = (result["gap_pct"] < -0.5).astype(int)  # > 0.5% gap down

        # Large gap flag
        gap_std = result["gap_pct"].rolling(self.vol_lookback).std()
        result["large_gap"] = (result["gap_pct"].abs() > 2 * gap_std).astype(int)

        # Gap fill (price returns to previous close)
        result["gap_filled"] = np.where(
            result["gap_pct"] > 0,
            result["low"]
            <= result["close"].shift(1),  # Gap up filled if low touches prev close
            result["high"]
            >= result["close"].shift(1),  # Gap down filled if high touches prev close
        ).astype(int)

        # Overnight return (for daily data or first bar of day)
        result["overnight_return"] = result["gap_pct"]

        # Gap to range ratio
        result["gap_to_range"] = result["gap"].abs() / (
            result["high"] - result["low"]
        ).replace(0, np.nan)

        return result

    def _compute_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute range-based features."""
        result = df.copy()

        # True range
        high_low = result["high"] - result["low"]
        high_close = (result["high"] - result["close"].shift(1)).abs()
        low_close = (result["low"] - result["close"].shift(1)).abs()
        result["true_range"] = pd.concat([high_low, high_close, low_close], axis=1).max(
            axis=1
        )

        # ATR
        result["atr"] = result["true_range"].rolling(14).mean()

        # Range as % of price
        result["range_pct"] = (result["high"] - result["low"]) / result["close"] * 100

        # Range z-score
        result["range_zscore"] = self._compute_zscore(
            result["range_pct"], self.vol_lookback
        )

        # Intraday momentum (close position in range)
        result["close_position"] = (result["close"] - result["low"]) / (
            result["high"] - result["low"]
        ).replace(0, np.nan)

        # Wide range bar
        avg_range = result["range_pct"].rolling(self.vol_lookback).mean()
        result["wide_range_bar"] = (result["range_pct"] > 1.5 * avg_range).astype(int)

        # Narrow range bar
        result["narrow_range_bar"] = (result["range_pct"] < 0.5 * avg_range).astype(int)

        # Range expansion/contraction
        result["range_change"] = result["range_pct"].pct_change()
        result["range_expanding"] = (result["range_change"] > 0.2).astype(int)
        result["range_contracting"] = (result["range_change"] < -0.2).astype(int)

        return result

    def _compute_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price/volume divergence features."""
        result = df.copy()

        # Price momentum
        price_mom = result["close"].pct_change(10)

        # Volume momentum
        vol_mom = result["volume"].pct_change(10)

        # Price up, volume down = bearish divergence
        result["bearish_divergence"] = ((price_mom > 0) & (vol_mom < -0.2)).astype(int)

        # Price down, volume down = bullish divergence
        result["bullish_divergence"] = ((price_mom < 0) & (vol_mom < -0.2)).astype(int)

        # Price up, volume up = confirmation
        result["bullish_confirmation"] = ((price_mom > 0) & (vol_mom > 0.2)).astype(int)

        # Price down, volume up = selling pressure
        result["selling_pressure"] = ((price_mom < 0) & (vol_mom > 0.2)).astype(int)

        # Divergence score (-1 to +1)
        result["divergence_score"] = np.where(
            result["bullish_divergence"] == 1,
            1,
            np.where(
                result["bearish_divergence"] == 1,
                -1,
                np.where(
                    result["bullish_confirmation"] == 1,
                    0.5,
                    np.where(result["selling_pressure"] == 1, -0.5, 0),
                ),
            ),
        )

        return result

    def _compute_zscore(self, series: pd.Series, lookback: int) -> pd.Series:
        """Compute rolling z-score."""
        mean = series.rolling(lookback).mean()
        std = series.rolling(lookback).std()
        return (series - mean) / std.replace(0, np.nan)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this class."""
        return [
            # Volume imbalance
            "is_up_bar",
            "is_down_bar",
            "up_volume",
            "down_volume",
            "up_volume_sum",
            "down_volume_sum",
            "volume_imbalance",
            "volume_imbalance_zscore",
            "money_flow",
            "positive_mf",
            "negative_mf",
            "money_flow_ratio",
            "mfi",
            "volume_roc",
            "relative_volume",
            "relative_volume_zscore",
            # VWAP
            "vwap",
            "vwap_deviation",
            "vwap_deviation_zscore",
            "vwap_upper_band",
            "vwap_lower_band",
            "vwap_band_position",
            "above_vwap",
            "below_vwap",
            "bars_above_vwap",
            "bars_below_vwap",
            # Realized volatility
            "realized_vol_5",
            "realized_vol_10",
            "realized_vol_20",
            "vol_term_ratio",
            "vol_zscore",
            "vol_shock",
            "parkinson_vol",
            "gk_vol",
            "vol_regime",
            # Gaps
            "gap",
            "gap_pct",
            "gap_zscore",
            "gap_up",
            "gap_down",
            "large_gap",
            "gap_filled",
            "overnight_return",
            "gap_to_range",
            # Range
            "true_range",
            "atr",
            "range_pct",
            "range_zscore",
            "close_position",
            "wide_range_bar",
            "narrow_range_bar",
            "range_change",
            "range_expanding",
            "range_contracting",
            # Divergence
            "bearish_divergence",
            "bullish_divergence",
            "bullish_confirmation",
            "selling_pressure",
            "divergence_score",
        ]
