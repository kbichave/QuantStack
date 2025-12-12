"""
Volume-related features.

Includes volume ratio, OBV, and volume-price analysis.
"""

from typing import List
import pandas as pd
import numpy as np

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class VolumeFeatures(FeatureBase):
    """
    Volume technical indicators.

    Features:
    - Volume ratio (current vs average)
    - OBV (On-Balance Volume)
    - Volume z-score
    - Volume-price relationship indicators
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with volume features added
        """
        result = df.copy()
        close = result["close"]
        volume = result["volume"]
        high = result["high"]
        low = result["low"]

        # Volume moving average
        result["volume_ma"] = self.sma(volume, self.params.volume_ma_period)

        # Volume ratio (current vs average)
        vol_ma_safe = result["volume_ma"].replace(0, np.nan)
        result["volume_ratio"] = volume / vol_ma_safe

        # Volume z-score
        result["volume_zscore"] = self.zscore(volume, self.params.volume_ma_period)

        # High volume flag
        result["high_volume"] = (result["volume_ratio"] > 1.5).astype(int)
        result["low_volume"] = (result["volume_ratio"] < 0.5).astype(int)

        # On-Balance Volume (OBV)
        result["obv"] = self._compute_obv(close, volume)

        # OBV change (momentum of OBV)
        result["obv_change"] = result["obv"].diff(self.params.obv_period)

        # OBV z-score
        result["obv_zscore"] = self.zscore(result["obv"], self.params.obv_period * 2)

        # Volume-weighted average price (VWAP) - rolling
        result["vwap"] = self._compute_rolling_vwap(
            high,
            low,
            close,
            volume,
            period=self.params.volume_ma_period,
        )

        # Price distance from VWAP
        result["price_dist_vwap"] = (close - result["vwap"]) / result["vwap"] * 100

        # Money Flow Index (MFI)
        result["mfi"] = self._compute_mfi(
            high,
            low,
            close,
            volume,
            period=self.params.rsi_period,
        )

        # Accumulation/Distribution Line
        result["ad_line"] = self._compute_ad_line(high, low, close, volume)
        result["ad_change"] = result["ad_line"].diff(self.params.volume_ma_period)

        # Chaikin Money Flow
        result["cmf"] = self._compute_cmf(
            high,
            low,
            close,
            volume,
            period=self.params.volume_ma_period,
        )

        # Force Index
        result["force_index"] = self._compute_force_index(close, volume)
        result["force_index_ma"] = self.ema(result["force_index"], 13)

        # Volume trend (increasing or decreasing)
        vol_short_ma = self.sma(volume, 5)
        vol_long_ma = self.sma(volume, self.params.volume_ma_period)
        result["volume_trend"] = np.where(
            vol_short_ma > vol_long_ma, 1, np.where(vol_short_ma < vol_long_ma, -1, 0)
        )

        return result

    def _compute_obv(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Compute On-Balance Volume (OBV).

        OBV adds volume on up days, subtracts on down days.
        """
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        return obv

    def _compute_rolling_vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute rolling Volume-Weighted Average Price.

        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
        """
        typical_price = (high + low + close) / 3

        tp_volume = typical_price * volume

        rolling_tp_vol = tp_volume.rolling(window=period).sum()
        rolling_vol = volume.rolling(window=period).sum()

        rolling_vol_safe = rolling_vol.replace(0, np.nan)

        return rolling_tp_vol / rolling_vol_safe

    def _compute_mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute Money Flow Index (MFI).

        MFI is a volume-weighted RSI.
        """
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        # Positive and negative money flow
        tp_change = typical_price.diff()
        positive_flow = raw_money_flow.where(tp_change > 0, 0)
        negative_flow = raw_money_flow.where(tp_change < 0, 0)

        # Rolling sums
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()

        # Money flow ratio
        negative_sum_safe = negative_sum.replace(0, np.nan)
        mf_ratio = positive_sum / negative_sum_safe

        # MFI
        mfi = 100 - (100 / (1 + mf_ratio))

        return mfi

    def _compute_ad_line(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Compute Accumulation/Distribution Line.

        AD = Previous AD + Money Flow Volume
        Money Flow Volume = Money Flow Multiplier * Volume
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        """
        hl_range = high - low
        hl_range_safe = hl_range.replace(0, np.nan)

        mf_multiplier = ((close - low) - (high - close)) / hl_range_safe
        mf_volume = mf_multiplier * volume

        return mf_volume.cumsum()

    def _compute_cmf(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute Chaikin Money Flow.

        CMF = Sum(Money Flow Volume) / Sum(Volume) over period
        """
        hl_range = high - low
        hl_range_safe = hl_range.replace(0, np.nan)

        mf_multiplier = ((close - low) - (high - close)) / hl_range_safe
        mf_volume = mf_multiplier * volume

        rolling_mf_vol = mf_volume.rolling(window=period).sum()
        rolling_vol = volume.rolling(window=period).sum()
        rolling_vol_safe = rolling_vol.replace(0, np.nan)

        return rolling_mf_vol / rolling_vol_safe

    def _compute_force_index(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Compute Force Index.

        Force Index = Price Change * Volume
        """
        return close.diff() * volume

    def get_feature_names(self) -> List[str]:
        """Return list of volume feature names."""
        return [
            "volume_ma",
            "volume_ratio",
            "volume_zscore",
            "high_volume",
            "low_volume",
            "obv",
            "obv_change",
            "obv_zscore",
            "vwap",
            "price_dist_vwap",
            "mfi",
            "ad_line",
            "ad_change",
            "cmf",
            "force_index",
            "force_index_ma",
            "volume_trend",
        ]
