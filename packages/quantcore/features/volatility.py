"""
Volatility-related features.

Includes ATR, Bollinger Bands, and realized volatility.
"""

from typing import List
import pandas as pd
import numpy as np

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class VolatilityFeatures(FeatureBase):
    """
    Volatility technical indicators.

    Features:
    - ATR (Average True Range)
    - Bollinger Bands (width and position)
    - Realized volatility
    - Volatility regime indicators
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with volatility features added
        """
        result = df.copy()
        close = result["close"]
        high = result["high"]
        low = result["low"]

        # True Range and ATR
        result["true_range"] = self._compute_true_range(high, low, close)
        result["atr"] = self._compute_atr(result["true_range"], self.params.atr_period)

        # ATR as percentage of price
        result["atr_pct"] = (result["atr"] / close) * 100

        # ATR ratio (current vs rolling average)
        atr_ma = result["atr"].rolling(window=self.params.atr_period * 2).mean()
        result["atr_ratio"] = result["atr"] / atr_ma

        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = self._compute_bollinger_bands(
            close,
            period=self.params.bb_period,
            num_std=self.params.bb_std,
        )
        result["bb_middle"] = bb_middle
        result["bb_upper"] = bb_upper
        result["bb_lower"] = bb_lower

        # Bollinger Band width (normalized)
        result["bb_width"] = (bb_upper - bb_lower) / bb_middle * 100

        # Bollinger Band position (where price is within bands)
        bb_range = bb_upper - bb_lower
        bb_range = bb_range.replace(0, np.nan)
        result["bb_position"] = (close - bb_lower) / bb_range * 100

        # Price outside bands
        result["price_above_bb"] = (close > bb_upper).astype(int)
        result["price_below_bb"] = (close < bb_lower).astype(int)

        # Realized volatility (annualized)
        result["realized_vol"] = self._compute_realized_volatility(
            close,
            period=self.params.realized_vol_period,
        )

        # Volatility z-score (is current vol high or low vs recent)
        result["vol_zscore"] = self.zscore(
            result["realized_vol"],
            period=self.params.realized_vol_period * 2,
        )

        # Volatility regime (high/normal/low)
        result["vol_regime"] = self._classify_vol_regime(result["vol_zscore"])

        # Historical volatility ratio (short-term vs long-term)
        short_vol = self._compute_realized_volatility(close, period=10)
        long_vol = self._compute_realized_volatility(close, period=30)
        long_vol_safe = long_vol.replace(0, np.nan)
        result["hvol_ratio"] = short_vol / long_vol_safe

        # Intraday range (high-low as % of close)
        result["intraday_range"] = (high - low) / close * 100

        # Range expansion/contraction
        result["range_ma"] = (
            result["intraday_range"].rolling(window=self.params.atr_period).mean()
        )
        result["range_expansion"] = result["intraday_range"] / result["range_ma"]

        return result

    def _compute_true_range(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """
        Compute True Range.

        TR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def _compute_atr(
        self,
        true_range: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute Average True Range using exponential smoothing.
        """
        return true_range.ewm(span=period, adjust=False).mean()

    def _compute_bollinger_bands(
        self,
        close: pd.Series,
        period: int,
        num_std: float,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute Bollinger Bands.

        Middle = SMA
        Upper = Middle + num_std * StdDev
        Lower = Middle - num_std * StdDev
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        return middle, upper, lower

    def _compute_realized_volatility(
        self,
        close: pd.Series,
        period: int,
        annualization_factor: int = 252,
    ) -> pd.Series:
        """
        Compute realized (historical) volatility.

        Uses log returns and annualizes based on trading days.
        For hourly data, adjust annualization_factor accordingly.
        """
        log_returns = np.log(close / close.shift(1))

        # Rolling standard deviation of returns
        vol = log_returns.rolling(window=period).std()

        # Annualize (approximate - assumes 252 trading days, 6.5 hours per day)
        # For hourly data: sqrt(252 * 6.5) ≈ sqrt(1638) ≈ 40.5
        annualized = vol * np.sqrt(annualization_factor)

        return annualized * 100  # Express as percentage

    def _classify_vol_regime(self, vol_zscore: pd.Series) -> pd.Series:
        """
        Classify volatility regime.

        Returns:
            -1: Low volatility
             0: Normal volatility
             1: High volatility
        """
        return pd.cut(
            vol_zscore,
            bins=[-np.inf, -1, 1, np.inf],
            labels=[-1, 0, 1],
        ).astype(float)

    def get_feature_names(self) -> List[str]:
        """Return list of volatility feature names."""
        return [
            "true_range",
            "atr",
            "atr_pct",
            "atr_ratio",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "bb_position",
            "price_above_bb",
            "price_below_bb",
            "realized_vol",
            "vol_zscore",
            "vol_regime",
            "hvol_ratio",
            "intraday_range",
            "range_ma",
            "range_expansion",
        ]
