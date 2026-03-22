"""
Trend and mean-related features.

Includes EMAs, regression slope, z-score, price distance from mean,
Supertrend, Ichimoku Cloud, and Hull Moving Average.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from quantstack.core.features.base import FeatureBase


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

    def get_feature_names(self) -> list[str]:
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


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------


class SupertrendIndicator:
    """
    ATR-based trailing stop band that locks in trend direction.

    In an uptrend the indicator sits below price (lower band = support).
    In a downtrend it sits above price (upper band = resistance).
    The band never moves against the trend — memory prevents it from widening.

    Parameters
    ----------
    atr_length : int
        Period for ATR calculation (Wilder's smoothing via EMA).
    multiplier : float
        Number of ATR lengths added/subtracted from the midpoint.
    """

    def __init__(self, atr_length: int = 10, multiplier: float = 3.0) -> None:
        self.atr_length = atr_length
        self.multiplier = multiplier

    def compute(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        """
        Compute Supertrend.

        Parameters
        ----------
        high, low, close : pd.Series
            OHLCV price series with a shared DatetimeIndex.

        Returns
        -------
        pd.DataFrame with columns:
            supertrend   – the band value (float)
            st_direction – 1 = uptrend, -1 = downtrend
            st_uptrend   – bool True when in uptrend
        """
        hl2 = (high + low) / 2.0

        # True range → ATR (Wilder's = EMA with span = atr_length)
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=self.atr_length, adjust=False).mean()

        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr

        n = len(close)
        upper = basic_upper.copy()
        lower = basic_lower.copy()
        supertrend = pd.Series(np.nan, index=close.index)
        direction = pd.Series(0, index=close.index, dtype=int)

        for i in range(1, n):
            # Upper band: only tighten, never widen, unless price breaks above
            if (
                basic_upper.iloc[i] < upper.iloc[i - 1]
                or close.iloc[i - 1] > upper.iloc[i - 1]
            ):
                upper.iloc[i] = basic_upper.iloc[i]
            else:
                upper.iloc[i] = upper.iloc[i - 1]

            # Lower band: only raise, never drop, unless price breaks below
            if (
                basic_lower.iloc[i] > lower.iloc[i - 1]
                or close.iloc[i - 1] < lower.iloc[i - 1]
            ):
                lower.iloc[i] = basic_lower.iloc[i]
            else:
                lower.iloc[i] = lower.iloc[i - 1]

            # Direction: compare close to the *previous* supertrend value
            prev_st = supertrend.iloc[i - 1]
            if np.isnan(prev_st):
                # Initialise on first valid bar
                direction.iloc[i] = 1 if close.iloc[i] > upper.iloc[i] else -1
            elif prev_st == upper.iloc[i - 1]:
                # Was in downtrend
                direction.iloc[i] = 1 if close.iloc[i] > upper.iloc[i] else -1
            else:
                # Was in uptrend
                direction.iloc[i] = -1 if close.iloc[i] < lower.iloc[i] else 1

            supertrend.iloc[i] = (
                lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
            )

        return pd.DataFrame(
            {
                "supertrend": supertrend,
                "st_direction": direction,
                "st_uptrend": direction == 1,
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Ichimoku Cloud
# ---------------------------------------------------------------------------


class IchimokuCloud:
    """
    Five-component Japanese trend system.

    Components
    ----------
    tenkan_sen   : (9H + 9L) / 2        – conversion line (fast)
    kijun_sen    : (26H + 26L) / 2      – base line (medium)
    senkou_a     : (tenkan + kijun) / 2 shifted +displacement forward
    senkou_b     : (52H + 52L) / 2      shifted +displacement forward
    chikou_span  : close                 shifted -displacement backward

    The cloud (kumo) is the region between senkou_a and senkou_b.

    Notes
    -----
    *No lookahead*: senkou_a/b columns are shifted so the cloud appears
    at future index positions — they use only data available at computation
    time.  The forward shift is purely presentational; at bar i only bars
    ≤ i are used to compute the values.
    """

    def __init__(
        self,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ) -> None:
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

    @staticmethod
    def _midpoint(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        return (high.rolling(period).max() + low.rolling(period).min()) / 2.0

    def compute(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        """
        Compute all five Ichimoku components.

        Returns
        -------
        pd.DataFrame with columns:
            tenkan_sen, kijun_sen,
            senkou_a (shifted forward by displacement),
            senkou_b (shifted forward by displacement),
            chikou_span (close shifted backward by displacement),
            price_above_cloud, price_below_cloud,
            cloud_bullish (senkou_a > senkou_b at current bar)
        """
        tenkan_sen = self._midpoint(high, low, self.tenkan)
        kijun_sen = self._midpoint(high, low, self.kijun)
        senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(self.displacement)
        senkou_b = self._midpoint(high, low, self.senkou_b_period).shift(
            self.displacement
        )
        chikou_span = close.shift(-self.displacement)

        cloud_top = senkou_a.combine(senkou_b, max)
        cloud_bot = senkou_a.combine(senkou_b, min)

        return pd.DataFrame(
            {
                "tenkan_sen": tenkan_sen,
                "kijun_sen": kijun_sen,
                "senkou_a": senkou_a,
                "senkou_b": senkou_b,
                "chikou_span": chikou_span,
                "price_above_cloud": (close > cloud_top).astype(int),
                "price_below_cloud": (close < cloud_bot).astype(int),
                "cloud_bullish": (senkou_a > senkou_b).astype(int),
                "tenkan_above_kijun": (tenkan_sen > kijun_sen).astype(int),
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Hull Moving Average
# ---------------------------------------------------------------------------


class HullMovingAverage:
    """
    Hull Moving Average (HMA) — reduces lag by ~50% vs EMA of same period.

    Algorithm
    ---------
    1.  wma_half = WMA(close, period // 2)
    2.  wma_full = WMA(close, period)
    3.  raw      = 2 * wma_half - wma_full
    4.  hma      = WMA(raw, round(sqrt(period)))

    The double-weighting in step 3 removes most of the lag, and the final
    short WMA in step 4 smooths the result.
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period

    @staticmethod
    def _wma(series: pd.Series, period: int) -> pd.Series:
        """Linearly-weighted moving average."""
        weights = np.arange(1, period + 1, dtype=float)
        return series.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    def compute(self, close: pd.Series) -> pd.DataFrame:
        """
        Compute HMA and derived signals.

        Returns
        -------
        pd.DataFrame with columns:
            hma         – Hull Moving Average value
            hma_slope   – bar-over-bar change in HMA (positive = rising)
            hma_uptrend – True when hma_slope > 0
        """
        sqrt_period = max(2, round(math.sqrt(self.period)))
        half_period = max(2, self.period // 2)

        wma_half = self._wma(close, half_period)
        wma_full = self._wma(close, self.period)
        raw = 2.0 * wma_half - wma_full
        hma = self._wma(raw, sqrt_period)
        slope = hma - hma.shift(1)

        return pd.DataFrame(
            {
                "hma": hma,
                "hma_slope": slope,
                "hma_uptrend": (slope > 0).astype(int),
            },
            index=close.index,
        )
