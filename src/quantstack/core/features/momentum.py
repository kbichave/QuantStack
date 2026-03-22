"""
Momentum-related features.

Includes RSI, Stochastics, MACD, and Rate of Change.
"""

import numpy as np
import pandas as pd

from quantstack.core.features.base import FeatureBase


class MomentumFeatures(FeatureBase):
    """
    Momentum technical indicators.

    Features:
    - RSI (Relative Strength Index)
    - Stochastic Oscillator (%K and %D)
    - MACD (Moving Average Convergence Divergence)
    - ROC (Rate of Change)
    - Momentum divergence indicators
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with momentum features added
        """
        result = df.copy()
        close = result["close"]
        high = result["high"]
        low = result["low"]

        # RSI (only compute if not already present from TechnicalIndicators)
        if "rsi" not in result.columns:
            result["rsi"] = self._compute_rsi(close, self.params.rsi_period)

        # RSI zones
        result["rsi_oversold"] = (result["rsi"] < 30).astype(int)
        result["rsi_overbought"] = (result["rsi"] > 70).astype(int)

        # RSI divergence from extremes
        result["rsi_from_oversold"] = np.maximum(0, result["rsi"] - 30)
        result["rsi_from_overbought"] = np.maximum(0, 70 - result["rsi"])

        # Stochastic Oscillator (only compute if not already present)
        if "stoch_k" not in result.columns or "stoch_d" not in result.columns:
            stoch_k, stoch_d = self._compute_stochastic(
                high,
                low,
                close,
                k_period=self.params.stoch_k_period,
                d_period=self.params.stoch_d_period,
            )
            result["stoch_k"] = stoch_k
            result["stoch_d"] = stoch_d
        else:
            stoch_k = result["stoch_k"]
            stoch_d = result["stoch_d"]

        result["stoch_cross"] = np.where(
            stoch_k > stoch_d, 1, np.where(stoch_k < stoch_d, -1, 0)
        )

        # MACD (only compute if not already present)
        if "macd_line" not in result.columns:
            macd_line, signal_line, histogram = self._compute_macd(
                close,
                fast_period=self.params.macd_fast,
                slow_period=self.params.macd_slow,
                signal_period=self.params.macd_signal,
            )
            result["macd_line"] = macd_line
            result["macd_signal"] = signal_line
            result["macd_histogram"] = histogram
        else:
            macd_line = result["macd_line"]
            signal_line = result["macd_signal"]

        # MACD cross
        result["macd_cross"] = np.where(
            macd_line > signal_line, 1, np.where(macd_line < signal_line, -1, 0)
        )

        # Rate of Change
        result["roc"] = self._compute_roc(close, self.params.roc_period)

        # Momentum (simple price change)
        result["momentum"] = close - close.shift(self.params.roc_period)
        result["momentum_pct"] = close.pct_change(self.params.roc_period) * 100

        # Williams %R
        result["williams_r"] = self._compute_williams_r(
            high,
            low,
            close,
            period=self.params.stoch_k_period,
        )

        # RSI rate of change
        result["rsi_roc"] = result["rsi"].diff(5)

        # Combined momentum score
        result["momentum_score"] = self._compute_momentum_score(result)

        return result

    def _compute_rsi(
        self,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        # Use exponential moving average for smoothing
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.nan)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _compute_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int,
        d_period: int,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Compute Stochastic Oscillator.

        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        range_val = highest_high - lowest_low
        range_val = range_val.replace(0, np.nan)

        stoch_k = ((close - lowest_low) / range_val) * 100
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return stoch_k, stoch_d

    def _compute_macd(
        self,
        close: pd.Series,
        fast_period: int,
        slow_period: int,
        signal_period: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MACD (Moving Average Convergence Divergence).

        MACD Line = Fast EMA - Slow EMA
        Signal Line = EMA of MACD Line
        Histogram = MACD Line - Signal Line
        """
        fast_ema = self.ema(close, fast_period)
        slow_ema = self.ema(close, slow_period)

        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _compute_roc(
        self,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute Rate of Change.

        ROC = ((Close - Close_n) / Close_n) * 100
        """
        return ((close - close.shift(period)) / close.shift(period)) * 100

    def _compute_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute Williams %R.

        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        range_val = highest_high - lowest_low
        range_val = range_val.replace(0, np.nan)

        return ((highest_high - close) / range_val) * -100

    def _compute_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute combined momentum score (-100 to +100).

        Combines RSI, Stochastic, and MACD into a single score.
        """
        # Normalize each indicator to -50 to +50 range
        rsi_score = df["rsi"] - 50  # Already centered around 50
        stoch_score = df["stoch_k"] - 50  # Already centered around 50

        # MACD histogram normalized
        macd_std = df["macd_histogram"].rolling(20).std()
        macd_std = macd_std.replace(0, np.nan)
        macd_score = (df["macd_histogram"] / macd_std) * 25
        macd_score = macd_score.clip(-50, 50)

        # Combined score
        score = (rsi_score + stoch_score + macd_score) / 3
        return score.clip(-100, 100)

    def get_feature_names(self) -> list[str]:
        """Return list of momentum feature names."""
        return [
            "rsi",
            "rsi_oversold",
            "rsi_overbought",
            "rsi_from_oversold",
            "rsi_from_overbought",
            "stoch_k",
            "stoch_d",
            "stoch_cross",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "macd_cross",
            "roc",
            "momentum",
            "momentum_pct",
            "williams_r",
            "rsi_roc",
            "momentum_score",
        ]


# ---------------------------------------------------------------------------
# %R Trend Exhaustion (dual-period)
# ---------------------------------------------------------------------------


class PercentRExhaustion:
    """
    Dual-period Williams %R exhaustion detector.

    Fires when both a short-period and a long-period %R simultaneously
    enter overbought or oversold territory. Simultaneous extremes across
    timeframes indicate trend exhaustion — a counter-trend signal.

    Parameters
    ----------
    short : int
        Short lookback (default 14, ~3 weeks daily, ~14 bars hourly).
    long : int
        Long lookback (default 112, ~22 weeks daily — equivalent to a
        traditional 20-week cycle period).
    ob_threshold : float
        Overbought threshold in %R space. %R > ob_threshold (closer to 0)
        signals overbought. Default -20.
    os_threshold : float
        Oversold threshold. %R < os_threshold (closer to -100) signals
        oversold. Default -80.
    """

    def __init__(
        self,
        short: int = 14,
        long: int = 112,
        ob_threshold: float = -20.0,
        os_threshold: float = -80.0,
    ) -> None:
        self.short = short
        self.long = long
        self.ob_threshold = ob_threshold
        self.os_threshold = os_threshold

    @staticmethod
    def _williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        range_val = (highest_high - lowest_low).replace(0, np.nan)
        return ((highest_high - close) / range_val) * -100

    def compute(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        """
        Compute dual-period %R and exhaustion signals.

        Returns
        -------
        pd.DataFrame with columns:
            pct_r_short      – Williams %R at short period
            pct_r_long       – Williams %R at long period
            exhaustion_top   – 1 when both periods overbought (trend top)
            exhaustion_bottom– 1 when both periods oversold (trend bottom)
        """
        r_short = self._williams_r(high, low, close, self.short)
        r_long = self._williams_r(high, low, close, self.long)

        exhaustion_top = (
            (r_short > self.ob_threshold) & (r_long > self.ob_threshold)
        ).astype(int)
        exhaustion_bottom = (
            (r_short < self.os_threshold) & (r_long < self.os_threshold)
        ).astype(int)

        return pd.DataFrame(
            {
                "pct_r_short": r_short,
                "pct_r_long": r_long,
                "exhaustion_top": exhaustion_top,
                "exhaustion_bottom": exhaustion_bottom,
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Laguerre RSI + Laguerre Moving Average
# ---------------------------------------------------------------------------


class LaguerreRSI:
    """
    Laguerre RSI and Laguerre Moving Average.

    4-stage recursive Laguerre filter that dramatically reduces lag while
    preserving responsiveness. The RSI variant uses L0/L1 crossings to compute
    cumulative up/down components — resulting in a lower-whipsaw RSI proxy that
    is especially effective on daily timeframes.

    Mathematics
    -----------
    L0[i] = (1 - γ) × src[i] + γ × L0[i-1]
    L1[i] = -γ × L0[i] + L0[i-1] + γ × L1[i-1]
    L2[i] = -γ × L1[i] + L1[i-1] + γ × L2[i-1]
    L3[i] = -γ × L2[i] + L2[i-1] + γ × L3[i-1]

    cu = max(L0 - L1, 0) + max(L1 - L2, 0) + max(L2 - L3, 0)
    cd = max(L1 - L0, 0) + max(L2 - L1, 0) + max(L3 - L2, 0)
    lrsi = cu / (cu + cd)   [undefined when cu + cd = 0 → 0.5]

    lma = (L0 + 2*L1 + 2*L2 + L3) / 6   (weighted average of filter stages)

    Parameters
    ----------
    gamma : float
        Damping factor 0 < γ < 1. Higher = more smoothing, more lag.
        Default 0.5 (recommended by Ehlers).
    """

    def __init__(self, gamma: float = 0.5) -> None:
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self.gamma = gamma

    def compute(self, close: pd.Series) -> pd.DataFrame:
        """
        Parameters
        ----------
        close : pd.Series

        Returns
        -------
        pd.DataFrame with columns:
            lrsi        – Laguerre RSI [0, 1]
            lma         – Laguerre Moving Average
            lrsi_ob     – 1 when lrsi > 0.8 (overbought)
            lrsi_os     – 1 when lrsi < 0.2 (oversold)
        """
        import numpy as np

        src = close.values.astype(float)
        n = len(src)
        g = self.gamma

        l0 = np.zeros(n)
        l1 = np.zeros(n)
        l2 = np.zeros(n)
        l3 = np.zeros(n)

        # Warm-start all stages to first price so filter begins at steady state
        # (avoids ~10-bar transient where L1/L2/L3 climb from 0 and distort RSI)
        s0 = float(src[0]) if not (src[0] != src[0]) else 0.0
        for i in range(n):
            s = src[i] if not (src[i] != src[i]) else (src[i - 1] if i > 0 else s0)
            l0_prev = l0[i - 1] if i > 0 else s0
            l1_prev = l1[i - 1] if i > 0 else s0
            l2_prev = l2[i - 1] if i > 0 else s0
            l3_prev = l3[i - 1] if i > 0 else s0
            l0[i] = (1 - g) * s + g * l0_prev
            l1[i] = -g * l0[i] + l0_prev + g * l1_prev
            l2[i] = -g * l1[i] + l1_prev + g * l2_prev
            l3[i] = -g * l2[i] + l2_prev + g * l3_prev

        cu = np.maximum(l0 - l1, 0) + np.maximum(l1 - l2, 0) + np.maximum(l2 - l3, 0)
        cd = np.maximum(l1 - l0, 0) + np.maximum(l2 - l1, 0) + np.maximum(l3 - l2, 0)
        denom = cu + cd
        lrsi = np.where(denom > 0, cu / denom, 0.5)

        lma = (l0 + 2 * l1 + 2 * l2 + l3) / 6.0

        idx = close.index
        return pd.DataFrame(
            {
                "lrsi": pd.Series(lrsi, index=idx),
                "lma": pd.Series(lma, index=idx),
                "lrsi_ob": pd.Series((lrsi > 0.8).astype(int), index=idx),
                "lrsi_os": pd.Series((lrsi < 0.2).astype(int), index=idx),
            }
        )
