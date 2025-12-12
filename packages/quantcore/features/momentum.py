"""
Momentum-related features.

Includes RSI, Stochastics, MACD, and Rate of Change.
"""

from typing import List
import pandas as pd
import numpy as np

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


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

    def get_feature_names(self) -> List[str]:
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
