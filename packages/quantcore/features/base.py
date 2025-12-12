# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Base class for feature computation with timeframe awareness.

This module provides the abstract base class that all feature computation
classes inherit from. It ensures consistent interfaces and provides common
utility methods for technical analysis calculations.

Example
-------
>>> from quantcore.features.base import FeatureBase
>>> from quantcore.config.timeframes import Timeframe
>>>
>>> class MyFeatures(FeatureBase):
...     def compute(self, df):
...         df = df.copy()
...         df["my_feature"] = self.zscore(df["close"], self.params.ema_period)
...         return df
...
...     def get_feature_names(self):
...         return ["my_feature"]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from quantcore.config.timeframes import TIMEFRAME_PARAMS, Timeframe, TimeframeParams


class FeatureBase(ABC):
    """
    Abstract base class for feature computation.

    All feature classes inherit from this and implement compute().
    Provides common utilities for timeframe-aware feature calculation.

    Parameters
    ----------
    timeframe : Timeframe
        The timeframe for which to compute features. This determines
        the lookback periods used in calculations.

    Attributes
    ----------
    timeframe : Timeframe
        The configured timeframe.
    params : TimeframeParams
        Timeframe-specific parameters (EMA periods, ATR periods, etc.).

    Examples
    --------
    >>> from quantcore.features.base import FeatureBase
    >>> from quantcore.config.timeframes import Timeframe
    >>>
    >>> class RSIFeatures(FeatureBase):
    ...     def compute(self, df):
    ...         df = df.copy()
    ...         delta = df["close"].diff()
    ...         gain = delta.where(delta > 0, 0)
    ...         loss = (-delta).where(delta < 0, 0)
    ...         avg_gain = self.ema(gain, self.params.rsi_period)
    ...         avg_loss = self.ema(loss, self.params.rsi_period)
    ...         rs = avg_gain / avg_loss
    ...         df["rsi"] = 100 - (100 / (1 + rs))
    ...         return df
    ...
    ...     def get_feature_names(self):
    ...         return ["rsi"]

    Notes
    -----
    All feature computations should be lag-aware to prevent lookahead bias.
    Use the `lag_features` method to shift features by at least 1 period
    before using them in models.

    See Also
    --------
    quantcore.features.factory.FeatureFactory : Factory for computing multiple feature sets.
    quantcore.features.technical_indicators.TechnicalIndicators : Standard indicators.
    """

    def __init__(self, timeframe: Timeframe) -> None:
        """
        Initialize feature calculator.

        Parameters
        ----------
        timeframe : Timeframe
            Timeframe for parameter selection. Determines lookback periods
            and other timeframe-specific calculations.
        """
        self.timeframe = timeframe
        self.params: TimeframeParams = TIMEFRAME_PARAMS[timeframe]

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex. Must contain at minimum
            columns: 'open', 'high', 'low', 'close', 'volume'.

        Returns
        -------
        pd.DataFrame
            DataFrame with original data plus computed features as new columns.

        Raises
        ------
        ValueError
            If required columns are missing from the input DataFrame.

        Notes
        -----
        Implementations should not modify the input DataFrame. Always create
        a copy first with ``df = df.copy()``.
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Return list of feature names computed by this class.

        Returns
        -------
        List[str]
            Names of all features that compute() will add to the DataFrame.

        Examples
        --------
        >>> features = MyFeatures(Timeframe.HOURLY)
        >>> features.get_feature_names()
        ['my_feature_1', 'my_feature_2']
        """
        pass

    @staticmethod
    def lag_features(
        df: pd.DataFrame,
        columns: List[str],
        lag: int = 1,
    ) -> pd.DataFrame:
        """
        Lag features to prevent lookahead bias.

        Shifts specified columns forward in time, ensuring that at time t
        only information available up to time t-lag is used.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features to lag.
        columns : List[str]
            Column names to lag. Columns not in df are silently ignored.
        lag : int, default 1
            Number of periods to lag. Must be >= 0.

        Returns
        -------
        pd.DataFrame
            DataFrame with lagged features. First `lag` rows will have NaN
            for the lagged columns.

        Examples
        --------
        >>> df = pd.DataFrame({"close": [100, 101, 102], "rsi": [30, 40, 50]})
        >>> lagged = FeatureBase.lag_features(df, ["rsi"], lag=1)
        >>> lagged["rsi"].tolist()
        [nan, 30.0, 40.0]

        Notes
        -----
        This is critical for avoiding lookahead bias in backtesting. Features
        should always be lagged by at least 1 period before being used to
        generate trading signals.
        """
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[col] = result[col].shift(lag)
        return result

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Parameters
        ----------
        series : pd.Series
            Input time series.
        period : int
            EMA span (smoothing period).

        Returns
        -------
        pd.Series
            Exponential moving average series.

        Examples
        --------
        >>> prices = pd.Series([100, 101, 102, 103, 104])
        >>> ema = FeatureBase.ema(prices, period=3)
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Parameters
        ----------
        series : pd.Series
            Input time series.
        period : int
            Window size for averaging.

        Returns
        -------
        pd.Series
            Simple moving average series.

        Notes
        -----
        The first `period - 1` values will be NaN.
        """
        return series.rolling(window=period).mean()

    @staticmethod
    def rolling_std(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate rolling standard deviation.

        Parameters
        ----------
        series : pd.Series
            Input time series.
        period : int
            Window size for calculation.

        Returns
        -------
        pd.Series
            Rolling standard deviation series.
        """
        return series.rolling(window=period).std()

    @staticmethod
    def rolling_max(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate rolling maximum.

        Parameters
        ----------
        series : pd.Series
            Input time series.
        period : int
            Window size for calculation.

        Returns
        -------
        pd.Series
            Rolling maximum series.
        """
        return series.rolling(window=period).max()

    @staticmethod
    def rolling_min(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate rolling minimum.

        Parameters
        ----------
        series : pd.Series
            Input time series.
        period : int
            Window size for calculation.

        Returns
        -------
        pd.Series
            Rolling minimum series.
        """
        return series.rolling(window=period).min()

    @staticmethod
    def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate percentage change.

        Parameters
        ----------
        series : pd.Series
            Input time series.
        periods : int, default 1
            Number of periods for calculating change.

        Returns
        -------
        pd.Series
            Percentage change series (decimal form, not percentage).

        Examples
        --------
        >>> prices = pd.Series([100, 102, 101])
        >>> FeatureBase.pct_change(prices).tolist()
        [nan, 0.02, -0.0098...]
        """
        return series.pct_change(periods=periods)

    @staticmethod
    def log_return(series: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate log returns.

        Parameters
        ----------
        series : pd.Series
            Input price series.
        periods : int, default 1
            Number of periods for calculating return.

        Returns
        -------
        pd.Series
            Log return series: ln(P_t / P_{t-periods}).

        Notes
        -----
        Log returns are additive over time, making them useful for
        statistical analysis. For small returns, log returns ≈ simple returns.
        """
        return np.log(series / series.shift(periods))

    @staticmethod
    def zscore(
        series: pd.Series,
        period: int,
        min_periods: Optional[int] = None,
    ) -> pd.Series:
        """
        Calculate rolling z-score.

        The z-score measures how many standard deviations the current value
        is from the rolling mean.

        Parameters
        ----------
        series : pd.Series
            Input time series.
        period : int
            Lookback period for mean and standard deviation.
        min_periods : int, optional
            Minimum observations required. Default is period // 2.

        Returns
        -------
        pd.Series
            Z-score series: (x - mean) / std.

        Examples
        --------
        >>> prices = pd.Series([100, 102, 98, 103, 97])
        >>> zscore = FeatureBase.zscore(prices, period=3)

        Notes
        -----
        Z-scores are commonly used for mean reversion strategies. Values
        > 2 or < -2 indicate potential reversion opportunities.
        """
        if min_periods is None:
            min_periods = period // 2

        rolling_mean = series.rolling(window=period, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=period, min_periods=min_periods).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        return (series - rolling_mean) / rolling_std

    @staticmethod
    def normalize_to_range(
        series: pd.Series,
        period: int,
        min_val: float = 0,
        max_val: float = 100,
    ) -> pd.Series:
        """
        Normalize series to a range based on rolling min/max.

        Similar to Stochastic oscillator normalization.

        Parameters
        ----------
        series : pd.Series
            Input time series.
        period : int
            Lookback period for min/max calculation.
        min_val : float, default 0
            Output minimum value.
        max_val : float, default 100
            Output maximum value.

        Returns
        -------
        pd.Series
            Normalized series in [min_val, max_val] range.

        Examples
        --------
        >>> prices = pd.Series([100, 105, 95, 110, 90])
        >>> normalized = FeatureBase.normalize_to_range(prices, period=3)
        >>> # Result will be in [0, 100] range based on rolling min/max

        See Also
        --------
        zscore : Z-score normalization for mean reversion analysis.
        """
        roll_min = series.rolling(window=period).min()
        roll_max = series.rolling(window=period).max()

        range_val = roll_max - roll_min
        range_val = range_val.replace(0, np.nan)

        normalized = (series - roll_min) / range_val
        return normalized * (max_val - min_val) + min_val
