"""
Trendline features based on support/resistance line fitting.

Adapted from QuantAgent's trend analysis approach:
- Fits optimized support and resistance trendlines
- Computes slope, angle, distance metrics
- Detects breakouts and channel characteristics

References:
    QuantAgent: https://github.com/Y-Research-SBU/QuantAgent
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class TrendlineFeatures(FeatureBase):
    """
    Trendline-based features using optimized support/resistance fitting.

    Features computed:
    - Support/resistance slopes (close-based and H/L-based)
    - Distance to support/resistance lines
    - Trendline angles (degrees)
    - Channel width
    - Breakout signals
    """

    def __init__(self, timeframe: Timeframe, lookback_period: int = 50):
        """
        Initialize trendline feature calculator.

        Args:
            timeframe: Timeframe for feature computation
            lookback_period: Number of bars to use for trendline fitting
        """
        super().__init__(timeframe)
        self.lookback_period = lookback_period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trendline features for the given data.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with trendline features added
        """
        result = df.copy()

        if len(result) < self.lookback_period:
            logger.warning(
                f"Insufficient data for trendlines: {len(result)} < {self.lookback_period}"
            )
            # Initialize empty columns
            for col in self.get_feature_names():
                result[col] = np.nan
            return result

        # Compute rolling trendlines
        result = self._compute_rolling_trendlines(result)

        return result

    def _compute_rolling_trendlines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trendlines using a rolling window.

        This ensures features are causal (no lookahead bias).
        """
        result = df.copy()

        # Initialize feature columns
        result["tl_support_slope_close"] = np.nan
        result["tl_resist_slope_close"] = np.nan
        result["tl_support_slope_hl"] = np.nan
        result["tl_resist_slope_hl"] = np.nan
        result["tl_support_intercept_close"] = np.nan
        result["tl_resist_intercept_close"] = np.nan
        result["tl_dist_to_support"] = np.nan
        result["tl_dist_to_resist"] = np.nan
        result["tl_channel_width"] = np.nan
        result["tl_support_angle"] = np.nan
        result["tl_resist_angle"] = np.nan
        result["tl_price_position"] = np.nan  # 0=support, 1=resist, 0.5=middle
        result["tl_breakout_above"] = 0
        result["tl_breakout_below"] = 0

        # Compute for each bar using only past data
        for i in range(self.lookback_period, len(result)):
            window_close = result["close"].iloc[i - self.lookback_period : i].values
            window_high = result["high"].iloc[i - self.lookback_period : i].values
            window_low = result["low"].iloc[i - self.lookback_period : i].values
            current_close = result["close"].iloc[i]
            current_high = result["high"].iloc[i]
            current_low = result["low"].iloc[i]

            # Fit trendlines on close prices
            try:
                support_coef_c, resist_coef_c = self._fit_trendlines_single(
                    window_close
                )

                # Store slope and intercept
                result.iloc[i, result.columns.get_loc("tl_support_slope_close")] = (
                    support_coef_c[0]
                )
                result.iloc[i, result.columns.get_loc("tl_resist_slope_close")] = (
                    resist_coef_c[0]
                )
                result.iloc[i, result.columns.get_loc("tl_support_intercept_close")] = (
                    support_coef_c[1]
                )
                result.iloc[i, result.columns.get_loc("tl_resist_intercept_close")] = (
                    resist_coef_c[1]
                )

                # Project trendlines to current bar
                support_val = (
                    support_coef_c[0] * (len(window_close) - 1) + support_coef_c[1]
                )
                resist_val = (
                    resist_coef_c[0] * (len(window_close) - 1) + resist_coef_c[1]
                )

                # Distance to trendlines (normalized by price)
                result.iloc[i, result.columns.get_loc("tl_dist_to_support")] = (
                    (current_close - support_val) / current_close * 100
                )
                result.iloc[i, result.columns.get_loc("tl_dist_to_resist")] = (
                    (resist_val - current_close) / current_close * 100
                )

                # Channel width (normalized)
                channel_width = (resist_val - support_val) / current_close * 100
                result.iloc[i, result.columns.get_loc("tl_channel_width")] = (
                    channel_width
                )

                # Price position in channel (0=support, 1=resistance)
                if channel_width > 0:
                    price_pos = (current_close - support_val) / (
                        resist_val - support_val
                    )
                    result.iloc[i, result.columns.get_loc("tl_price_position")] = (
                        np.clip(price_pos, 0, 1)
                    )

                # Trendline angles (degrees)
                # Use average bar-to-bar price change for normalization
                avg_price_change = np.mean(np.abs(np.diff(window_close)))
                if avg_price_change > 0:
                    support_angle = np.degrees(
                        np.arctan(support_coef_c[0] / avg_price_change)
                    )
                    resist_angle = np.degrees(
                        np.arctan(resist_coef_c[0] / avg_price_change)
                    )
                    result.iloc[i, result.columns.get_loc("tl_support_angle")] = (
                        support_angle
                    )
                    result.iloc[i, result.columns.get_loc("tl_resist_angle")] = (
                        resist_angle
                    )

                # Breakout detection (price crosses trendline)
                if i > self.lookback_period:
                    prev_high = result["high"].iloc[i - 1]
                    prev_low = result["low"].iloc[i - 1]
                    prev_resist = (
                        resist_coef_c[0] * (len(window_close) - 2) + resist_coef_c[1]
                    )
                    prev_support = (
                        support_coef_c[0] * (len(window_close) - 2) + support_coef_c[1]
                    )

                    # Breakout above resistance
                    if prev_high <= prev_resist and current_high > resist_val:
                        result.iloc[i, result.columns.get_loc("tl_breakout_above")] = 1

                    # Breakout below support
                    if prev_low >= prev_support and current_low < support_val:
                        result.iloc[i, result.columns.get_loc("tl_breakout_below")] = 1

            except Exception as e:
                logger.debug(f"Trendline fitting failed at index {i}: {e}")
                continue

            # Fit trendlines on high/low
            try:
                support_coef_hl, resist_coef_hl = self._fit_trendlines_high_low(
                    window_high, window_low, window_close
                )
                result.iloc[i, result.columns.get_loc("tl_support_slope_hl")] = (
                    support_coef_hl[0]
                )
                result.iloc[i, result.columns.get_loc("tl_resist_slope_hl")] = (
                    resist_coef_hl[0]
                )
            except Exception as e:
                logger.debug(f"H/L trendline fitting failed at index {i}: {e}")
                continue

        return result

    def _fit_trendlines_single(
        self, data: np.ndarray
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Fit support and resistance trendlines to a single price series.

        Adapted from QuantAgent's fit_trendlines_single().

        Args:
            data: Price series (typically close prices)

        Returns:
            Tuple of (support_coefs, resist_coefs) where each is (slope, intercept)
        """
        # Find line of best fit (least squares)
        x = np.arange(len(data))
        coefs = np.polyfit(x, data, 1)

        # Get points of line
        line_points = coefs[0] * x + coefs[1]

        # Find upper and lower pivot points
        upper_pivot = (data - line_points).argmax()
        lower_pivot = (data - line_points).argmin()

        # Optimize the slope for both trendlines
        support_coefs = self._optimize_slope(True, lower_pivot, coefs[0], data)
        resist_coefs = self._optimize_slope(False, upper_pivot, coefs[0], data)

        return support_coefs, resist_coefs

    def _fit_trendlines_high_low(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Fit trendlines using high/low data.

        Adapted from QuantAgent's fit_trendlines_high_low().

        Args:
            high: High prices
            low: Low prices
            close: Close prices (for pivot detection)

        Returns:
            Tuple of (support_coefs, resist_coefs)
        """
        x = np.arange(len(close))
        coefs = np.polyfit(x, close, 1)

        line_points = coefs[0] * x + coefs[1]
        upper_pivot = (high - line_points).argmax()
        lower_pivot = (low - line_points).argmin()

        support_coefs = self._optimize_slope(True, lower_pivot, coefs[0], low)
        resist_coefs = self._optimize_slope(False, upper_pivot, coefs[0], high)

        return support_coefs, resist_coefs

    def _optimize_slope(
        self,
        support: bool,
        pivot: int,
        init_slope: float,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Optimize trendline slope to minimize error while respecting constraints.

        Adapted from QuantAgent's optimize_slope().

        Args:
            support: True for support line, False for resistance
            pivot: Index of pivot point
            init_slope: Initial slope estimate
            y: Price data

        Returns:
            Tuple of (slope, intercept)
        """
        # Amount to change slope by
        slope_unit = (y.max() - y.min()) / len(y)

        # Optimization variables
        opt_step = 1.0
        min_step = 0.0001
        curr_step = opt_step

        # Initialize at the slope of the line of best fit
        best_slope = init_slope
        best_err = self._check_trend_line(support, pivot, init_slope, y)

        if best_err < 0:
            # Initial slope is invalid, return as-is
            return (init_slope, -init_slope * pivot + y[pivot])

        get_derivative = True
        derivative = None

        max_iterations = 100
        iteration = 0

        while curr_step > min_step and iteration < max_iterations:
            iteration += 1

            if get_derivative:
                # Numerical differentiation
                slope_change = best_slope + slope_unit * min_step
                test_err = self._check_trend_line(support, pivot, slope_change, y)
                derivative = test_err - best_err

                if test_err < 0.0:
                    slope_change = best_slope - slope_unit * min_step
                    test_err = self._check_trend_line(support, pivot, slope_change, y)
                    derivative = best_err - test_err

                if test_err < 0.0:
                    # Derivative failed, stop
                    break

                get_derivative = False

            if derivative > 0.0:
                test_slope = best_slope - slope_unit * curr_step
            else:
                test_slope = best_slope + slope_unit * curr_step

            test_err = self._check_trend_line(support, pivot, test_slope, y)

            if test_err < 0 or test_err >= best_err:
                # Slope failed or didn't reduce error
                curr_step *= 0.5
            else:
                # Test slope reduced error
                best_err = test_err
                best_slope = test_slope
                get_derivative = True

        # Return slope and intercept
        return (best_slope, -best_slope * pivot + y[pivot])

    def _check_trend_line(
        self,
        support: bool,
        pivot: int,
        slope: float,
        y: np.ndarray,
    ) -> float:
        """
        Check if trendline is valid and compute error.

        Adapted from QuantAgent's check_trend_line().

        Args:
            support: True for support line
            pivot: Pivot point index
            slope: Slope to test
            y: Price data

        Returns:
            Squared sum of errors, or -1 if invalid
        """
        # Find intercept of line through pivot with given slope
        intercept = -slope * pivot + y[pivot]

        line_vals = slope * np.arange(len(y)) + intercept
        diffs = line_vals - y

        # Check validity
        if support and diffs.max() > 1e-5:
            return -1.0
        elif not support and diffs.min() < -1e-5:
            return -1.0

        # Squared sum of differences
        err = (diffs**2.0).sum()
        return err

    def get_feature_names(self) -> List[str]:
        """Return list of trendline feature names."""
        return [
            "tl_support_slope_close",
            "tl_resist_slope_close",
            "tl_support_slope_hl",
            "tl_resist_slope_hl",
            "tl_support_intercept_close",
            "tl_resist_intercept_close",
            "tl_dist_to_support",
            "tl_dist_to_resist",
            "tl_channel_width",
            "tl_support_angle",
            "tl_resist_angle",
            "tl_price_position",
            "tl_breakout_above",
            "tl_breakout_below",
        ]
