"""
Seasonality features for commodity trading.

Computes time-based patterns, inventory cycles, and session-based features.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class SeasonalityFeatures(FeatureBase):
    """
    Seasonality features for commodity trading.

    Features:
    - Time-of-day patterns
    - Day-of-week patterns
    - Month-of-year patterns
    - Inventory cycle (EIA Wednesday)
    - NYMEX session features
    - Seasonal return anomalies
    """

    # NYMEX crude oil pit trading: 9:00 AM - 2:30 PM ET (now electronic 24h)
    # Key times: 10:30 AM ET (EIA release), pit close 2:30 PM
    NYMEX_PIT_OPEN_HOUR = 9
    NYMEX_PIT_CLOSE_HOUR = 14
    NYMEX_PIT_CLOSE_MINUTE = 30
    EIA_RELEASE_HOUR = 10
    EIA_RELEASE_MINUTE = 30

    def __init__(
        self,
        timeframe: Timeframe,
        lookback_seasonal: int = 52,  # Weeks for seasonal patterns
    ):
        """
        Initialize seasonality feature calculator.

        Args:
            timeframe: Timeframe for parameter adjustment
            lookback_seasonal: Lookback for seasonal statistics
        """
        super().__init__(timeframe)
        self.lookback_seasonal = lookback_seasonal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute seasonality features.

        Args:
            df: OHLCV DataFrame with DatetimeIndex

        Returns:
            DataFrame with seasonality features added
        """
        result = df.copy()

        # Ensure we have a DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, converting")
            result.index = pd.to_datetime(result.index)

        # Time-based features
        result = self._compute_time_features(result)

        # Inventory cycle features
        result = self._compute_inventory_cycle_features(result)

        # NYMEX session features
        result = self._compute_session_features(result)

        # Seasonal patterns
        result = self._compute_seasonal_patterns(result)

        return result

    def _compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic time-based features."""
        result = df.copy()

        # Hour of day (for intraday data)
        result["hour"] = result.index.hour
        result["minute"] = result.index.minute

        # Day of week (0 = Monday, 4 = Friday)
        result["day_of_week"] = result.index.dayofweek
        result["is_monday"] = (result["day_of_week"] == 0).astype(int)
        result["is_friday"] = (result["day_of_week"] == 4).astype(int)

        # Day of month
        result["day_of_month"] = result.index.day
        result["is_month_start"] = (result["day_of_month"] <= 5).astype(int)
        result["is_month_end"] = (result["day_of_month"] >= 25).astype(int)

        # Month of year
        result["month"] = result.index.month
        result["quarter"] = result.index.quarter

        # Seasonal periods
        result["is_driving_season"] = result["month"].isin([5, 6, 7, 8, 9]).astype(int)
        result["is_heating_season"] = (
            result["month"].isin([11, 12, 1, 2, 3]).astype(int)
        )
        result["is_shoulder_season"] = result["month"].isin([4, 10]).astype(int)

        # Week of year
        result["week_of_year"] = result.index.isocalendar().week.astype(int)

        # Year
        result["year"] = result.index.year

        return result

    def _compute_inventory_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EIA inventory cycle features."""
        result = df.copy()

        # Days to next Wednesday (EIA day)
        days_to_eia = (2 - result.index.dayofweek) % 7
        result["days_to_eia"] = days_to_eia

        # Days since last EIA
        days_since_eia = (result.index.dayofweek - 2) % 7
        result["days_since_eia"] = days_since_eia

        # Is EIA day
        result["is_eia_day"] = (result.index.dayofweek == 2).astype(int)

        # Is pre-EIA (Tuesday)
        result["is_pre_eia"] = (result.index.dayofweek == 1).astype(int)

        # Is post-EIA (Thursday)
        result["is_post_eia"] = (result.index.dayofweek == 3).astype(int)

        # Inventory week phase (0=Mon, 1=Tue pre, 2=Wed release, 3=Thu post, 4=Fri)
        result["inventory_week_phase"] = result.index.dayofweek

        # EIA proximity score (higher = closer to release)
        # Peaks on Wednesday
        result["eia_proximity"] = 1.0 - (result["days_to_eia"] / 7)

        # For intraday data, add hour-based EIA features
        if self.timeframe in [Timeframe.H1, Timeframe.H4]:
            result["is_eia_hour"] = (
                (result.index.dayofweek == 2)
                & (result.index.hour >= self.EIA_RELEASE_HOUR)
                & (result.index.hour < self.EIA_RELEASE_HOUR + 2)
            ).astype(int)

            result["is_pre_eia_hour"] = (
                (result.index.dayofweek == 2)
                & (result.index.hour < self.EIA_RELEASE_HOUR)
            ).astype(int)

        return result

    def _compute_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute NYMEX session features."""
        result = df.copy()

        if self.timeframe not in [Timeframe.H1, Timeframe.H4]:
            # Session features only relevant for intraday
            result["is_pit_session"] = 1
            result["is_asian_session"] = 0
            result["is_european_session"] = 0
            result["is_us_session"] = 1
            return result

        hour = result.index.hour

        # NYMEX pit session (approximate)
        result["is_pit_session"] = (
            (hour >= self.NYMEX_PIT_OPEN_HOUR) & (hour < self.NYMEX_PIT_CLOSE_HOUR)
        ).astype(int)

        # Global sessions (assuming ET timezone)
        # Asian: 7 PM - 3 AM ET
        result["is_asian_session"] = ((hour >= 19) | (hour < 3)).astype(int)

        # European: 3 AM - 9 AM ET
        result["is_european_session"] = ((hour >= 3) & (hour < 9)).astype(int)

        # US session: 9 AM - 5 PM ET
        result["is_us_session"] = ((hour >= 9) & (hour < 17)).astype(int)

        # Key times
        result["is_market_open"] = (hour == 9).astype(int)
        result["is_pit_close"] = (
            (hour == self.NYMEX_PIT_CLOSE_HOUR)
            & (result.index.minute >= self.NYMEX_PIT_CLOSE_MINUTE)
        ).astype(int)

        # Time of day cycle (normalized 0-1)
        result["time_of_day"] = (hour * 60 + result.index.minute) / (24 * 60)

        # Session transition
        result["session_transition"] = (
            result["is_asian_session"].diff().abs()
            + result["is_european_session"].diff().abs()
            + result["is_us_session"].diff().abs()
        ).clip(0, 1)

        return result

    def _compute_seasonal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute seasonal return patterns."""
        result = df.copy()

        # Returns
        returns = result["close"].pct_change()

        # Day-of-week average returns (rolling)
        for dow in range(5):
            dow_mask = result["day_of_week"] == dow
            dow_returns = returns.where(dow_mask)
            col_name = f"dow_{dow}_avg_return"
            result[col_name] = dow_returns.rolling(
                min(self.lookback_seasonal * 5, len(result)), min_periods=10
            ).mean()

        # Current day vs average
        result["dow_return_vs_avg"] = returns - result.apply(
            lambda row: (
                result.loc[
                    : row.name, f"dow_{int(row['day_of_week'])}_avg_return"
                ].iloc[-1]
                if not pd.isna(row["day_of_week"])
                else np.nan
            ),
            axis=1,
        )

        # Month-of-year average returns
        for month in range(1, 13):
            month_mask = result["month"] == month
            month_returns = returns.where(month_mask)
            col_name = f"month_{month}_avg_return"
            result[col_name] = month_returns.expanding(min_periods=5).mean()

        # Seasonal strength (how strong is current seasonal pattern)
        result["seasonal_strength"] = result.apply(
            lambda row: (
                abs(
                    result.loc[
                        : row.name, f"month_{int(row['month'])}_avg_return"
                    ].iloc[-1]
                )
                if not pd.isna(row["month"])
                else np.nan
            ),
            axis=1,
        )

        # Historical same-week return
        result["same_week_hist_return"] = returns.rolling(
            self.lookback_seasonal, min_periods=4
        ).apply(
            lambda x: (
                x[x.index.isocalendar().week == x.index[-1].isocalendar().week].mean()
                if hasattr(x.index, "isocalendar")
                else np.nan
            ),
            raw=False,
        )

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this class."""
        features = [
            # Time features
            "hour",
            "minute",
            "day_of_week",
            "is_monday",
            "is_friday",
            "day_of_month",
            "is_month_start",
            "is_month_end",
            "month",
            "quarter",
            "week_of_year",
            "year",
            "is_driving_season",
            "is_heating_season",
            "is_shoulder_season",
            # Inventory cycle
            "days_to_eia",
            "days_since_eia",
            "is_eia_day",
            "is_pre_eia",
            "is_post_eia",
            "inventory_week_phase",
            "eia_proximity",
            "is_eia_hour",
            "is_pre_eia_hour",
            # Session features
            "is_pit_session",
            "is_asian_session",
            "is_european_session",
            "is_us_session",
            "is_market_open",
            "is_pit_close",
            "time_of_day",
            "session_transition",
            # Seasonal patterns
            "dow_return_vs_avg",
            "seasonal_strength",
            "same_week_hist_return",
        ]

        # Add dow and month average columns
        features.extend([f"dow_{i}_avg_return" for i in range(5)])
        features.extend([f"month_{i}_avg_return" for i in range(1, 13)])

        return features
