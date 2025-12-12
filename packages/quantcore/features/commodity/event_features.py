"""
Event features for commodity trading.

Computes features based on proximity to key events (EIA, OPEC, FOMC, CPI).
"""

from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class EventFeatures(FeatureBase):
    """
    Event-based features for commodity trading.

    Features:
    - Days to/from EIA inventory report
    - Days to/from OPEC meetings
    - Days to/from FOMC decisions
    - Days to/from CPI releases
    - Event proximity scores
    - Post-event drift patterns
    """

    def __init__(
        self,
        timeframe: Timeframe,
        event_lookback: int = 20,
        event_lookforward: int = 5,
    ):
        """
        Initialize event feature calculator.

        Args:
            timeframe: Timeframe for parameter adjustment
            event_lookback: Days to look back for post-event effects
            event_lookforward: Days to look forward for pre-event effects
        """
        super().__init__(timeframe)
        self.event_lookback = event_lookback
        self.event_lookforward = event_lookforward

    def compute(
        self,
        df: pd.DataFrame,
        event_calendar: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute event features.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            event_calendar: DataFrame with event dates (from adapter)

        Returns:
            DataFrame with event features added
        """
        result = df.copy()

        # Ensure DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)

        # If event calendar provided, use it
        if event_calendar is not None and not event_calendar.empty:
            result = self._compute_from_calendar(result, event_calendar)
        else:
            # Generate approximate event dates
            result = self._compute_estimated_events(result)

        # Compute event-based return patterns
        result = self._compute_event_return_patterns(result)

        return result

    def _compute_from_calendar(
        self,
        df: pd.DataFrame,
        event_calendar: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute features from event calendar."""
        result = df.copy()

        # Process each event type
        for event_type in ["EIA", "OPEC", "FOMC", "CPI"]:
            events = event_calendar[event_calendar["event_type"] == event_type]

            if events.empty:
                self._add_empty_event_features(result, event_type.lower())
                continue

            event_dates = pd.to_datetime(events["date"]).values

            # Days to next event
            result[f"days_to_{event_type.lower()}"] = result.index.map(
                lambda x: self._days_to_next_event(x, event_dates)
            )

            # Days since last event
            result[f"days_since_{event_type.lower()}"] = result.index.map(
                lambda x: self._days_since_last_event(x, event_dates)
            )

            # Is event day
            result[f"is_{event_type.lower()}_day"] = result.index.map(
                lambda x: int(
                    any(abs((x - pd.Timestamp(d)).days) == 0 for d in event_dates)
                )
            )

            # Pre-event window (1-3 days before)
            result[f"is_pre_{event_type.lower()}"] = (
                (result[f"days_to_{event_type.lower()}"] > 0)
                & (result[f"days_to_{event_type.lower()}"] <= 3)
            ).astype(int)

            # Post-event window (1-3 days after)
            result[f"is_post_{event_type.lower()}"] = (
                (result[f"days_since_{event_type.lower()}"] > 0)
                & (result[f"days_since_{event_type.lower()}"] <= 3)
            ).astype(int)

            # Event proximity score (peaks at 1.0 on event day)
            days_away = np.minimum(
                result[f"days_to_{event_type.lower()}"].abs(),
                result[f"days_since_{event_type.lower()}"].abs(),
            )
            result[f"{event_type.lower()}_proximity"] = np.exp(-days_away / 3)

        # Combined event score
        result["any_event_proximity"] = (
            result["eia_proximity"]
            + result["opec_proximity"]
            + result["fomc_proximity"]
            + result["cpi_proximity"]
        ) / 4

        # High event risk flag
        result["high_event_risk"] = (
            (result["days_to_eia"] <= 1)
            | (result["days_to_opec"] <= 1)
            | (result["days_to_fomc"] <= 1)
        ).astype(int)

        return result

    def _compute_estimated_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate event features when no calendar available."""
        result = df.copy()

        # EIA: Every Wednesday
        result["days_to_eia"] = (2 - result.index.dayofweek) % 7
        result["days_since_eia"] = (result.index.dayofweek - 2) % 7
        result["is_eia_day"] = (result.index.dayofweek == 2).astype(int)
        result["is_pre_eia"] = (result.index.dayofweek == 1).astype(int)
        result["is_post_eia"] = (result.index.dayofweek == 3).astype(int)
        result["eia_proximity"] = np.exp(-result["days_to_eia"] / 3)

        # OPEC: Estimate first week of month
        result["days_to_opec"] = (7 - result.index.day) % 30
        result["days_since_opec"] = (result.index.day - 1) % 30
        result["is_opec_day"] = (result.index.day <= 5).astype(int)
        result["is_pre_opec"] = (
            (result.index.day >= 27) | (result.index.day <= 2)
        ).astype(int)
        result["is_post_opec"] = (
            (result.index.day > 5) & (result.index.day <= 8)
        ).astype(int)
        result["opec_proximity"] = np.exp(-np.minimum(result["days_to_opec"], 7) / 3)

        # FOMC: Estimate mid-month (around 15th, every 6 weeks)
        days_from_mid = np.abs(result.index.day - 15)
        result["days_to_fomc"] = np.minimum(days_from_mid, 30 - days_from_mid)
        result["days_since_fomc"] = result["days_to_fomc"]  # Symmetric
        result["is_fomc_day"] = (
            (result.index.day >= 14) & (result.index.day <= 16)
        ).astype(int)
        result["is_pre_fomc"] = (
            (result.index.day >= 11) & (result.index.day <= 13)
        ).astype(int)
        result["is_post_fomc"] = (
            (result.index.day >= 17) & (result.index.day <= 19)
        ).astype(int)
        result["fomc_proximity"] = np.exp(-result["days_to_fomc"] / 5)

        # CPI: Estimate around 12th of month
        days_from_cpi = np.abs(result.index.day - 12)
        result["days_to_cpi"] = np.minimum(days_from_cpi, 30 - days_from_cpi)
        result["days_since_cpi"] = result["days_to_cpi"]
        result["is_cpi_day"] = (
            (result.index.day >= 11) & (result.index.day <= 13)
        ).astype(int)
        result["is_pre_cpi"] = (
            (result.index.day >= 8) & (result.index.day <= 10)
        ).astype(int)
        result["is_post_cpi"] = (
            (result.index.day >= 14) & (result.index.day <= 16)
        ).astype(int)
        result["cpi_proximity"] = np.exp(-result["days_to_cpi"] / 5)

        # Combined scores
        result["any_event_proximity"] = (
            result["eia_proximity"]
            + result["opec_proximity"]
            + result["fomc_proximity"]
            + result["cpi_proximity"]
        ) / 4

        result["high_event_risk"] = (
            (result["days_to_eia"] <= 1)
            | (result["is_opec_day"] == 1)
            | (result["is_fomc_day"] == 1)
        ).astype(int)

        return result

    def _compute_event_return_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute return patterns around events."""
        result = df.copy()
        returns = result["close"].pct_change()

        # EIA day returns
        eia_returns = returns.where(result["is_eia_day"] == 1)
        result["eia_day_avg_return"] = eia_returns.expanding(min_periods=5).mean()
        result["eia_day_avg_vol"] = eia_returns.expanding(min_periods=5).std()

        # Pre-EIA returns
        pre_eia_returns = returns.where(result["is_pre_eia"] == 1)
        result["pre_eia_avg_return"] = pre_eia_returns.expanding(min_periods=5).mean()

        # Post-EIA returns
        post_eia_returns = returns.where(result["is_post_eia"] == 1)
        result["post_eia_avg_return"] = post_eia_returns.expanding(min_periods=5).mean()

        # Post-event drift (cumulative return in days after event)
        result["post_eia_drift"] = (
            result["close"].pct_change(3).where(result["days_since_eia"].between(1, 3))
        )

        # Event surprise indicator (return vs average on event day)
        result["eia_surprise"] = returns - result["eia_day_avg_return"]
        result["eia_surprise"] = result["eia_surprise"].where(result["is_eia_day"] == 1)

        # Volatility expansion around events
        vol_20 = returns.rolling(20).std()
        result["event_vol_ratio"] = returns.abs() / vol_20.replace(0, np.nan)

        # Event clustering (multiple events in short period)
        result["event_cluster"] = (
            result["is_eia_day"]
            + result["is_opec_day"]
            + result["is_fomc_day"]
            + result["is_cpi_day"]
        )

        return result

    def _days_to_next_event(self, date: pd.Timestamp, event_dates: np.ndarray) -> int:
        """Calculate days to next event."""
        future_events = [d for d in event_dates if pd.Timestamp(d) > date]
        if not future_events:
            return 999  # No future events
        next_event = min(future_events, key=lambda d: pd.Timestamp(d) - date)
        return (pd.Timestamp(next_event) - date).days

    def _days_since_last_event(
        self, date: pd.Timestamp, event_dates: np.ndarray
    ) -> int:
        """Calculate days since last event."""
        past_events = [d for d in event_dates if pd.Timestamp(d) <= date]
        if not past_events:
            return 999  # No past events
        last_event = max(past_events, key=lambda d: pd.Timestamp(d))
        return (date - pd.Timestamp(last_event)).days

    def _add_empty_event_features(self, df: pd.DataFrame, event_type: str) -> None:
        """Add empty event feature columns."""
        df[f"days_to_{event_type}"] = np.nan
        df[f"days_since_{event_type}"] = np.nan
        df[f"is_{event_type}_day"] = 0
        df[f"is_pre_{event_type}"] = 0
        df[f"is_post_{event_type}"] = 0
        df[f"{event_type}_proximity"] = 0

    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this class."""
        features = []

        for event_type in ["eia", "opec", "fomc", "cpi"]:
            features.extend(
                [
                    f"days_to_{event_type}",
                    f"days_since_{event_type}",
                    f"is_{event_type}_day",
                    f"is_pre_{event_type}",
                    f"is_post_{event_type}",
                    f"{event_type}_proximity",
                ]
            )

        features.extend(
            [
                "any_event_proximity",
                "high_event_risk",
                "eia_day_avg_return",
                "eia_day_avg_vol",
                "pre_eia_avg_return",
                "post_eia_avg_return",
                "post_eia_drift",
                "eia_surprise",
                "event_vol_ratio",
                "event_cluster",
            ]
        )

        return features
