"""
Point-in-Time (PIT) publication delay registry.

Maps feature data sources to the calendar-day lag between the reference
period end and the date the data becomes publicly available.  Applied as
a forward shift during backtesting so that features are only consumed
after they would realistically be known.

Example:
    A quarterly fundamental (e.g. Sloan accruals) covering Q1 (ending
    2024-03-31) is typically filed within 45 days, so `available_date`
    = 2024-05-15.  The backtest should not use this value before that date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger


# Calendar-day delays by data source.
# Earnings announcements happen in real-time so delay is 0.
# SEC 10-Q filings are due 40-45 days after fiscal quarter end;
# we use 45 as a conservative upper bound.
# SEC 10-K filings are due 60 days after fiscal year end.
PUBLICATION_DELAYS: dict[str, int] = {
    "earnings_surprise": 0,
    "earnings_calendar": 0,
    "fundamental_quarterly": 45,
    "fundamental_annual": 60,
    "analyst_estimates": 0,
    "insider_transactions": 2,
    "institutional_holdings": 45,  # 13F filing deadline
    "economic_indicators": 0,
}


@dataclass(frozen=True)
class FeatureTimestamp:
    """Temporal metadata for a single feature observation."""

    feature_name: str
    source: str
    raw_date: datetime
    available_date: datetime

    @classmethod
    def from_source(
        cls, feature_name: str, source: str, raw_date: datetime
    ) -> "FeatureTimestamp":
        delay_days = PUBLICATION_DELAYS.get(source, 0)
        if delay_days == 0 and source not in PUBLICATION_DELAYS:
            logger.warning(
                f"PIT registry: unknown source '{source}' for feature "
                f"'{feature_name}' — assuming 0-day delay"
            )
        return cls(
            feature_name=feature_name,
            source=source,
            raw_date=raw_date,
            available_date=raw_date + timedelta(days=delay_days),
        )


def shift_to_available(
    df: pd.DataFrame, feature_col: str, source: str
) -> pd.Series:
    """Shift a feature column forward by its publication delay.

    For daily-indexed DataFrames the shift is expressed in trading days
    (delay_calendar_days / 1.4 ≈ trading days, rounded up).  This is a
    conservative approximation; the exact mapping depends on market
    holidays.

    Args:
        df: DataFrame with a DatetimeIndex.
        feature_col: Column name to shift.
        source: Data source key in ``PUBLICATION_DELAYS``.

    Returns:
        Shifted Series (same index, values pushed forward in time).
    """
    delay_days = PUBLICATION_DELAYS.get(source, 0)
    if delay_days == 0:
        return df[feature_col]

    # Convert calendar days → approximate trading days (conservative).
    trading_day_shift = max(1, int(delay_days / 1.4) + 1)
    shifted = df[feature_col].shift(trading_day_shift)
    logger.debug(
        f"PIT shift: {feature_col} ({source}) shifted {trading_day_shift} bars "
        f"(~{delay_days} calendar days)"
    )
    return shifted
