"""Tests for 4.6 — Point-in-Time Data Semantics (QS-B5)."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.pit_registry import (
    PUBLICATION_DELAYS,
    FeatureTimestamp,
    shift_to_available,
)


class TestPublicationDelays:
    def test_earnings_zero_delay(self):
        """Earnings surprise has 0-day delay (real-time announcement)."""
        assert PUBLICATION_DELAYS["earnings_surprise"] == 0

    def test_quarterly_45_days(self):
        """Quarterly fundamentals have 45-day SEC 10-Q delay."""
        assert PUBLICATION_DELAYS["fundamental_quarterly"] == 45

    def test_annual_60_days(self):
        """Annual fundamentals have 60-day SEC 10-K delay."""
        assert PUBLICATION_DELAYS["fundamental_annual"] == 60


class TestFeatureTimestamp:
    def test_from_source_with_delay(self):
        """Quarterly source shifts raw_date by 45 days."""
        ft = FeatureTimestamp.from_source(
            "sloan_accruals",
            "fundamental_quarterly",
            datetime(2024, 3, 31),
        )
        expected = datetime(2024, 3, 31) + timedelta(days=45)
        assert ft.available_date == expected
        assert ft.source == "fundamental_quarterly"

    def test_from_source_zero_delay(self):
        """Earnings surprise has no delay: available_date == raw_date."""
        ft = FeatureTimestamp.from_source(
            "eps_surprise",
            "earnings_surprise",
            datetime(2024, 4, 15),
        )
        assert ft.available_date == ft.raw_date

    def test_unknown_source_defaults_zero(self):
        """Unknown source defaults to 0-day delay (available_date == raw_date)."""
        ft = FeatureTimestamp.from_source(
            "mystery_feature",
            "unknown_source_xyz",
            datetime(2024, 1, 1),
        )
        assert ft.available_date == ft.raw_date


class TestShiftToAvailable:
    def _make_df(self, n: int = 100) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=n)
        return pd.DataFrame(
            {"accruals": np.arange(n, dtype=float), "close": 100.0 + np.arange(n)},
            index=dates,
        )

    def test_quarterly_shift_introduces_nans(self):
        """45-day delay should NaN the first ~33 trading days."""
        df = self._make_df()
        shifted = shift_to_available(df, "accruals", "fundamental_quarterly")
        # First trading_day_shift values should be NaN
        trading_day_shift = max(1, int(45 / 1.4) + 1)
        assert shifted.isna().sum() >= trading_day_shift

    def test_zero_delay_no_shift(self):
        """Earnings (0-day delay) returns the column unchanged."""
        df = self._make_df()
        shifted = shift_to_available(df, "accruals", "earnings_surprise")
        pd.testing.assert_series_equal(shifted, df["accruals"])

    def test_shift_preserves_index(self):
        """Shifted series has the same index as input."""
        df = self._make_df()
        shifted = shift_to_available(df, "accruals", "fundamental_quarterly")
        assert shifted.index.equals(df.index)
