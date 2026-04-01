# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Fourier calendar features."""

import numpy as np
import pandas as pd

from quantstack.core.features.calendar_features import FourierCalendarFeatures


class TestFourierCalendarFeatures:
    def _idx(self, n: int = 200) -> pd.DatetimeIndex:
        return pd.date_range("2023-01-02", periods=n, freq="B")  # business days

    def test_returns_dataframe(self):
        result = FourierCalendarFeatures().compute(self._idx())
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        result = FourierCalendarFeatures().compute(self._idx())
        for col in (
            "sin_dow",
            "cos_dow",
            "sin_dom",
            "cos_dom",
            "sin_moy",
            "cos_moy",
            "sin_woy",
            "cos_woy",
            "is_month_end",
            "is_quarter_end",
            "is_opex",
        ):
            assert col in result.columns, f"{col} missing"

    def test_sin_cos_bounded(self):
        result = FourierCalendarFeatures().compute(self._idx())
        for col in (
            "sin_dow",
            "cos_dow",
            "sin_dom",
            "cos_dom",
            "sin_moy",
            "cos_moy",
            "sin_woy",
            "cos_woy",
        ):
            assert result[col].min() >= -1.01
            assert result[col].max() <= 1.01

    def test_binary_columns(self):
        result = FourierCalendarFeatures().compute(self._idx(500))
        for col in ("is_month_end", "is_quarter_end", "is_opex"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_opex_on_third_friday(self):
        """Third Friday of Jan 2023 is Jan 20."""
        idx = pd.DatetimeIndex(["2023-01-20"])
        result = FourierCalendarFeatures().compute(idx)
        assert result["is_opex"].iloc[0] == 1

    def test_opex_not_on_random_day(self):
        idx = pd.DatetimeIndex(["2023-01-10"])  # Tuesday
        result = FourierCalendarFeatures().compute(idx)
        assert result["is_opex"].iloc[0] == 0

    def test_month_end_fires(self):
        result = FourierCalendarFeatures().compute(self._idx(500))
        assert result["is_month_end"].sum() > 0

    def test_single_date(self):
        idx = pd.DatetimeIndex(["2023-06-15"])
        result = FourierCalendarFeatures().compute(idx)
        assert len(result) == 1
