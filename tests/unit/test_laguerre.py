# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LaguerreRSI."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.momentum import LaguerreRSI


@pytest.fixture
def close_series():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.Series(close, index=dates)


class TestLaguerreRSI:
    def test_returns_dataframe(self, close_series):
        result = LaguerreRSI().compute(close_series)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, close_series):
        result = LaguerreRSI().compute(close_series)
        assert {"lrsi", "lma", "lrsi_ob", "lrsi_os"}.issubset(set(result.columns))

    def test_lrsi_in_zero_one(self, close_series):
        result = LaguerreRSI().compute(close_series)
        assert (result["lrsi"] >= 0).all()
        assert (result["lrsi"] <= 1).all()

    def test_lrsi_ob_os_binary(self, close_series):
        result = LaguerreRSI().compute(close_series)
        assert set(result["lrsi_ob"].unique()).issubset({0, 1})
        assert set(result["lrsi_os"].unique()).issubset({0, 1})

    def test_ob_os_never_simultaneous(self, close_series):
        result = LaguerreRSI().compute(close_series)
        simultaneous = ((result["lrsi_ob"] == 1) & (result["lrsi_os"] == 1)).sum()
        assert simultaneous == 0

    def test_lma_within_price_range(self, close_series):
        result = LaguerreRSI().compute(close_series)
        assert (result["lma"] >= close_series.min() * 0.95).all()
        assert (result["lma"] <= close_series.max() * 1.05).all()

    def test_gamma_zero_invalid(self):
        with pytest.raises(ValueError):
            LaguerreRSI(gamma=0.0)

    def test_gamma_one_invalid(self):
        with pytest.raises(ValueError):
            LaguerreRSI(gamma=1.0)

    def test_flat_price_lrsi_half(self):
        """With constant price, cu = cd = 0 → lrsi defaults to 0.5."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close = pd.Series(np.full(50, 100.0), index=dates)
        result = LaguerreRSI().compute(close)
        # After warmup, lrsi should stabilise at 0.5
        assert (result["lrsi"].iloc[10:] - 0.5).abs().max() < 0.01

    def test_strong_uptrend_high_lrsi(self):
        """Persistent uptrend → lrsi should be elevated."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close = pd.Series(np.linspace(100, 200, 50), index=dates)
        result = LaguerreRSI().compute(close)
        assert result["lrsi"].iloc[20:].mean() > 0.6

    def test_strong_downtrend_low_lrsi(self):
        """Persistent downtrend → lrsi should be depressed."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close = pd.Series(np.linspace(200, 100, 50), index=dates)
        result = LaguerreRSI().compute(close)
        assert result["lrsi"].iloc[20:].mean() < 0.4

    def test_no_lookahead(self):
        """Values for first N bars unchanged when bar N+1 is appended."""
        dates = pd.date_range(start="2023-01-01", periods=51, freq="D")
        np.random.seed(7)
        vals = 100 + np.cumsum(np.random.randn(51) * 0.5)
        close_n = pd.Series(vals[:50], index=dates[:50])
        close_n1 = pd.Series(vals[:51], index=dates[:51])
        r_n = LaguerreRSI().compute(close_n)
        r_n1 = LaguerreRSI().compute(close_n1)
        pd.testing.assert_series_equal(
            r_n["lrsi"], r_n1["lrsi"].iloc[:50], check_names=False
        )

    def test_same_length_as_input(self, close_series):
        result = LaguerreRSI().compute(close_series)
        assert len(result) == len(close_series)
