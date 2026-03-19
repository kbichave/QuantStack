# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.features.rates — yield curve and dual momentum signals."""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.rates import DualMomentum, YieldCurveFeatures


# ---------------------------------------------------------------------------
# YieldCurveFeatures
# ---------------------------------------------------------------------------


@pytest.fixture
def yield_data():
    """250 bars of synthetic Treasury yield data."""
    dates = pd.date_range(start="2022-01-01", periods=250, freq="D")
    np.random.seed(42)
    rate_3m = pd.Series(2.0 + np.cumsum(np.random.randn(250) * 0.03), index=dates).clip(0)
    rate_2y = pd.Series(2.5 + np.cumsum(np.random.randn(250) * 0.04), index=dates).clip(0)
    rate_10y = pd.Series(3.0 + np.cumsum(np.random.randn(250) * 0.03), index=dates).clip(0)
    return rate_3m, rate_2y, rate_10y


class TestYieldCurveFeatures:
    def test_returns_dataframe(self, yield_data):
        r3, r2, r10 = yield_data
        result = YieldCurveFeatures().compute(r3, r2, r10)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, yield_data):
        r3, r2, r10 = yield_data
        result = YieldCurveFeatures().compute(r3, r2, r10)
        expected = {
            "spread_2s10s", "spread_3m10y",
            "spread_2s10s_smooth", "spread_3m10y_smooth",
            "curve_inverted", "curve_deeply_inv",
            "spread_2s10s_zscore", "spread_3m10y_zscore",
        }
        assert expected.issubset(set(result.columns))

    def test_spread_2s10s_formula(self, yield_data):
        r3, r2, r10 = yield_data
        result = YieldCurveFeatures().compute(r3, r2, r10)
        expected = r10 - r2
        pd.testing.assert_series_equal(result["spread_2s10s"], expected, check_names=False)

    def test_spread_3m10y_formula(self, yield_data):
        r3, r2, r10 = yield_data
        result = YieldCurveFeatures().compute(r3, r2, r10)
        expected = r10 - r3
        pd.testing.assert_series_equal(result["spread_3m10y"], expected, check_names=False)

    def test_inverted_when_2y_above_10y(self):
        """When 2Y yield > 10Y yield, curve is inverted."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        rate_3m = pd.Series(np.full(50, 5.0), index=dates)
        rate_2y = pd.Series(np.full(50, 5.0), index=dates)  # 2Y = 5%
        rate_10y = pd.Series(np.full(50, 4.0), index=dates) # 10Y = 4% → inverted

        result = YieldCurveFeatures().compute(rate_3m, rate_2y, rate_10y)
        assert (result["curve_inverted"] == 1).all()

    def test_not_inverted_in_normal_curve(self):
        """Normal curve: 10Y > 2Y → inverted = 0."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        rate_3m = pd.Series(np.full(50, 2.0), index=dates)
        rate_2y = pd.Series(np.full(50, 3.0), index=dates)
        rate_10y = pd.Series(np.full(50, 4.5), index=dates)

        result = YieldCurveFeatures().compute(rate_3m, rate_2y, rate_10y)
        assert (result["curve_inverted"] == 0).all()

    def test_deeply_inverted_when_spread_below_minus_50bps(self):
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        rate_3m = pd.Series(np.full(50, 2.0), index=dates)
        rate_2y = pd.Series(np.full(50, 5.5), index=dates)   # 2Y = 5.5
        rate_10y = pd.Series(np.full(50, 4.5), index=dates)  # 10Y = 4.5 → spread = -1.0

        result = YieldCurveFeatures().compute(rate_3m, rate_2y, rate_10y)
        assert (result["curve_deeply_inv"] == 1).all()

    def test_binary_signal_columns(self, yield_data):
        r3, r2, r10 = yield_data
        result = YieldCurveFeatures().compute(r3, r2, r10)
        for col in ("curve_inverted", "curve_deeply_inv"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})


# ---------------------------------------------------------------------------
# DualMomentum
# ---------------------------------------------------------------------------


@pytest.fixture
def price_series():
    """300-bar price series — long enough for 12m lookback."""
    dates = pd.date_range(start="2021-01-01", periods=300, freq="D")
    np.random.seed(7)
    close = 100 + np.cumsum(np.random.randn(300) * 0.5)
    return pd.Series(close, index=dates)


class TestDualMomentum:
    def test_returns_dataframe(self, price_series):
        result = DualMomentum().compute(price_series)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, price_series):
        result = DualMomentum().compute(price_series)
        expected = {"momentum_12m1m", "abs_momentum_signal", "momentum_6m", "momentum_3m"}
        assert expected.issubset(set(result.columns))

    def test_signal_binary(self, price_series):
        result = DualMomentum().compute(price_series)
        vals = result["abs_momentum_signal"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_positive_momentum_in_strong_uptrend(self):
        """Consistent uptrend → positive 12m-1m momentum → signal = 1 after warmup."""
        dates = pd.date_range(start="2021-01-01", periods=300, freq="D")
        close = pd.Series(np.linspace(100, 300, 300), index=dates)
        lookback, skip = 252, 21
        result = DualMomentum(abs_lookback=lookback, skip_period=skip).compute(close)
        # Only check bars after both past_price and recent_price are available
        warmup = lookback  # past_price needs lookback bars; recent_price needs skip bars
        valid = result["abs_momentum_signal"].iloc[warmup:]
        assert valid.mean() == 1.0

    def test_negative_momentum_in_strong_downtrend(self):
        """Consistent downtrend → negative 12m-1m momentum → signal = 0 after warmup."""
        dates = pd.date_range(start="2021-01-01", periods=300, freq="D")
        close = pd.Series(np.linspace(300, 100, 300), index=dates)
        lookback, skip = 252, 21
        result = DualMomentum(abs_lookback=lookback, skip_period=skip).compute(close)
        warmup = lookback
        valid = result["abs_momentum_signal"].iloc[warmup:]
        assert valid.mean() == 0.0

    def test_momentum_12m1m_positive_in_uptrend(self):
        dates = pd.date_range(start="2021-01-01", periods=300, freq="D")
        close = pd.Series(np.linspace(100, 200, 300), index=dates)
        result = DualMomentum().compute(close)
        mom = result["momentum_12m1m"].dropna()
        assert (mom > 0).all()

    def test_all_momentum_columns_finite(self, price_series):
        result = DualMomentum().compute(price_series)
        for col in ("momentum_6m", "momentum_3m"):
            valid = result[col].dropna()
            assert np.isfinite(valid).all()


# ---------------------------------------------------------------------------
# SpreadSignals
# ---------------------------------------------------------------------------


from quantcore.features.rates import SpreadSignals


@pytest.fixture
def spread_data():
    """300 daily bars of 3M Treasury and overnight rate."""
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    np.random.seed(17)
    rate_3m = pd.Series(4.0 + np.cumsum(np.random.randn(300) * 0.02), index=dates).clip(0)
    overnight = pd.Series(3.5 + np.cumsum(np.random.randn(300) * 0.015), index=dates).clip(0)
    return rate_3m, overnight


class TestSpreadSignals:
    def test_returns_dataframe(self, spread_data):
        rate_3m, overnight = spread_data
        result = SpreadSignals().compute(rate_3m, overnight)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, spread_data):
        rate_3m, overnight = spread_data
        result = SpreadSignals().compute(rate_3m, overnight)
        for col in ("ted_spread", "ted_spread_smooth", "ted_spread_zscore",
                    "fra_ois_approx", "credit_stress", "spread_widening"):
            assert col in result.columns

    def test_ted_spread_formula(self, spread_data):
        rate_3m, overnight = spread_data
        result = SpreadSignals().compute(rate_3m, overnight)
        expected = rate_3m - overnight
        pd.testing.assert_series_equal(
            result["ted_spread"], expected, check_names=False
        )

    def test_fra_ois_equals_ted_spread(self, spread_data):
        """fra_ois_approx is the same calculation as ted_spread post-LIBOR."""
        rate_3m, overnight = spread_data
        result = SpreadSignals().compute(rate_3m, overnight)
        pd.testing.assert_series_equal(
            result["fra_ois_approx"], result["ted_spread"], check_names=False
        )

    def test_credit_stress_binary(self, spread_data):
        rate_3m, overnight = spread_data
        result = SpreadSignals().compute(rate_3m, overnight)
        assert set(result["credit_stress"].unique()).issubset({0, 1})

    def test_credit_stress_fires_above_threshold(self):
        """Construct data where spread clearly exceeds threshold."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        rate_3m = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0], index=dates)
        overnight = pd.Series([4.0, 4.0, 4.0, 4.0, 4.0], index=dates)  # spread = 1.0 > 0.5
        result = SpreadSignals(stress_threshold=0.5).compute(rate_3m, overnight)
        assert (result["credit_stress"] == 1).all()

    def test_credit_stress_absent_below_threshold(self):
        """Spread < threshold → credit_stress = 0."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        rate_3m = pd.Series([4.1, 4.1, 4.1, 4.1, 4.1], index=dates)
        overnight = pd.Series([4.0, 4.0, 4.0, 4.0, 4.0], index=dates)  # spread = 0.1 < 0.5
        result = SpreadSignals(stress_threshold=0.5).compute(rate_3m, overnight)
        assert (result["credit_stress"] == 0).all()

    def test_spread_widening_detects_increases(self):
        """spread_widening = 1 on bars where TED spread increased."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        rate_3m = pd.Series([4.0, 4.1, 4.3, 4.2, 4.5], index=dates)
        overnight = pd.Series([3.9, 3.9, 3.9, 3.9, 3.9], index=dates)
        # spread = [0.1, 0.2, 0.4, 0.3, 0.6] → widening at bars 1,2,4
        result = SpreadSignals().compute(rate_3m, overnight)
        widening = result["spread_widening"].values
        assert widening[1] == 1   # 0.2 > 0.1
        assert widening[2] == 1   # 0.4 > 0.2
        assert widening[3] == 0   # 0.3 < 0.4
        assert widening[4] == 1   # 0.6 > 0.3

    def test_smooth_is_rolling_mean(self, spread_data):
        """Smoothed spread equals rolling mean of raw spread."""
        rate_3m, overnight = spread_data
        smooth_period = 5
        result = SpreadSignals(smooth_period=smooth_period).compute(rate_3m, overnight)
        expected_smooth = result["ted_spread"].rolling(smooth_period).mean()
        pd.testing.assert_series_equal(
            result["ted_spread_smooth"], expected_smooth, check_names=False
        )

    def test_zscore_finite_after_warmup(self, spread_data):
        """Z-score should be finite after 252-bar warmup."""
        rate_3m, overnight = spread_data
        result = SpreadSignals().compute(rate_3m, overnight)
        z = result["ted_spread_zscore"]
        finite_z = z.dropna()
        assert np.isfinite(finite_z).all()

    def test_preserves_index(self, spread_data):
        """Output index matches input index."""
        rate_3m, overnight = spread_data
        result = SpreadSignals().compute(rate_3m, overnight)
        pd.testing.assert_index_equal(result.index, rate_3m.index)
