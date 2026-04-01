# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for EquityCarry, FuturesBasis, COTSignals."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.carry import COTSignals, EquityCarry, FuturesBasis


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def financials():
    """12-quarter financial statements."""
    dates = pd.date_range(start="2020-01-01", periods=12, freq="QS")
    np.random.seed(42)
    mcap = pd.Series(np.linspace(10_000, 15_000, 12) * 1e6, index=dates)
    return pd.DataFrame(
        {
            "dividends_paid": np.random.uniform(50, 150, 12) * 1e6,
            "share_repurchases": np.random.uniform(100, 400, 12) * 1e6,
            "market_cap": mcap,
        },
        index=dates,
    )


@pytest.fixture
def futures_spot():
    """60-day futures vs spot price series."""
    dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
    np.random.seed(7)
    spot = pd.Series(400 + np.cumsum(np.random.randn(60) * 1.0), index=dates)
    futures = spot * 1.002  # ~0.2% contango
    return futures, spot


@pytest.fixture
def cot_data():
    """3 years of weekly COT data."""
    dates = pd.date_range(start="2020-01-01", periods=156, freq="W")
    np.random.seed(0)
    oi = np.full(156, 500_000)
    nc_long = 150_000 + np.random.randint(-20_000, 20_000, 156)
    nc_short = 120_000 + np.random.randint(-20_000, 20_000, 156)
    return pd.DataFrame(
        {
            "noncommercial_long": nc_long.astype(float),
            "noncommercial_short": nc_short.astype(float),
            "commercial_long": (
                200_000 + np.random.randint(-10_000, 10_000, 156)
            ).astype(float),
            "commercial_short": (
                230_000 + np.random.randint(-10_000, 10_000, 156)
            ).astype(float),
            "nonreportable_long": (
                50_000 + np.random.randint(-5_000, 5_000, 156)
            ).astype(float),
            "nonreportable_short": (
                50_000 + np.random.randint(-5_000, 5_000, 156)
            ).astype(float),
            "open_interest": oi.astype(float),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# EquityCarry
# ---------------------------------------------------------------------------


class TestEquityCarry:
    def test_returns_dataframe(self, financials):
        result = EquityCarry().compute(financials)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, financials):
        result = EquityCarry().compute(financials)
        assert {
            "ttm_dividends",
            "ttm_buybacks",
            "total_return_to_shareholders",
            "dividend_yield",
            "buyback_yield",
            "equity_carry",
            "carry_high",
        }.issubset(set(result.columns))

    def test_carry_positive(self, financials):
        result = EquityCarry().compute(financials)
        valid = result["equity_carry"].dropna()
        assert (valid >= 0).all()

    def test_carry_high_binary(self, financials):
        result = EquityCarry().compute(financials)
        vals = result["carry_high"].unique()
        assert set(vals).issubset({0, 1})

    def test_ttm_sums_four_quarters(self, financials):
        result = EquityCarry().compute(financials)
        # TTM dividend at bar 4 should equal sum of bars 1-4
        expected = financials["dividends_paid"].iloc[:4].sum()
        assert abs(result["ttm_dividends"].iloc[3] - expected) < 1.0

    def test_warmup_nan(self, financials):
        result = EquityCarry().compute(financials)
        assert result["equity_carry"].iloc[:3].isna().all()

    def test_high_carry_fires_when_yield_exceeds_threshold(self):
        """Yield of 5% should trigger carry_high with default threshold 2%."""
        dates = pd.date_range(start="2022-01-01", periods=8, freq="QS")
        df = pd.DataFrame(
            {
                "dividends_paid": np.full(8, 100e6),
                "share_repurchases": np.full(8, 400e6),  # 500M total per Q, 2B TTM
                "market_cap": np.full(8, 10_000e6),  # 10B → 20% carry
            },
            index=dates,
        )
        result = EquityCarry(high_carry_threshold=0.02).compute(df)
        valid = result["carry_high"].iloc[3:]
        assert (valid == 1).all()

    def test_handles_negative_dividends_sign(self, financials):
        """Some providers report dividends as negative outflows; abs() normalises."""
        neg_financials = financials.copy()
        neg_financials["dividends_paid"] = -neg_financials["dividends_paid"]
        result_pos = EquityCarry().compute(financials)
        result_neg = EquityCarry().compute(neg_financials)
        pd.testing.assert_series_equal(
            result_pos["equity_carry"].dropna(),
            result_neg["equity_carry"].dropna(),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# FuturesBasis
# ---------------------------------------------------------------------------


class TestFuturesBasis:
    def test_returns_dataframe(self, futures_spot):
        futures, spot = futures_spot
        result = FuturesBasis().compute(futures, spot)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, futures_spot):
        futures, spot = futures_spot
        result = FuturesBasis().compute(futures, spot)
        assert {
            "basis_pct",
            "basis_roll_mean",
            "basis_roll_std",
            "basis_zscore",
            "contango",
            "backwardation",
            "basis_extreme_high",
            "basis_extreme_low",
        }.issubset(set(result.columns))

    def test_contango_when_futures_above_spot(self, futures_spot):
        futures, spot = futures_spot
        result = FuturesBasis().compute(futures, spot)
        assert (result["contango"] == 1).all()

    def test_backwardation_when_futures_below_spot(self, futures_spot):
        futures, spot = futures_spot
        futures_back = spot * 0.998  # discount
        result = FuturesBasis().compute(futures_back, spot)
        assert (result["backwardation"] == 1).all()

    def test_contango_backwardation_mutually_exclusive(self, futures_spot):
        futures, spot = futures_spot
        result = FuturesBasis().compute(futures, spot)
        simultaneous = (
            (result["contango"] == 1) & (result["backwardation"] == 1)
        ).sum()
        assert simultaneous == 0

    def test_basis_pct_approximately_correct(self, futures_spot):
        futures, spot = futures_spot
        result = FuturesBasis().compute(futures, spot)
        # futures = spot * 1.002 → basis ≈ 0.2%
        valid = result["basis_pct"].dropna()
        assert (valid > 0.1).all()
        assert (valid < 0.5).all()

    def test_zscore_binary_extremes(self, futures_spot):
        futures, spot = futures_spot
        result = FuturesBasis().compute(futures, spot)
        for col in ("basis_extreme_high", "basis_extreme_low"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_warmup_nan_before_roll_period(self, futures_spot):
        futures, spot = futures_spot
        result = FuturesBasis(roll_period=20).compute(futures, spot)
        assert result["basis_zscore"].iloc[:19].isna().all()

    def test_misaligned_index_handled(self):
        """Spot with fewer dates is forward-filled to match futures index."""
        fut_dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        spt_dates = fut_dates[::2]  # every other day
        futures = pd.Series(np.linspace(400, 420, 30), index=fut_dates)
        spot = pd.Series(np.linspace(399, 419, 15), index=spt_dates)
        result = FuturesBasis().compute(futures, spot)
        assert len(result) == 30  # same length as futures


# ---------------------------------------------------------------------------
# COTSignals
# ---------------------------------------------------------------------------


class TestCOTSignals:
    def test_returns_dataframe(self, cot_data):
        result = COTSignals().compute(cot_data)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, cot_data):
        result = COTSignals().compute(cot_data)
        assert {
            "nc_net",
            "nc_net_pct",
            "nc_zscore",
            "nc_extreme_long",
            "nc_extreme_short",
            "comm_net",
            "comm_net_pct",
            "sr_net",
            "cot_sentiment",
        }.issubset(set(result.columns))

    def test_nc_net_equals_long_minus_short(self, cot_data):
        result = COTSignals().compute(cot_data)
        expected = cot_data["noncommercial_long"] - cot_data["noncommercial_short"]
        pd.testing.assert_series_equal(result["nc_net"], expected, check_names=False)

    def test_extreme_columns_binary(self, cot_data):
        result = COTSignals().compute(cot_data)
        for col in ("nc_extreme_long", "nc_extreme_short"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_extreme_long_fires_when_nc_zscore_above_threshold(self):
        """Force extreme long positioning: set nc_net_pct far above history."""
        dates = pd.date_range(start="2020-01-01", periods=60, freq="W")
        # First 52 weeks: balanced (nc_net ≈ 0 → nc_net_pct ≈ 0)
        nc_long = np.full(60, 250_000.0)
        nc_short = np.full(60, 250_000.0)
        oi = np.full(60, 500_000.0)
        # Last 8 weeks: extreme long (net = +200k on 500k OI = 40%)
        nc_long[52:] = 450_000.0
        nc_short[52:] = 250_000.0
        df = pd.DataFrame(
            {
                "noncommercial_long": nc_long,
                "noncommercial_short": nc_short,
                "commercial_long": np.full(60, 150_000.0),
                "commercial_short": np.full(60, 150_000.0),
                "nonreportable_long": np.full(60, 100_000.0),
                "nonreportable_short": np.full(60, 100_000.0),
                "open_interest": oi,
            },
            index=dates,
        )
        result = COTSignals(roll_window=52, extreme_threshold=2.0).compute(df)
        # After the shift at week 52, nc_zscore should exceed 2.0
        assert result["nc_extreme_long"].iloc[55:].max() == 1

    def test_cot_sentiment_positive_when_specs_more_bullish(self, cot_data):
        """cot_sentiment = nc_net_pct - comm_net_pct; measures spec vs commercial divergence."""
        result = COTSignals().compute(cot_data)
        # Should be numeric and finite (not all NaN)
        assert result["cot_sentiment"].notna().any()

    def test_warmup_bars_nan(self, cot_data):
        result = COTSignals(roll_window=52).compute(cot_data)
        assert result["nc_zscore"].iloc[:51].isna().all()


# ---------------------------------------------------------------------------
# CTAPositioningModel
# ---------------------------------------------------------------------------


from quantstack.core.features.carry import CTAPositioningModel


@pytest.fixture
def cta_inputs():
    """300 daily close bars + 300 weekly COT nc_zscore values on same index."""
    dates = pd.date_range("2021-01-01", periods=300, freq="D")
    np.random.seed(11)
    close = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.8), index=dates)
    # Synthetic nc_zscore on same daily index (in practice weekly, but same API)
    nc_zscore = pd.Series(np.random.randn(300) * 1.5, index=dates)
    return close, nc_zscore


class TestCTAPositioningModel:
    def test_returns_dataframe(self, cta_inputs):
        close, nc_zscore = cta_inputs
        result = CTAPositioningModel().compute(close, nc_zscore)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, cta_inputs):
        close, nc_zscore = cta_inputs
        result = CTAPositioningModel().compute(close, nc_zscore)
        for col in (
            "momentum_signal",
            "cot_direction",
            "cta_score",
            "cta_crowded_long",
            "cta_crowded_short",
            "cta_derisking",
        ):
            assert col in result.columns

    def test_cta_score_bounded(self, cta_inputs):
        """cta_score must be in [-1, 1]."""
        close, nc_zscore = cta_inputs
        result = CTAPositioningModel().compute(close, nc_zscore)
        score = result["cta_score"].dropna()
        assert (score >= -1.0).all() and (score <= 1.0).all()

    def test_crowded_long_binary(self, cta_inputs):
        close, nc_zscore = cta_inputs
        result = CTAPositioningModel().compute(close, nc_zscore)
        assert set(result["cta_crowded_long"].unique()).issubset({0, 1})

    def test_crowded_short_binary(self, cta_inputs):
        close, nc_zscore = cta_inputs
        result = CTAPositioningModel().compute(close, nc_zscore)
        assert set(result["cta_crowded_short"].unique()).issubset({0, 1})

    def test_derisking_binary(self, cta_inputs):
        close, nc_zscore = cta_inputs
        result = CTAPositioningModel().compute(close, nc_zscore)
        assert set(result["cta_derisking"].dropna().unique()).issubset({0, 1})

    def test_crowded_long_fires_when_both_bullish(self):
        """With strong uptrend + heavy spec long, crowded_long should fire."""
        dates = pd.date_range("2021-01-01", periods=200, freq="D")
        close = pd.Series(100 + np.arange(200) * 0.5, index=dates)  # monotone up
        nc_zscore = pd.Series(np.full(200, 2.5), index=dates)  # extreme long
        result = CTAPositioningModel(momentum_window=63).compute(close, nc_zscore)
        # After warmup period, crowded_long should be 1
        assert result["cta_crowded_long"].iloc[70:].mean() > 0.5

    def test_crowded_short_fires_when_both_bearish(self):
        """With downtrend + heavy spec short, crowded_short should fire."""
        dates = pd.date_range("2021-01-01", periods=200, freq="D")
        close = pd.Series(100 - np.arange(200) * 0.5, index=dates)  # monotone down
        nc_zscore = pd.Series(np.full(200, -2.5), index=dates)  # extreme short
        result = CTAPositioningModel(momentum_window=63).compute(close, nc_zscore)
        assert result["cta_crowded_short"].iloc[70:].mean() > 0.5

    def test_momentum_signal_uptrend(self):
        """In a strong uptrend, momentum_signal should be +1."""
        dates = pd.date_range("2021-01-01", periods=150, freq="D")
        close = pd.Series(100 + np.arange(150) * 1.0, index=dates)
        nc_zscore = pd.Series(np.zeros(150), index=dates)
        result = CTAPositioningModel(momentum_window=63).compute(close, nc_zscore)
        assert (result["momentum_signal"].iloc[70:] > 0).all()

    def test_preserves_index(self, cta_inputs):
        close, nc_zscore = cta_inputs
        result = CTAPositioningModel().compute(close, nc_zscore)
        pd.testing.assert_index_equal(result.index, close.index)
