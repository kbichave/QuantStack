# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for EarningsSurpriseSignals and AnalystRevisionSignals.
"""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.earnings_signals import EarningsSurpriseSignals, AnalystRevisionSignals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_earnings(n: int = 12, seed: int = 42) -> pd.DataFrame:
    """Return a quarterly earnings history with realistic surprises."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-15", periods=n, freq="QS")
    estimated_eps = 1.0 + np.arange(n) * 0.05
    actual_eps = estimated_eps + np.random.randn(n) * 0.10
    return pd.DataFrame({
        "report_date": dates,
        "actual_eps": actual_eps,
        "estimated_eps": estimated_eps,
    })


def _make_estimates(n: int = 12, seed: int = 7) -> pd.DataFrame:
    """Return an analyst estimate history."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="ME")
    # Trending upward revisions
    eps_estimate = 2.0 + np.arange(n) * 0.03 + np.random.randn(n) * 0.05
    analyst_count = np.random.randint(5, 20, n)
    return pd.DataFrame({
        "estimate_date": dates,
        "eps_estimate": eps_estimate,
        "analyst_count": analyst_count,
    })


# ---------------------------------------------------------------------------
# EarningsSurpriseSignals
# ---------------------------------------------------------------------------

class TestEarningsSurpriseSignals:
    def test_returns_dataframe(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        for col in ("eps_surprise", "eps_surprise_pct", "sue",
                    "sue_positive", "sue_negative", "beat_streak", "miss_streak"):
            assert col in result.columns, f"missing: {col}"

    def test_same_length_as_input(self):
        df = _make_earnings(16)
        result = EarningsSurpriseSignals().compute(df)
        assert len(result) == 16

    def test_eps_surprise_correct_sign(self):
        df = _make_earnings()
        result = EarningsSurpriseSignals().compute(df)
        expected = df.sort_values("report_date")["actual_eps"].values - df.sort_values("report_date")["estimated_eps"].values
        np.testing.assert_allclose(result["eps_surprise"].values, expected, atol=1e-9)

    def test_sue_positive_and_negative_mutually_exclusive(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        both = (result["sue_positive"] == 1) & (result["sue_negative"] == 1)
        assert not both.any()

    def test_sue_binary_values(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        assert result["sue_positive"].isin([0, 1]).all()
        assert result["sue_negative"].isin([0, 1]).all()

    def test_beat_streak_non_negative(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        assert (result["beat_streak"] >= 0).all()

    def test_miss_streak_non_negative(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        assert (result["miss_streak"] >= 0).all()

    def test_beat_miss_streaks_mutually_exclusive(self):
        """A quarter cannot have both beat_streak > 0 and miss_streak > 0."""
        result = EarningsSurpriseSignals().compute(_make_earnings())
        both_positive = (result["beat_streak"] > 0) & (result["miss_streak"] > 0)
        assert not both_positive.any()

    def test_consistent_beat_when_all_beats(self):
        """When every quarter beats, beat_streak should be monotonically increasing."""
        n = 10
        df = pd.DataFrame({
            "report_date": pd.date_range("2020-01-01", periods=n, freq="QS"),
            "actual_eps": 1.0 + np.arange(n) * 0.1 + 0.1,
            "estimated_eps": 1.0 + np.arange(n) * 0.1,
        })
        result = EarningsSurpriseSignals().compute(df)
        # All quarters beat — beat_streak should be non-zero for all
        assert (result["beat_streak"] > 0).all()

    def test_sorted_by_report_date(self):
        df = _make_earnings()
        # Shuffle the input
        df_shuffled = df.sample(frac=1, random_state=99).reset_index(drop=True)
        result = EarningsSurpriseSignals().compute(df_shuffled)
        dates = result["report_date"].values
        assert (dates[1:] >= dates[:-1]).all()

    def test_zero_estimated_eps_no_crash(self):
        df = pd.DataFrame({
            "report_date": pd.date_range("2020-01-01", periods=4, freq="QS"),
            "actual_eps": [1.0, 1.1, 0.9, 1.05],
            "estimated_eps": [0.0, 1.0, 0.95, 1.0],  # first is zero
        })
        result = EarningsSurpriseSignals().compute(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# AnalystRevisionSignals
# ---------------------------------------------------------------------------

class TestAnalystRevisionSignals:
    def test_returns_dataframe(self):
        result = AnalystRevisionSignals().compute(_make_estimates())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self):
        result = AnalystRevisionSignals().compute(_make_estimates())
        # Core columns the plan specified
        for col in ("revision_momentum", "estimate_dispersion"):
            assert col in result.columns, f"missing: {col}"

    def test_same_length_as_input(self):
        df = _make_estimates(16)
        result = AnalystRevisionSignals().compute(df)
        assert len(result) == 16

    def test_dispersion_non_negative(self):
        result = AnalystRevisionSignals().compute(_make_estimates())
        non_null = result["estimate_dispersion"].dropna()
        assert (non_null >= 0).all()

    def test_positive_revision_momentum_on_upward_trend(self):
        """Analyst estimates that trend strongly upward should show positive revision momentum."""
        n = 20
        df = pd.DataFrame({
            "estimate_date": pd.date_range("2022-01-01", periods=n, freq="ME"),
            "eps_estimate": 2.0 + np.arange(n) * 0.1,  # monotone increase
            "analyst_count": np.full(n, 10),
        })
        result = AnalystRevisionSignals().compute(df)
        # Later rows (after warmup) should have positive revision momentum
        non_null = result["revision_momentum"].dropna()
        assert (non_null.iloc[-3:] > 0).all()

    def test_sorted_by_estimate_date(self):
        df = _make_estimates()
        df_shuffled = df.sample(frac=1, random_state=13).reset_index(drop=True)
        result = AnalystRevisionSignals().compute(df_shuffled)
        dates = pd.to_datetime(result["estimate_date"]).values
        assert (dates[1:] >= dates[:-1]).all()


# ---------------------------------------------------------------------------
# EarningsImpliedMove
# ---------------------------------------------------------------------------


from quantcore.features.earnings_signals import EarningsImpliedMove


def _make_implied_move_df(n: int = 10, seed: int = 7) -> pd.DataFrame:
    """Synthetic earnings events with IV, DTE, and price data."""
    np.random.seed(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="QS")
    atm_iv = 0.30 + np.random.randn(n) * 0.05  # ~30% IV
    dte_days = np.full(n, 30)                    # 30 days to expiry
    pre_price = 100.0 + np.random.randn(n) * 5
    # realized moves: sometimes bigger, sometimes smaller than implied
    realized_pct = (0.03 + np.random.randn(n) * 0.04).clip(0.005)
    post_price = pre_price * (1 + realized_pct * np.random.choice([-1, 1], n))
    return pd.DataFrame({
        "period_end": dates,
        "atm_iv": atm_iv.clip(0.05),
        "dte_days": dte_days,
        "pre_earnings_price": pre_price.clip(50),
        "post_earnings_price": post_price.clip(50),
    })


class TestEarningsImpliedMove:
    def test_returns_dataframe(self):
        result = EarningsImpliedMove().compute(_make_implied_move_df())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self):
        result = EarningsImpliedMove().compute(_make_implied_move_df())
        for col in ("implied_move_pct", "realized_move_pct", "iv_overstated",
                    "move_surprise", "move_surprise_norm", "historical_iv_bias"):
            assert col in result.columns

    def test_implied_move_formula(self):
        """implied_move = IV × sqrt(dte / 365) × 100."""
        df = _make_implied_move_df(n=5)
        result = EarningsImpliedMove().compute(df)
        # Re-sort df to match result sort order
        df_sorted = df.copy()
        df_sorted["period_end"] = pd.to_datetime(df_sorted["period_end"])
        df_sorted = df_sorted.sort_values("period_end").reset_index(drop=True)
        expected = df_sorted["atm_iv"] * np.sqrt(df_sorted["dte_days"] / 365.0) * 100
        pd.testing.assert_series_equal(
            result["implied_move_pct"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            rtol=1e-6,
        )

    def test_realized_move_is_non_negative(self):
        result = EarningsImpliedMove().compute(_make_implied_move_df())
        assert (result["realized_move_pct"].dropna() >= 0).all()

    def test_iv_overstated_is_binary(self):
        result = EarningsImpliedMove().compute(_make_implied_move_df())
        vals = result["iv_overstated"].unique()
        assert set(vals).issubset({0, 1})

    def test_iv_overstated_logic(self):
        """Construct a case where implied > realized to assert iv_overstated = 1."""
        # IV = 30%, DTE = 30 → implied_move = 0.30 * sqrt(30/365) * 100 ≈ 8.6%
        # Realized move = 2% (much smaller) → iv_overstated = 1
        df = pd.DataFrame({
            "period_end": ["2023-01-01"],
            "atm_iv": [0.30],
            "dte_days": [30],
            "pre_earnings_price": [100.0],
            "post_earnings_price": [102.0],  # 2% realized
        })
        result = EarningsImpliedMove().compute(df)
        assert result["iv_overstated"].iloc[0] == 1

    def test_move_surprise_sign(self):
        """When realized > implied, move_surprise > 0."""
        # IV = 5%, DTE = 5 → implied ≈ 0.65%; realized = 10% → surprise > 0
        df = pd.DataFrame({
            "period_end": ["2023-01-01"],
            "atm_iv": [0.05],
            "dte_days": [5],
            "pre_earnings_price": [100.0],
            "post_earnings_price": [110.0],  # 10% realized
        })
        result = EarningsImpliedMove().compute(df)
        assert result["move_surprise"].iloc[0] > 0

    def test_zero_dte_produces_nan(self):
        """dte_days = 0 should produce NaN implied move (no valid option)."""
        df = pd.DataFrame({
            "period_end": ["2023-01-01"],
            "atm_iv": [0.25],
            "dte_days": [0],
            "pre_earnings_price": [100.0],
            "post_earnings_price": [105.0],
        })
        result = EarningsImpliedMove().compute(df)
        assert pd.isna(result["implied_move_pct"].iloc[0])

    def test_invalid_iv_produces_nan(self):
        """atm_iv below min_iv threshold should produce NaN."""
        df = pd.DataFrame({
            "period_end": ["2023-01-01"],
            "atm_iv": [0.005],  # below min_iv=0.01
            "dte_days": [30],
            "pre_earnings_price": [100.0],
            "post_earnings_price": [103.0],
        })
        result = EarningsImpliedMove().compute(df)
        assert pd.isna(result["implied_move_pct"].iloc[0])

    def test_historical_iv_bias_requires_warmup(self):
        """First few rows should be NaN (need min_periods=2 for rolling)."""
        result = EarningsImpliedMove().compute(_make_implied_move_df(n=8))
        assert pd.isna(result["historical_iv_bias"].iloc[0])

    def test_sorted_by_period_end(self):
        """Output is always sorted by period_end regardless of input order."""
        df = _make_implied_move_df(n=8)
        df_shuffled = df.sample(frac=1, random_state=99).reset_index(drop=True)
        result = EarningsImpliedMove().compute(df_shuffled)
        dates = pd.to_datetime(result["period_end"]).values
        assert (dates[1:] >= dates[:-1]).all()
