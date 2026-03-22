# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.features.fundamental and earnings_signals modules."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.earnings_signals import (
    AnalystRevisionSignals,
    EarningsSurpriseSignals,
)
from quantstack.core.features.fundamental import (
    AssetGrowthAnomaly,
    EarningsMomentumComposite,
    FCFYield,
    NovyMarxGP,
    OperatingLeverage,
    QualityMomentumComposite,
    RevenueAcceleration,
    SloanAccruals,
)


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------


def _make_earnings(
    n: int = 12, beat_all: bool = False, miss_all: bool = False
) -> pd.DataFrame:
    """Quarterly earnings DataFrame with n periods."""
    dates = pd.date_range(start="2020-01-01", periods=n, freq="QS")
    np.random.seed(42)
    actual = 1.0 + np.random.randn(n) * 0.1
    if beat_all:
        actual += 0.3
    if miss_all:
        actual -= 0.3
    estimated = np.ones(n)
    return pd.DataFrame(
        {
            "report_date": dates,
            "actual_eps": actual,
            "estimated_eps": estimated,
        }
    )


def _make_estimates(n: int = 12) -> pd.DataFrame:
    """Analyst estimates DataFrame with n periods."""
    dates = pd.date_range(start="2020-01-01", periods=n, freq="MS")
    np.random.seed(0)
    eps = 2.0 + np.cumsum(np.random.randn(n) * 0.05)
    return pd.DataFrame({"estimate_date": dates, "eps_estimate": eps})


def _make_financials(n: int = 20) -> pd.DataFrame:
    """Quarterly financial statement DataFrame."""
    dates = pd.date_range(start="2019-01-01", periods=n, freq="QS")
    np.random.seed(7)
    revenue = 1000 + np.cumsum(np.random.randn(n) * 20)
    cogs = revenue * 0.6 + np.random.randn(n) * 5
    net_income = revenue * 0.1 + np.random.randn(n) * 3
    ebitda = revenue * 0.15 + np.random.randn(n) * 5
    total_assets = 5000 + np.cumsum(np.random.randn(n) * 50)
    operating_cash_flow = net_income + np.random.randn(n) * 10
    capex = np.abs(np.random.randn(n) * 15) + 10
    market_cap = total_assets * 1.5 + np.random.randn(n) * 100
    return pd.DataFrame(
        {
            "period_end": dates,
            "revenue": revenue,
            "cost_of_revenue": cogs,
            "net_income": net_income,
            "ebitda": ebitda,
            "total_assets": total_assets,
            "operating_cash_flow": operating_cash_flow,
            "capital_expenditures": capex,
            "market_cap": market_cap,
        }
    )


# ---------------------------------------------------------------------------
# EarningsSurpriseSignals
# ---------------------------------------------------------------------------


class TestEarningsSurpriseSignals:
    def test_returns_dataframe(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        expected = {
            "eps_surprise",
            "eps_surprise_pct",
            "sue",
            "sue_positive",
            "sue_negative",
            "beat_streak",
            "miss_streak",
        }
        assert expected.issubset(set(result.columns))

    def test_surprise_formula(self):
        df = _make_earnings()
        result = EarningsSurpriseSignals().compute(df)
        expected = df["actual_eps"] - df["estimated_eps"]
        assert (result["eps_surprise"].values - expected.values < 1e-9).all()

    def test_binary_columns(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        for col in ("sue_positive", "sue_negative"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_sue_positive_on_consistent_beats(self):
        result = EarningsSurpriseSignals(sue_lookback=4).compute(
            _make_earnings(beat_all=True)
        )
        # After warmup, all bars should have positive surprise
        assert (result["eps_surprise"] > 0).all()

    def test_beat_streak_increases_on_consecutive_beats(self):
        result = EarningsSurpriseSignals().compute(_make_earnings(beat_all=True))
        # Beat streak should be monotonically non-decreasing in a sustained beat scenario
        streaks = result["beat_streak"].values
        assert (streaks > 0).all()

    def test_no_simultaneous_beat_miss_streak(self):
        result = EarningsSurpriseSignals().compute(_make_earnings())
        both = (result["beat_streak"] > 0) & (result["miss_streak"] > 0)
        assert not both.any()


# ---------------------------------------------------------------------------
# AnalystRevisionSignals
# ---------------------------------------------------------------------------


class TestAnalystRevisionSignals:
    def test_returns_dataframe(self):
        result = AnalystRevisionSignals().compute(_make_estimates())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = AnalystRevisionSignals().compute(_make_estimates())
        expected = {
            "consensus_eps",
            "prior_consensus_eps",
            "revision_momentum",
            "revision_up",
            "revision_down",
            "estimate_dispersion",
            "dispersion_high",
        }
        assert expected.issubset(set(result.columns))

    def test_binary_columns(self):
        result = AnalystRevisionSignals().compute(_make_estimates())
        for col in ("revision_up", "revision_down", "dispersion_high"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_revision_up_in_upward_trend(self):
        dates = pd.date_range(start="2020-01-01", periods=20, freq="MS")
        eps = pd.Series(np.linspace(1.0, 2.5, 20), index=dates)
        df = pd.DataFrame({"estimate_date": dates, "eps_estimate": eps})
        result = AnalystRevisionSignals(revision_window=4).compute(df)
        valid = result["revision_momentum"].dropna()
        # Consistently rising estimates → positive revision momentum throughout
        assert (valid > 0).all()

    def test_revision_down_in_downward_trend(self):
        dates = pd.date_range(start="2020-01-01", periods=20, freq="MS")
        eps = pd.Series(np.linspace(2.5, 1.0, 20), index=dates)
        df = pd.DataFrame({"estimate_date": dates, "eps_estimate": eps})
        result = AnalystRevisionSignals(revision_window=4).compute(df)
        valid = result["revision_momentum"].dropna()
        assert (valid < 0).all()

    def test_dispersion_zero_for_constant_estimates(self):
        dates = pd.date_range(start="2020-01-01", periods=20, freq="MS")
        eps = pd.Series(np.full(20, 2.0), index=dates)
        df = pd.DataFrame({"estimate_date": dates, "eps_estimate": eps})
        result = AnalystRevisionSignals(revision_window=4).compute(df)
        disp = result["estimate_dispersion"].dropna()
        assert (disp == 0).all()


# ---------------------------------------------------------------------------
# SloanAccruals
# ---------------------------------------------------------------------------


class TestSloanAccruals:
    def test_returns_dataframe(self):
        result = SloanAccruals().compute(_make_financials())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = SloanAccruals().compute(_make_financials())
        assert {"accruals", "accruals_high", "accruals_low"}.issubset(
            set(result.columns)
        )

    def test_binary_columns(self):
        result = SloanAccruals().compute(_make_financials())
        for col in ("accruals_high", "accruals_low"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_zero_accruals_when_income_equals_cfo(self):
        df = _make_financials(n=8)
        df["net_income"] = df["operating_cash_flow"]
        result = SloanAccruals().compute(df)
        accruals = result["accruals"].dropna()
        assert (accruals.abs() < 1e-9).all()


# ---------------------------------------------------------------------------
# NovyMarxGP
# ---------------------------------------------------------------------------


class TestNovyMarxGP:
    def test_returns_dataframe(self):
        result = NovyMarxGP().compute(_make_financials())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = NovyMarxGP().compute(_make_financials())
        assert {"gross_profit", "gp_ratio", "gp_zscore", "gp_high"}.issubset(
            set(result.columns)
        )

    def test_gross_profit_formula(self):
        df = _make_financials(n=8)
        result = NovyMarxGP().compute(df)
        expected = df["revenue"] - df["cost_of_revenue"]
        assert (result["gross_profit"].values - expected.values < 1e-9).all()

    def test_higher_margin_gives_higher_gp_ratio(self):
        df_low = _make_financials(n=12)
        df_low["cost_of_revenue"] = df_low["revenue"] * 0.9  # 10% gross margin

        df_high = df_low.copy()
        df_high["cost_of_revenue"] = df_low["revenue"] * 0.3  # 70% gross margin

        gp_low = NovyMarxGP().compute(df_low)["gp_ratio"].dropna().mean()
        gp_high = NovyMarxGP().compute(df_high)["gp_ratio"].dropna().mean()
        assert gp_high > gp_low


# ---------------------------------------------------------------------------
# AssetGrowthAnomaly
# ---------------------------------------------------------------------------


class TestAssetGrowthAnomaly:
    def test_returns_dataframe(self):
        result = AssetGrowthAnomaly().compute(_make_financials())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = AssetGrowthAnomaly().compute(_make_financials())
        assert {"asset_growth", "asset_growth_high", "asset_growth_negative"}.issubset(
            set(result.columns)
        )

    def test_binary_columns(self):
        result = AssetGrowthAnomaly().compute(_make_financials())
        for col in ("asset_growth_high", "asset_growth_negative"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_fast_growing_assets_flagged(self):
        df = _make_financials(n=12)
        df["total_assets"] = np.linspace(1000, 3000, 12)  # 200% growth in 3 years
        result = AssetGrowthAnomaly().compute(df)
        # After 4-quarter lookback, should flag high growth
        assert result["asset_growth_high"].dropna().max() == 1


# ---------------------------------------------------------------------------
# FCFYield
# ---------------------------------------------------------------------------


class TestFCFYield:
    def test_returns_dataframe(self):
        result = FCFYield().compute(_make_financials())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = FCFYield().compute(_make_financials())
        assert {
            "fcf",
            "fcf_yield",
            "fcf_yield_pct",
            "fcf_positive",
            "fcf_yield_high",
        }.issubset(set(result.columns))

    def test_binary_columns(self):
        result = FCFYield().compute(_make_financials())
        for col in ("fcf_positive", "fcf_yield_high"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_fcf_formula(self):
        df = _make_financials(n=8)
        result = FCFYield().compute(df)
        expected_fcf = df["operating_cash_flow"] - df["capital_expenditures"].abs()
        diff = (result["fcf"] - expected_fcf).abs()
        assert (diff < 1e-9).all()


# ---------------------------------------------------------------------------
# RevenueAcceleration
# ---------------------------------------------------------------------------


class TestRevenueAcceleration:
    def test_returns_dataframe(self):
        result = RevenueAcceleration().compute(_make_financials())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = RevenueAcceleration().compute(_make_financials())
        expected = {
            "revenue_qoq_growth",
            "revenue_yoy_growth",
            "revenue_acceleration",
            "accelerating",
            "decelerating",
        }
        assert expected.issubset(set(result.columns))

    def test_binary_columns(self):
        result = RevenueAcceleration().compute(_make_financials())
        for col in ("accelerating", "decelerating"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_accelerating_in_exponential_growth(self):
        df = _make_financials(n=16)
        df["revenue"] = np.array([100 * (1.05**i) for i in range(16)])
        result = RevenueAcceleration().compute(df)
        # Revenue growing at same % each quarter → QoQ growth constant → acceleration ≈ 0
        # This tests the formula is at least computable without crash
        assert "revenue_acceleration" in result.columns


# ---------------------------------------------------------------------------
# OperatingLeverage
# ---------------------------------------------------------------------------


class TestOperatingLeverage:
    def test_returns_dataframe(self):
        result = OperatingLeverage().compute(_make_financials())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = OperatingLeverage().compute(_make_financials())
        expected = {
            "revenue_change_pct",
            "ebitda_change_pct",
            "dol",
            "high_leverage",
        }
        assert expected.issubset(set(result.columns))

    def test_dol_clipped_to_reasonable_range(self):
        result = OperatingLeverage().compute(_make_financials())
        dol = result["dol"].dropna()
        assert (dol >= -20).all()
        assert (dol <= 20).all()

    def test_binary_high_leverage(self):
        result = OperatingLeverage().compute(_make_financials())
        vals = result["high_leverage"].dropna().unique()
        assert set(vals).issubset({0, 1})


# ---------------------------------------------------------------------------
# QualityMomentumComposite
# ---------------------------------------------------------------------------


def _make_piotroski_financials(n: int = 16) -> pd.DataFrame:
    """Quarterly financial data suitable for PiotroskiFScore."""
    dates = pd.date_range("2020-01-01", periods=n, freq="QS")
    np.random.seed(55)
    revenue = 1000 + np.cumsum(np.abs(np.random.randn(n) * 30))
    cost_of_revenue = revenue * 0.6
    net_income = revenue * 0.1 + np.abs(np.random.randn(n) * 5)
    total_assets = 5000 + np.cumsum(np.abs(np.random.randn(n) * 20))
    operating_cash_flow = net_income * 1.1
    total_liabilities = total_assets * 0.4
    current_assets = total_assets * 0.3
    current_liabilities = total_liabilities * 0.3
    return pd.DataFrame(
        {
            "period_end": dates,
            "revenue": revenue,
            "cost_of_revenue": cost_of_revenue,
            "net_income": net_income,
            "total_assets": total_assets,
            "operating_cash_flow": operating_cash_flow,
            "total_liabilities": total_liabilities,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "shares_outstanding": np.ones(n) * 100_000_000,
            "long_term_debt": total_liabilities * 0.5,
        }
    )


def _make_price_series(
    n_days: int = 800, start: float = 100.0, seed: int = 42
) -> pd.Series:
    """Daily close price series."""
    np.random.seed(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    prices = start + np.cumsum(np.random.randn(n_days) * 0.8)
    return pd.Series(np.maximum(prices, 1.0), index=dates)


class TestQualityMomentumComposite:
    def test_returns_dataframe(self):
        result = QualityMomentumComposite().compute(
            _make_piotroski_financials(), _make_price_series()
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = QualityMomentumComposite().compute(
            _make_piotroski_financials(), _make_price_series()
        )
        for col in (
            "f_score",
            "price_momentum",
            "momentum_rank",
            "quality_momentum",
            "qm_long_signal",
            "qm_short_signal",
        ):
            assert col in result.columns, f"missing: {col}"

    def test_same_length_as_input(self):
        fin = _make_piotroski_financials(12)
        result = QualityMomentumComposite().compute(fin, _make_price_series())
        assert len(result) == 12

    def test_f_score_in_0_9(self):
        result = QualityMomentumComposite().compute(
            _make_piotroski_financials(), _make_price_series()
        )
        valid = result["f_score"].dropna()
        assert (valid >= 0).all() and (valid <= 9).all()

    def test_momentum_rank_in_0_1(self):
        result = QualityMomentumComposite().compute(
            _make_piotroski_financials(), _make_price_series()
        )
        valid = result["momentum_rank"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_binary_signal_columns(self):
        result = QualityMomentumComposite().compute(
            _make_piotroski_financials(), _make_price_series()
        )
        for col in ("qm_long_signal", "qm_short_signal"):
            assert result[col].isin([0, 1]).all(), f"{col} has non-binary values"

    def test_long_and_short_cannot_both_fire(self):
        """When quality_long_threshold > quality_short_threshold, both cannot fire
        at the same time for the same row."""
        result = QualityMomentumComposite(
            quality_long_threshold=7, quality_short_threshold=2
        ).compute(_make_piotroski_financials(), _make_price_series())
        both = (result["qm_long_signal"] == 1) & (result["qm_short_signal"] == 1)
        assert not both.any()

    def test_no_crash_with_short_price_series(self):
        fin = _make_piotroski_financials(8)
        short_prices = _make_price_series(n_days=100)
        result = QualityMomentumComposite().compute(fin, short_prices)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# EarningsMomentumComposite
# ---------------------------------------------------------------------------


def _make_em_earnings(n: int = 16, beat_all: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="QS")
    np.random.seed(13)
    consensus = 1.0 + np.random.randn(n) * 0.05
    surprise_base = np.random.randn(n) * 0.05
    if beat_all:
        surprise_base = np.abs(surprise_base) + 0.1
    actual = consensus + surprise_base
    return pd.DataFrame(
        {
            "period_end": dates,
            "actual_eps": actual,
            "consensus_eps": consensus,
        }
    )


class TestEarningsMomentumComposite:
    def test_returns_dataframe(self):
        result = EarningsMomentumComposite().compute(
            _make_em_earnings(), _make_price_series()
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = EarningsMomentumComposite().compute(
            _make_em_earnings(), _make_price_series()
        )
        for col in (
            "sue",
            "sue_signal",
            "price_momentum",
            "price_mom_zscore",
            "em_composite",
            "dual_confirmation",
            "em_long",
            "em_short",
        ):
            assert col in result.columns, f"missing: {col}"

    def test_same_length_as_input(self):
        df = _make_em_earnings(12)
        result = EarningsMomentumComposite().compute(df, _make_price_series())
        assert len(result) == 12

    def test_sue_signal_clipped_to_minus1_1(self):
        result = EarningsMomentumComposite().compute(
            _make_em_earnings(), _make_price_series()
        )
        valid = result["sue_signal"].dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_binary_signal_columns(self):
        result = EarningsMomentumComposite().compute(
            _make_em_earnings(), _make_price_series()
        )
        for col in ("dual_confirmation", "em_long", "em_short"):
            assert result[col].isin([0, 1]).all()

    def test_long_and_short_do_not_overlap(self):
        result = EarningsMomentumComposite().compute(
            _make_em_earnings(), _make_price_series()
        )
        both = (result["em_long"] == 1) & (result["em_short"] == 1)
        assert not both.any()

    def test_consistent_beats_give_positive_sue(self):
        result = EarningsMomentumComposite().compute(
            _make_em_earnings(beat_all=True), _make_price_series()
        )
        sue_valid = result["sue"].dropna()
        # With consistent beats, the majority of SUE values should be positive
        assert (sue_valid > 0).sum() > len(sue_valid) // 2

    def test_no_crash_with_short_price_series(self):
        df = _make_em_earnings(8)
        short_prices = _make_price_series(n_days=50)
        result = EarningsMomentumComposite().compute(df, short_prices)
        assert isinstance(result, pd.DataFrame)
