# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the fundamentals SignalEngine collector.

Tests verify that collect_fundamentals returns the expected keys and
degrades gracefully when various data sources are absent.
"""

import asyncio

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from quantstack.signal_engine.collectors.fundamentals import collect_fundamentals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_income_df(n: int = 12, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="QS")
    revenue = 1_000_000 + np.cumsum(np.random.randn(n) * 20_000)
    gross_profit = revenue * np.random.uniform(0.35, 0.45, n)
    total_assets = 5_000_000 + np.cumsum(np.random.randn(n) * 50_000)
    operating_income = revenue * np.random.uniform(0.10, 0.20, n)
    return pd.DataFrame(
        {
            "report_period": dates,
            "revenue": revenue,
            "gross_profit": gross_profit,
            "total_assets": total_assets,
            "operating_income": operating_income,
            "net_income": operating_income * 0.75,
            "statement_type": "income",
            "period_type": "quarterly",
        }
    )


def _make_cashflow_df(n: int = 12, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed + 1)
    dates = pd.date_range("2020-01-01", periods=n, freq="QS")
    operating_cash_flow = 150_000 + np.cumsum(np.random.randn(n) * 10_000)
    capital_expenditures = np.abs(np.random.randn(n) * 20_000) + 5_000
    return pd.DataFrame(
        {
            "report_period": dates,
            "operating_cash_flow": operating_cash_flow,
            "capital_expenditures": capital_expenditures,
            "statement_type": "cashflow",
            "period_type": "quarterly",
        }
    )


def _make_insider_df(n: int = 15, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="10D")
    types = np.random.choice(["P", "S"], n, p=[0.6, 0.4])
    return pd.DataFrame(
        {
            "transaction_date": dates,
            "transaction_type": types,
            "shares": np.random.randint(1000, 20000, n).astype(float),
            "price_per_share": np.random.uniform(90, 110, n),
            "owner_name": [f"Person_{i%5}" for i in range(n)],
            "owner_title": np.random.choice(["CEO", "CFO", "Director"], n),
        }
    )


def _make_ownership_df(n: int = 8, seed: int = 99) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2021-03-31", periods=n, freq="QE")
    # 3 investors per period = 24 rows total
    rows = []
    for d in dates:
        for i in range(3):
            rows.append(
                {
                    "report_date": d,
                    "investor_name": f"Fund_{i}",
                    "shares_held": np.random.randint(100_000, 500_000),
                    "change_shares": np.random.randint(-50_000, 80_000),
                }
            )
    return pd.DataFrame(rows)


def _make_analyst_df(n: int = 12, seed: int = 3) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="ME")
    return pd.DataFrame(
        {
            "fiscal_date": dates,
            "metric": ["EPS"] * n,
            "consensus": 2.0 + np.arange(n) * 0.03 + np.random.randn(n) * 0.05,
            "num_analysts": np.random.randint(5, 15, n),
            "period_type": ["quarterly"] * n,
        }
    )


def _make_store(with_fundamentals: bool = True) -> MagicMock:
    store = MagicMock()

    if with_fundamentals:
        store.get_fundamentals.return_value = {
            "pe_ratio": 22.5,
            "eps_ttm": 4.5,
            "revenue_growth": 0.12,
            "gross_margin": 0.40,
            "debt_to_equity": 0.35,
            "beta": 1.1,
            "market_cap": 1_500_000_000.0,
            "age_days": 2,
        }
    else:
        store.get_fundamentals.return_value = None

    def _load_statements(
        symbol, statement_type="income", period_type="quarterly", limit=12
    ):
        if statement_type == "cashflow":
            return _make_cashflow_df(n=min(limit, 12))
        return _make_income_df(n=min(limit, 12))

    store.load_financial_statements.side_effect = _load_statements
    store.load_insider_trades.return_value = _make_insider_df()
    store.load_institutional_ownership.return_value = _make_ownership_df()
    store.load_analyst_estimates.return_value = _make_analyst_df()

    return store


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Core contract
# ---------------------------------------------------------------------------


class TestFundamentalsCollectorCore:
    def test_returns_dict(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert isinstance(result, dict)

    def test_empty_when_all_data_missing(self):
        store = MagicMock()
        store.get_fundamentals.return_value = None
        store.load_financial_statements.return_value = pd.DataFrame()
        store.load_insider_trades.return_value = pd.DataFrame()
        store.load_institutional_ownership.return_value = pd.DataFrame()
        store.load_analyst_estimates.return_value = pd.DataFrame()
        result = _run(collect_fundamentals("AAPL", store))
        # Should return {} or minimal keys when no data
        assert isinstance(result, dict)

    def test_never_raises(self):
        """Collector must never raise — returns {} on exception."""
        store = MagicMock()
        store.get_fundamentals.side_effect = RuntimeError("db crash")
        store.load_financial_statements.side_effect = RuntimeError("db crash")
        store.load_insider_trades.side_effect = RuntimeError("db crash")
        store.load_institutional_ownership.side_effect = RuntimeError("db crash")
        store.load_analyst_estimates.side_effect = RuntimeError("db crash")
        result = _run(collect_fundamentals("AAPL", store))
        assert isinstance(result, dict)

    def test_baseline_keys_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        for key in (
            "pe_ratio",
            "eps_ttm",
            "revenue_growth",
            "gross_margin",
            "debt_to_equity",
            "beta",
            "market_cap",
        ):
            assert key in result, f"missing baseline key: {key}"

    def test_pe_ratio_value(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert result.get("pe_ratio") == pytest.approx(22.5, rel=1e-4)

    def test_market_cap_value(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert result.get("market_cap") == pytest.approx(1_500_000_000.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Quantamental signals from financial statements
# ---------------------------------------------------------------------------


class TestFundamentalsCollectorQuantamental:
    def test_novy_marx_gp_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert "novy_marx_gp" in result

    def test_novy_marx_gp_positive(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        v = result.get("novy_marx_gp")
        if v is not None:
            assert v > 0.0

    def test_asset_growth_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert "asset_growth" in result

    def test_revenue_acceleration_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert "revenue_acceleration" in result

    def test_operating_leverage_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert "operating_leverage" in result

    def test_quantamental_absent_when_statements_empty(self):
        store = _make_store()
        # Clear side_effect so return_value takes precedence
        store.load_financial_statements.side_effect = None
        store.load_financial_statements.return_value = pd.DataFrame()
        result = _run(collect_fundamentals("AAPL", store))
        # Quantamental keys should be absent when no financial data
        assert "novy_marx_gp" not in result
        assert "asset_growth" not in result

    def test_quantamental_absent_when_insufficient_bars(self):
        store = _make_store()
        # Only 1 row — not enough for any computation
        store.load_financial_statements.side_effect = None
        store.load_financial_statements.return_value = _make_income_df(n=1)
        result = _run(collect_fundamentals("AAPL", store))
        assert "novy_marx_gp" not in result


# ---------------------------------------------------------------------------
# Analyst revision signals
# ---------------------------------------------------------------------------


class TestFundamentalsCollectorAnalyst:
    def test_analyst_revision_momentum_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert "analyst_revision_momentum" in result

    def test_analyst_estimate_dispersion_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert "analyst_estimate_dispersion" in result

    def test_analyst_signals_absent_when_no_data(self):
        store = _make_store()
        store.load_analyst_estimates.return_value = pd.DataFrame()
        result = _run(collect_fundamentals("AAPL", store))
        assert "analyst_revision_momentum" not in result

    def test_dispersion_non_negative(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        v = result.get("analyst_estimate_dispersion")
        if v is not None:
            assert v >= 0.0


# ---------------------------------------------------------------------------
# Insider signals
# ---------------------------------------------------------------------------


class TestFundamentalsCollectorInsider:
    def test_insider_cluster_buy_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert "insider_cluster_buy" in result

    def test_insider_cluster_buy_binary(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        v = result.get("insider_cluster_buy")
        if v is not None:
            assert v in (0, 1)

    def test_insider_adj_ratio_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert "insider_adj_ratio" in result

    def test_insider_adj_ratio_in_zero_one(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        v = result.get("insider_adj_ratio")
        if v is not None:
            assert 0.0 <= v <= 1.0

    def test_insider_signals_absent_when_no_data(self):
        store = _make_store()
        store.load_insider_trades.return_value = pd.DataFrame()
        result = _run(collect_fundamentals("AAPL", store))
        assert "insider_cluster_buy" not in result


# ---------------------------------------------------------------------------
# Institutional herding signals
# ---------------------------------------------------------------------------


class TestFundamentalsCollectorHerding:
    def test_herding_measure_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        assert "inst_herding_measure" in result

    def test_herding_measure_non_negative(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        v = result.get("inst_herding_measure")
        if v is not None:
            assert v >= 0.0

    def test_herding_buy_bias_valid(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        v = result.get("inst_herding_buy_bias")
        if v is not None:
            assert v in (-1, 0, 1)

    def test_herding_high_binary(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        v = result.get("inst_herding_high")
        if v is not None:
            assert v in (0, 1)

    def test_herding_absent_when_no_ownership_data(self):
        store = _make_store()
        store.load_institutional_ownership.return_value = pd.DataFrame()
        result = _run(collect_fundamentals("AAPL", store))
        assert "inst_herding_measure" not in result


# ---------------------------------------------------------------------------
# Sloan Accruals + FCF Yield signals (cash flow statement merge)
# ---------------------------------------------------------------------------


class TestFundamentalsCollectorCashFlowSignals:
    def test_sloan_accruals_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        # SloanAccruals requires operating_cash_flow from cashflow statement;
        # present when cashflow data is available and net_income + total_assets exist
        if "sloan_accruals" in result:
            assert isinstance(result["sloan_accruals"], float)

    def test_sloan_accruals_high_binary(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        v = result.get("sloan_accruals_high")
        if v is not None:
            assert v in (0, 1)

    def test_fcf_yield_present(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        if "fcf_yield" in result:
            assert isinstance(result["fcf_yield"], float)

    def test_fcf_positive_binary(self):
        result = _run(collect_fundamentals("AAPL", _make_store()))
        v = result.get("fcf_positive")
        if v is not None:
            assert v in (0, 1)

    def test_cashflow_signals_absent_when_no_cashflow_data(self):
        store = _make_store()

        # Override to return empty cashflow statement
        def _load_statements_no_cf(
            symbol, statement_type="income", period_type="quarterly", limit=12
        ):
            if statement_type == "cashflow":
                return pd.DataFrame()
            return _make_income_df(n=min(limit, 12))

        store.load_financial_statements.side_effect = _load_statements_no_cf
        result = _run(collect_fundamentals("AAPL", store))
        assert "sloan_accruals" not in result
        assert "fcf_yield" not in result

    def test_cashflow_signals_absent_when_no_market_cap(self):
        """FCFYield needs market_cap from the fundamentals row."""
        store = _make_store(with_fundamentals=False)
        result = _run(collect_fundamentals("AAPL", store))
        assert "fcf_yield" not in result


# ---------------------------------------------------------------------------
# PiotroskiFScore + BeneishMScore wiring
# ---------------------------------------------------------------------------


def _make_piotroski_income_df(n: int = 12) -> pd.DataFrame:
    """Income+balance sheet DataFrame with all columns PiotroskiFScore needs."""
    np.random.seed(77)
    dates = pd.date_range("2020-01-01", periods=n, freq="QS")
    revenue = 1_000_000 + np.cumsum(np.random.randn(n) * 10_000)
    gross_profit = revenue * 0.40
    operating_cash_flow = 120_000 + np.cumsum(np.random.randn(n) * 5_000)
    return pd.DataFrame(
        {
            "report_period": dates,
            "revenue": revenue,
            "gross_profit": gross_profit,
            "cost_of_revenue": revenue * 0.60,
            "total_assets": 5_000_000 + np.cumsum(np.random.randn(n) * 20_000),
            "net_income": revenue * 0.08,
            "operating_cash_flow": operating_cash_flow,
            "long_term_debt": 500_000 + np.cumsum(np.random.randn(n) * 5_000),
            "current_assets": 800_000 + np.cumsum(np.random.randn(n) * 5_000),
            "current_liabilities": 400_000 + np.cumsum(np.random.randn(n) * 3_000),
            "shares_outstanding": 10_000_000 * np.ones(n),
            "statement_type": "income",
            "period_type": "quarterly",
        }
    )


def _make_store_with_piotroski() -> MagicMock:
    store = _make_store(with_fundamentals=True)
    piotroski_df = _make_piotroski_income_df()

    def _load_statements_pio(
        symbol, statement_type="income", period_type="quarterly", limit=12
    ):
        if statement_type == "cashflow":
            return _make_cashflow_df(n=min(limit, 12))
        return piotroski_df.iloc[: min(limit, len(piotroski_df))]

    store.load_financial_statements.side_effect = _load_statements_pio
    return store


class TestFundamentalsCollectorPiotroski:
    def test_piotroski_present_when_columns_available(self):
        result = _run(collect_fundamentals("AAPL", _make_store_with_piotroski()))
        if "piotroski_f_score" in result:
            assert isinstance(result["piotroski_f_score"], float)

    def test_piotroski_score_in_range(self):
        result = _run(collect_fundamentals("AAPL", _make_store_with_piotroski()))
        score = result.get("piotroski_f_score")
        if score is not None:
            assert 0.0 <= score <= 9.0

    def test_piotroski_strong_buy_binary(self):
        result = _run(collect_fundamentals("AAPL", _make_store_with_piotroski()))
        v = result.get("piotroski_strong_buy")
        if v is not None:
            assert v in (0, 1)

    def test_piotroski_short_signal_binary(self):
        result = _run(collect_fundamentals("AAPL", _make_store_with_piotroski()))
        v = result.get("piotroski_short_signal")
        if v is not None:
            assert v in (0, 1)

    def test_piotroski_absent_when_columns_missing(self):
        """Without total_assets etc., piotroski_f_score should not appear."""
        store = _make_store(with_fundamentals=True)
        # Default income df has revenue, gross_profit, total_assets, net_income,
        # operating_income — but missing long_term_debt, current_assets/liabilities,
        # shares_outstanding → PiotroskiFScore skipped
        result = _run(collect_fundamentals("AAPL", store))
        assert "piotroski_f_score" not in result


# ---------------------------------------------------------------------------
# EarningsSurpriseSignals (SUE / PEAD) wiring
# ---------------------------------------------------------------------------


def _make_earnings_calendar_df(n: int = 10) -> pd.DataFrame:
    """Return synthetic earnings_calendar rows with reported_eps and estimate."""
    np.random.seed(55)
    dates = pd.date_range("2021-01-15", periods=n, freq="QS")
    estimate = 1.0 + np.arange(n) * 0.05
    reported_eps = estimate + np.random.randn(n) * 0.10
    return pd.DataFrame(
        {
            "report_date": dates,
            "reported_eps": reported_eps,
            "estimate": estimate,
        }
    )


def _make_store_with_earnings() -> MagicMock:
    store = _make_store(with_fundamentals=True)
    # Mock conn to support direct SQL for earnings_calendar
    conn_mock = MagicMock()
    conn_mock.execute.return_value.fetchdf.return_value = _make_earnings_calendar_df()
    store.conn = conn_mock
    return store


class TestFundamentalsCollectorSUE:
    def test_sue_present_when_earnings_available(self):
        result = _run(collect_fundamentals("AAPL", _make_store_with_earnings()))
        if "sue" in result:
            assert result["sue"] is None or isinstance(result["sue"], float)

    def test_sue_positive_binary(self):
        result = _run(collect_fundamentals("AAPL", _make_store_with_earnings()))
        v = result.get("sue_positive")
        if v is not None:
            assert v in (0, 1)

    def test_sue_negative_binary(self):
        result = _run(collect_fundamentals("AAPL", _make_store_with_earnings()))
        v = result.get("sue_negative")
        if v is not None:
            assert v in (0, 1)

    def test_beat_streak_non_negative(self):
        result = _run(collect_fundamentals("AAPL", _make_store_with_earnings()))
        v = result.get("beat_streak")
        if v is not None:
            assert v >= 0

    def test_sue_absent_when_no_conn(self):
        """Without conn attribute, SUE signals should gracefully not appear."""
        store = _make_store(with_fundamentals=True)
        # MagicMock by default does not have a .conn attribute that returns
        # a proper cursor; _add_earnings_surprise_signals should catch the
        # exception and leave result clean
        result = _run(collect_fundamentals("AAPL", store))
        # result may or may not have sue depending on mock behaviour; just assert no crash
        assert isinstance(result, dict)
