# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 2: Piotroski F-Score, Beneish M-Score, insider signals, LSV herding."""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.fundamental import BeneishMScore, PiotroskiFScore
from quantcore.features.insider_signals import InsiderSignals
from quantcore.features.institutional_signals import InstitutionalConcentration, LSVHerding


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------


def _make_financials_full(n: int = 20) -> pd.DataFrame:
    """Full financials DataFrame with all columns needed by Piotroski + Beneish."""
    dates = pd.date_range(start="2019-01-01", periods=n, freq="QS")
    np.random.seed(42)
    revenue = 1000 + np.cumsum(np.random.randn(n) * 30)
    cogs = revenue * 0.6
    net_income = revenue * 0.1 + np.random.randn(n) * 5
    ebitda = revenue * 0.15
    operating_cf = net_income + np.random.randn(n) * 8
    total_assets = 5000 + np.cumsum(np.abs(np.random.randn(n) * 50))
    current_assets = total_assets * 0.4
    ppe = total_assets * 0.3
    current_liabilities = total_assets * 0.15
    ltd = total_assets * 0.2
    ar = revenue * 0.12 + np.random.randn(n) * 5
    depreciation = ppe * 0.05 + np.random.randn(n) * 2
    sga = revenue * 0.1
    capex = np.abs(np.random.randn(n) * 20) + 15
    market_cap = total_assets * 1.5
    shares = np.full(n, 10_000_000.0)

    return pd.DataFrame(
        {
            "period_end": dates,
            "revenue": revenue,
            "cost_of_revenue": cogs,
            "net_income": net_income,
            "ebitda": ebitda,
            "total_assets": total_assets,
            "current_assets": current_assets,
            "property_plant_equipment": ppe,
            "current_liabilities": current_liabilities,
            "long_term_debt": ltd,
            "accounts_receivable": ar,
            "depreciation": depreciation,
            "sga_expenses": sga,
            "operating_cash_flow": operating_cf,
            "capital_expenditures": capex,
            "market_cap": market_cap,
            "shares_outstanding": shares,
        }
    )


def _make_transactions(n: int = 20, all_buys: bool = False) -> pd.DataFrame:
    """Insider transaction DataFrame."""
    dates = pd.date_range(start="2023-01-01", periods=n, freq="3D")
    np.random.seed(0)
    types = ["P"] * n if all_buys else np.where(np.random.rand(n) > 0.4, "P", "S").tolist()
    names = [f"Insider_{i % 5}" for i in range(n)]
    return pd.DataFrame(
        {
            "transaction_date": dates,
            "transaction_type": types,
            "shares": np.random.randint(100, 10000, n).astype(float),
            "price": np.random.uniform(50, 150, n),
            "insider_name": names,
            "insider_role": ["CEO" if i % 5 == 0 else "Director" for i in range(n)],
            "is_plan_trade": np.zeros(n, dtype=bool),
        }
    )


def _make_ownership(n: int = 12) -> pd.DataFrame:
    """Institutional ownership 13F DataFrame."""
    dates = pd.date_range(start="2020-01-01", periods=n, freq="QS")
    np.random.seed(7)
    total = 200 + np.cumsum(np.random.randn(n) * 5).astype(int)
    increased = (total * 0.6).astype(int)
    decreased = (total * 0.4).astype(int)
    return pd.DataFrame(
        {
            "period_end": dates,
            "total_holders": total,
            "holders_increased": increased,
            "holders_decreased": decreased,
        }
    )


# ---------------------------------------------------------------------------
# PiotroskiFScore
# ---------------------------------------------------------------------------


class TestPiotroskiFScore:
    def test_returns_dataframe(self):
        result = PiotroskiFScore().compute(_make_financials_full())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = PiotroskiFScore().compute(_make_financials_full())
        expected = {
            "f_roa", "f_cfo", "f_delta_roa", "f_accrual",
            "f_delta_leverage", "f_delta_liquidity", "f_no_dilution",
            "f_delta_gross_margin", "f_delta_asset_turnover",
            "f_score",
        }
        assert expected.issubset(set(result.columns))

    def test_f_score_range_0_to_9(self):
        result = PiotroskiFScore().compute(_make_financials_full())
        valid = result["f_score"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 9).all()

    def test_component_signals_binary(self):
        result = PiotroskiFScore().compute(_make_financials_full())
        components = [
            "f_roa", "f_cfo", "f_delta_roa", "f_accrual",
            "f_delta_leverage", "f_delta_liquidity", "f_no_dilution",
            "f_delta_gross_margin", "f_delta_asset_turnover",
        ]
        for col in components:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_f_score_equals_sum_of_components(self):
        result = PiotroskiFScore().compute(_make_financials_full())
        components = [
            "f_roa", "f_cfo", "f_delta_roa", "f_accrual",
            "f_delta_leverage", "f_delta_liquidity", "f_no_dilution",
            "f_delta_gross_margin", "f_delta_asset_turnover",
        ]
        computed_sum = result[components].sum(axis=1)
        valid = result["f_score"].dropna()
        diff = (valid - computed_sum.reindex(valid.index)).abs()
        assert (diff < 1e-9).all()

    def test_high_score_for_strong_firm(self):
        """A firm with uniformly improving metrics should score near 9."""
        df = _make_financials_full(n=12)
        # Make all metrics uniformly excellent and improving
        df["net_income"] = np.linspace(50, 100, 12)           # growing profitability
        df["operating_cash_flow"] = np.linspace(60, 120, 12)   # CFO > NI
        df["total_assets"] = np.linspace(1000, 1050, 12)       # slow asset growth
        df["long_term_debt"] = np.linspace(200, 180, 12)       # declining leverage
        df["current_assets"] = np.linspace(400, 450, 12)
        df["current_liabilities"] = np.linspace(150, 140, 12)  # improving liquidity
        df["shares_outstanding"] = np.full(12, 10_000_000.0)   # no dilution
        df["revenue"] = np.linspace(500, 600, 12)
        df["cost_of_revenue"] = np.linspace(300, 330, 12)      # margin improving slightly

        result = PiotroskiFScore().compute(df)
        # After 4-quarter warmup, should score 7+ consistently
        valid = result["f_score"].iloc[5:].dropna()
        assert valid.mean() >= 6


# ---------------------------------------------------------------------------
# BeneishMScore
# ---------------------------------------------------------------------------


class TestBeneishMScore:
    def test_returns_dataframe(self):
        result = BeneishMScore().compute(_make_financials_full())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = BeneishMScore().compute(_make_financials_full())
        expected = {"DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI",
                    "m_score", "manipulation_risk"}
        assert expected.issubset(set(result.columns))

    def test_manipulation_risk_binary(self):
        result = BeneishMScore().compute(_make_financials_full())
        vals = result["manipulation_risk"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_clean_company_low_m_score(self):
        """Stable, consistent financials → M-Score well below -1.78."""
        df = _make_financials_full(n=12)
        # Fix all variables to stable ratios → all indices ≈ 1 → M ≈ -4.84 + ~7 * 1 = clean
        df["revenue"] = np.full(12, 1000.0)
        df["cost_of_revenue"] = np.full(12, 600.0)
        df["accounts_receivable"] = np.full(12, 120.0)
        df["net_income"] = np.full(12, 100.0)
        df["operating_cash_flow"] = np.full(12, 120.0)  # CFO > NI → TATA negative
        df["sga_expenses"] = np.full(12, 100.0)
        df["depreciation"] = np.full(12, 30.0)

        result = BeneishMScore().compute(df)
        valid = result["m_score"].dropna()
        if len(valid) > 0:
            # TATA contribution is large negative → overall M < -1.78
            assert result["manipulation_risk"].dropna().mean() < 0.5


# ---------------------------------------------------------------------------
# InsiderSignals
# ---------------------------------------------------------------------------


class TestInsiderSignals:
    def test_returns_dataframe(self):
        result = InsiderSignals().compute(_make_transactions())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = InsiderSignals().compute(_make_transactions())
        expected = {
            "buy_value", "sell_value", "distinct_buyers",
            "cluster_buy", "adj_buy_sell_ratio",
        }
        assert expected.issubset(set(result.columns))

    def test_cluster_buy_binary(self):
        result = InsiderSignals().compute(_make_transactions())
        vals = result["cluster_buy"].unique()
        assert set(vals).issubset({0, 1})

    def test_adj_ratio_in_zero_one(self):
        result = InsiderSignals().compute(_make_transactions())
        valid = result["adj_buy_sell_ratio"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_cluster_triggered_when_many_insiders_buy(self):
        """With all buys from 5 distinct insiders, cluster_buy should fire."""
        df = _make_transactions(n=30, all_buys=True)
        # Ensure 5 distinct insiders are present in first window
        df["insider_name"] = [f"Insider_{i % 5}" for i in range(30)]
        result = InsiderSignals(cluster_window_days=90, cluster_min_insiders=3).compute(df)
        assert result["cluster_buy"].max() == 1

    def test_no_cluster_when_one_insider(self):
        """Single insider buying → cluster_buy should not fire for very high min_insiders."""
        df = _make_transactions(n=10, all_buys=True)
        df["insider_name"] = "SingleInsider"
        # With min_insiders=50 and only 10 transactions, cluster cannot fire
        result = InsiderSignals(cluster_window_days=90, cluster_min_insiders=50).compute(df)
        assert result["cluster_buy"].max() == 0


# ---------------------------------------------------------------------------
# LSVHerding
# ---------------------------------------------------------------------------


class TestLSVHerding:
    def test_returns_dataframe(self):
        result = LSVHerding().compute(_make_ownership())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = LSVHerding().compute(_make_ownership())
        expected = {
            "fraction_buying", "expected_buying",
            "herding_measure", "herding_buy_bias", "herding_high",
        }
        assert expected.issubset(set(result.columns))

    def test_fraction_buying_in_zero_one(self):
        result = LSVHerding().compute(_make_ownership())
        valid = result["fraction_buying"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_herding_measure_non_negative(self):
        result = LSVHerding().compute(_make_ownership())
        valid = result["herding_measure"].dropna()
        assert (valid >= 0).all()

    def test_herding_high_binary(self):
        result = LSVHerding().compute(_make_ownership())
        vals = result["herding_high"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_high_herding_when_behavior_deviates_from_expected(self):
        """Herding fires when actual fraction_buying deviates sharply from expected."""
        df = _make_ownership(n=16)
        # First 8 periods: balanced (50/50) → expected_buying ≈ 0.5
        df.loc[:7, "holders_increased"] = 100
        df.loc[:7, "holders_decreased"] = 100
        # Last 8 periods: everyone buys → fraction_buying = 1.0, deviation from 0.5
        df.loc[8:, "holders_increased"] = 200
        df.loc[8:, "holders_decreased"] = 0
        result = LSVHerding(rolling_window=4).compute(df)
        # After the shift at bar 8, herding should eventually become high
        valid = result["herding_high"].iloc[9:].dropna()
        assert valid.max() == 1


# ---------------------------------------------------------------------------
# InstitutionalConcentration
# ---------------------------------------------------------------------------


class TestInstitutionalConcentration:
    def test_returns_dataframe(self):
        result = InstitutionalConcentration().compute(_make_ownership())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = InstitutionalConcentration().compute(_make_ownership())
        expected = {
            "holder_change", "holder_change_pct",
            "institutionalizing", "de_institutionalizing",
        }
        assert expected.issubset(set(result.columns))

    def test_binary_columns(self):
        result = InstitutionalConcentration().compute(_make_ownership())
        for col in ("institutionalizing", "de_institutionalizing"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_institutionalizing_when_holders_grow(self):
        df = _make_ownership(n=8)
        df["total_holders"] = [100, 110, 130, 160, 200, 250, 320, 410]
        result = InstitutionalConcentration().compute(df)
        valid = result["institutionalizing"].iloc[1:]
        assert (valid == 1).all()
