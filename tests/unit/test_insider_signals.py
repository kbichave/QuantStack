# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for InsiderSignals (Form 4 cluster buy detection, adjusted ratio,
ownership stake change).
"""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.insider_signals import InsiderSignals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transactions(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Return a realistic Form 4 transaction history."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="5D")
    types = np.random.choice(["P", "S"], size=n, p=[0.6, 0.4])
    insiders = np.random.choice(["CEO", "CFO", "Director_A", "Director_B", "VP"], size=n)
    names = [f"Person_{r}" for r in insiders]
    shares = np.random.randint(1000, 50000, n).astype(float)
    price = np.random.uniform(80.0, 120.0, n)
    is_plan = np.random.choice([True, False], size=n, p=[0.1, 0.9])
    return pd.DataFrame({
        "transaction_date": dates,
        "transaction_type": types,
        "shares": shares,
        "price": price,
        "insider_name": names,
        "insider_role": insiders,
        "is_plan_trade": is_plan,
    })


def _buy_cluster_transactions() -> pd.DataFrame:
    """Return transactions where 4 distinct insiders buy within 30 days."""
    return pd.DataFrame({
        "transaction_date": pd.to_datetime([
            "2023-06-01", "2023-06-05", "2023-06-10", "2023-06-15", "2023-06-20",
        ]),
        "transaction_type": ["P", "P", "P", "P", "S"],
        "shares": [5000.0, 3000.0, 4000.0, 6000.0, 2000.0],
        "price": [100.0, 101.0, 99.0, 102.0, 103.0],
        "insider_name": ["Person_CEO", "Person_CFO", "Person_Dir_A", "Person_Dir_B", "Person_VP"],
        "insider_role": ["CEO", "CFO", "Director", "Director", "VP"],
        "is_plan_trade": [False, False, False, False, False],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInsiderSignalsCore:
    def test_returns_dataframe(self):
        result = InsiderSignals().compute(_make_transactions())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self):
        result = InsiderSignals().compute(_make_transactions())
        for col in ("buy_value", "sell_value", "distinct_buyers",
                    "cluster_buy", "adj_buy_sell_ratio"):
            assert col in result.columns, f"missing: {col}"

    def test_same_length_as_input(self):
        df = _make_transactions(30)
        result = InsiderSignals().compute(df)
        assert len(result) == 30

    def test_cluster_buy_is_binary(self):
        result = InsiderSignals().compute(_make_transactions())
        assert result["cluster_buy"].isin([0, 1]).all()

    def test_adj_buy_sell_ratio_in_zero_one(self):
        result = InsiderSignals().compute(_make_transactions())
        valid = result["adj_buy_sell_ratio"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_buy_value_non_negative(self):
        result = InsiderSignals().compute(_make_transactions())
        assert (result["buy_value"].dropna() >= 0).all()

    def test_sell_value_non_negative(self):
        result = InsiderSignals().compute(_make_transactions())
        assert (result["sell_value"].dropna() >= 0).all()

    def test_distinct_buyers_non_negative(self):
        result = InsiderSignals().compute(_make_transactions())
        assert (result["distinct_buyers"].dropna() >= 0).all()


class TestInsiderSignalsClusterBuy:
    def test_cluster_buy_fires_when_4_insiders_buy(self):
        """When 4 distinct insiders buy within the cluster window, cluster_buy=1."""
        df = _buy_cluster_transactions()
        result = InsiderSignals(cluster_window_days=30, cluster_min_insiders=3).compute(df)
        # At least one row should have cluster_buy = 1
        assert result["cluster_buy"].max() == 1

    def test_cluster_buy_does_not_fire_for_only_sells(self):
        """When all transactions are sells, cluster_buy should never fire."""
        df = pd.DataFrame({
            "transaction_date": pd.date_range("2023-01-01", periods=5, freq="3D"),
            "transaction_type": ["S"] * 5,
            "shares": [5000.0] * 5,
            "price": [100.0] * 5,
            "insider_name": [f"Person_{i}" for i in range(5)],
            "insider_role": ["Director"] * 5,
            "is_plan_trade": [False] * 5,
        })
        result = InsiderSignals(cluster_min_insiders=3).compute(df)
        assert result["cluster_buy"].max() == 0

    def test_plan_trades_excluded_from_ratio(self):
        """Sells marked as plan trades (10b5-1) should not affect adj_buy_sell_ratio."""
        df = pd.DataFrame({
            "transaction_date": pd.to_datetime(["2023-01-01", "2023-01-05", "2023-01-10"]),
            "transaction_type": ["P", "S", "S"],
            "shares": [10000.0, 5000.0, 5000.0],
            "price": [100.0, 105.0, 106.0],
            "insider_name": ["Person_CEO", "Person_VP", "Person_Dir"],
            "insider_role": ["CEO", "VP", "Director"],
            "is_plan_trade": [False, True, True],  # both sells are plan trades
        })
        result = InsiderSignals().compute(df)
        # With plan sells excluded, denominator = buy_value only → ratio = 1.0
        ratio_last = result["adj_buy_sell_ratio"].iloc[-1]
        if ratio_last is not None and not pd.isna(ratio_last):
            assert ratio_last == pytest.approx(1.0, abs=1e-6)


class TestInsiderSignalsEdgeCases:
    def test_empty_input_no_crash(self):
        df = pd.DataFrame(columns=[
            "transaction_date", "transaction_type", "shares",
            "price", "insider_name", "insider_role", "is_plan_trade",
        ])
        result = InsiderSignals().compute(df)
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_transaction_date(self):
        df = _make_transactions()
        df_shuffled = df.sample(frac=1, random_state=77).reset_index(drop=True)
        result = InsiderSignals().compute(df_shuffled)
        # Result is indexed by transaction_date (or has it as a column)
        if "transaction_date" in result.columns:
            dates = pd.to_datetime(result["transaction_date"]).values
        else:
            dates = pd.to_datetime(result.index).values
        assert (dates[1:] >= dates[:-1]).all()
