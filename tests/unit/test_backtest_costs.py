"""Tests for 4.1 — Realistic Transaction Costs (QS-B1)."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.backtesting.engine import BacktestConfig, BacktestEngine


def _make_price_data(n: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    """Deterministic uptrending price data."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    prices = start_price + np.arange(n) * 0.5  # steady uptrend
    return pd.DataFrame(
        {
            "open": prices - 0.1,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": np.full(n, 1_000_000),
        },
        index=dates,
    )


def _make_signals(price_data: pd.DataFrame) -> pd.DataFrame:
    """Buy at bar 10, sell at bar 50."""
    signals = pd.DataFrame(
        {"signal": 0, "signal_direction": "NONE"}, index=price_data.index
    )
    signals.iloc[10] = {"signal": 1, "signal_direction": "LONG"}
    # Stay in position until bar 50
    for i in range(11, 50):
        signals.iloc[i] = {"signal": 1, "signal_direction": "LONG"}
    signals.iloc[50] = {"signal": 0, "signal_direction": "NONE"}
    return signals


class TestBpsTransactionCosts:
    def test_default_30_bps(self):
        """Default BacktestConfig uses 30 bps all-in cost."""
        cfg = BacktestConfig()
        assert cfg.all_in_cost_bps == 30.0

    def test_bps_cost_reduces_pnl_vs_zero_cost(self):
        """A 30 bps cost should reduce P&L compared to 0 cost."""
        prices = _make_price_data()
        signals = _make_signals(prices)

        zero_cost = BacktestConfig(all_in_cost_bps=0.0)
        high_cost = BacktestConfig(all_in_cost_bps=30.0)

        result_zero = BacktestEngine(zero_cost).run(signals, prices)
        result_high = BacktestEngine(high_cost).run(signals, prices)

        assert result_zero.total_return > result_high.total_return

    def test_higher_bps_means_lower_pnl(self):
        """50 bps should reduce P&L more than 15 bps."""
        prices = _make_price_data()
        signals = _make_signals(prices)

        low = BacktestConfig(all_in_cost_bps=15.0)
        high = BacktestConfig(all_in_cost_bps=50.0)

        r_low = BacktestEngine(low).run(signals, prices)
        r_high = BacktestEngine(high).run(signals, prices)

        assert r_low.total_return > r_high.total_return

    def test_cost_fraction_math(self):
        """30 bps = 0.003 = 0.3% per leg."""
        cfg = BacktestConfig(all_in_cost_bps=30.0)
        cost_frac = cfg.all_in_cost_bps / 10_000
        assert abs(cost_frac - 0.003) < 1e-10
