"""Tests for TCA monthly recalibration (section-08)."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from quantstack.core.execution.tca_recalibration import (
    MIN_TRADES_FOR_FIT,
    RecalibrationResult,
    _fit_segment,
    run_tca_recalibration,
)


def _synthetic_trades(
    n: int, true_eta: float = 0.2, true_gamma: float = 0.4, noise: float = 0.001
) -> list[dict]:
    """Generate synthetic trades with known ground-truth coefficients."""
    np.random.seed(42)
    trades = []
    for _ in range(n):
        adv = np.random.uniform(100_000, 5_000_000)
        shares = np.random.uniform(100, adv * 0.05)
        daily_vol = np.random.uniform(0.005, 0.03)
        price = np.random.uniform(20, 300)
        pr = shares / adv

        # True model: normalized_slippage = gamma * pr + eta * pr^0.6
        true_slippage = true_gamma * pr + true_eta * (pr ** 0.6)
        noisy_slippage = true_slippage + np.random.normal(0, noise)

        # shortfall_bps = normalized_slippage * daily_vol * 10_000
        shortfall_bps = noisy_slippage * daily_vol * 10_000

        trades.append({
            "symbol": f"SYM{_}",
            "shares": shares,
            "shortfall_bps": shortfall_bps,
            "adv": adv,
            "daily_vol": daily_vol,
            "price": price,
            "adv_dollars": adv * price,
        })
    return trades


class TestFitSegment:
    def test_known_coefficients_recovered(self):
        """With 100 synthetic trades, fitted η and γ are within 10% of true values."""
        trades = _synthetic_trades(100, true_eta=0.2, true_gamma=0.4, noise=0.0005)
        result = _fit_segment("test", trades)

        assert not result.skipped
        assert result.eta == pytest.approx(0.2, rel=0.10)
        assert result.gamma == pytest.approx(0.4, rel=0.10)
        assert result.r_squared > 0.8

    def test_fewer_than_50_trades_skipped(self):
        """Segment with < 50 trades is skipped."""
        trades = _synthetic_trades(30)
        result = _fit_segment("small", trades)

        assert result.skipped
        assert "Insufficient" in result.skip_reason
        assert result.n_trades == 30

    def test_exactly_50_trades_runs(self):
        """Segment with exactly 50 trades is NOT skipped."""
        trades = _synthetic_trades(50)
        result = _fit_segment("test", trades)

        assert not result.skipped
        assert result.n_trades == 50

    def test_positive_coefficients_clamped(self):
        """Negative fitted coefficients are clamped to 0.001."""
        # Create trades where slippage is near zero → fit might go negative
        trades = _synthetic_trades(60, true_eta=0.001, true_gamma=0.001, noise=0.01)
        result = _fit_segment("test", trades)

        assert not result.skipped
        assert result.eta >= 0.001
        assert result.gamma >= 0.001


class TestRunRecalibration:
    def test_no_trades_returns_empty(self):
        """No qualifying trades → empty result list."""
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []

        results = run_tca_recalibration(conn)
        assert results == []

    def test_writes_to_tca_coefficients(self):
        """Successful fit writes INSERT to tca_coefficients."""
        trades = _synthetic_trades(60)

        # Mock conn for fetch and persist
        conn = MagicMock()

        # First call: _fetch_trade_data query
        fetch_rows = [
            (t["symbol"], t["shares"], t["shortfall_bps"], 5.0,
             t["adv"], t["daily_vol"], t["price"])
            for t in trades
        ]
        conn.execute.return_value.fetchall.return_value = fetch_rows

        results = run_tca_recalibration(conn)

        # Should have 3 segments (large, small, market_wide)
        assert len(results) == 3

        # At least market_wide should have been fitted (60 trades)
        market_wide = [r for r in results if r.symbol_group == "market_wide"]
        assert len(market_wide) == 1
        assert not market_wide[0].skipped

        # Verify INSERT was called
        insert_calls = [
            c for c in conn.execute.call_args_list
            if "INSERT INTO tca_coefficients" in str(c)
        ]
        assert len(insert_calls) >= 1

    def test_multiple_runs_add_rows(self):
        """Two recalibration runs insert separate rows (no upsert)."""
        trades = _synthetic_trades(60)
        conn = MagicMock()
        fetch_rows = [
            (t["symbol"], t["shares"], t["shortfall_bps"], 5.0,
             t["adv"], t["daily_vol"], t["price"])
            for t in trades
        ]
        conn.execute.return_value.fetchall.return_value = fetch_rows

        run_tca_recalibration(conn)
        run_tca_recalibration(conn)

        insert_calls = [
            c for c in conn.execute.call_args_list
            if "INSERT INTO tca_coefficients" in str(c)
        ]
        # Each run inserts for fitted segments; two runs → double the inserts
        assert len(insert_calls) >= 2
