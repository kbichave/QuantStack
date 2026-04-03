"""Tests for risk monitoring nodes (Section 06)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.risk.monitoring import (
    RiskSnapshot,
    check_drawdown_cascade,
    check_factor_limits,
    compute_correlation_matrix,
    compute_factor_exposures,
    compute_risk_snapshot,
    compute_var_historical,
)


# -- 6.1 Portfolio Risk Snapshot --


def test_snapshot_captures_all_metrics():
    """Snapshot has equity, exposure, P&L, position count."""
    positions = [
        {"symbol": "AAPL", "quantity": 100, "current_price": 150.0, "side": "long", "avg_cost": 140.0},
        {"symbol": "MSFT", "quantity": -50, "current_price": 300.0, "side": "short", "avg_cost": 310.0},
    ]
    snap = compute_risk_snapshot(positions, total_equity=100_000, daily_pnl=500)

    assert snap.total_equity == 100_000
    assert snap.gross_exposure == 100 * 150 + 50 * 300  # 15000 + 15000 = 30000
    assert snap.net_exposure == 100 * 150 - 50 * 300  # 15000 - 15000 = 0
    assert snap.position_count == 2
    assert snap.daily_pnl == 500
    assert 0 <= snap.largest_position_pct <= 1.0


def test_snapshot_empty_portfolio():
    """Empty portfolio produces valid snapshot with zero exposure."""
    snap = compute_risk_snapshot([], total_equity=50_000, daily_pnl=0)

    assert snap.gross_exposure == 0.0
    assert snap.net_exposure == 0.0
    assert snap.position_count == 0
    assert snap.largest_position_pct == 0.0


def test_snapshot_largest_position_pct():
    """Largest position % computed correctly."""
    positions = [
        {"symbol": "AAPL", "quantity": 100, "current_price": 200.0, "side": "long"},
        {"symbol": "MSFT", "quantity": 50, "current_price": 100.0, "side": "long"},
    ]
    snap = compute_risk_snapshot(positions, total_equity=100_000, daily_pnl=0)
    # AAPL = $20,000 / $100,000 = 20%
    assert abs(snap.largest_position_pct - 0.20) < 0.01


# -- 6.2 Factor Exposure Check --


def _synthetic_returns(n: int = 120, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)

    market = rng.normal(0.001, 0.01, n)
    # AAPL has beta ~1.5 to market
    aapl = 1.5 * market + rng.normal(0, 0.005, n)
    msft = 0.8 * market + rng.normal(0, 0.005, n)

    pos_returns = pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=dates)
    factor_returns = pd.DataFrame({
        "market": market,
        "momentum": rng.normal(0, 0.008, n),
        "value": rng.normal(0, 0.006, n),
    }, index=dates)

    weights = {"AAPL": 0.6, "MSFT": 0.4}
    return pos_returns, factor_returns, weights


def test_factor_betas_computed():
    """Factor betas computed from rolling regression."""
    pos_ret, fac_ret, weights = _synthetic_returns()
    betas = compute_factor_exposures(pos_ret, fac_ret, weights, window=60)

    assert "market_beta" in betas
    assert betas["market_beta"] is not None
    # Portfolio beta to market should be positive (AAPL=1.5, MSFT=0.8, weighted)
    assert betas["market_beta"] > 0


def test_factor_soft_limit_flagged():
    """Soft limit violation flagged for momentum/value beta > 0.5."""
    warnings = check_factor_limits(
        {"market_beta": 0.3, "momentum_beta": 0.7, "value_beta": 0.2}
    )
    assert any("momentum_beta" in w for w in warnings)


def test_factor_hard_limit_flagged():
    """|market_beta| > 2.0 flagged as hard limit."""
    warnings = check_factor_limits(
        {"market_beta": 2.5, "momentum_beta": 0.1, "value_beta": 0.1}
    )
    assert any("HARD" in w for w in warnings)


def test_factor_insufficient_data():
    """< 60 days of data -> None betas (no error)."""
    pos_ret = pd.DataFrame({"AAPL": np.zeros(30)}, index=pd.bdate_range("2024-01-01", periods=30))
    fac_ret = pd.DataFrame({"market": np.zeros(30)}, index=pos_ret.index)
    betas = compute_factor_exposures(pos_ret, fac_ret, {"AAPL": 1.0}, window=60)
    assert betas["market_beta"] is None


# -- 6.3 Correlation Monitor --


def test_correlation_matrix_computed():
    """60-day rolling correlation computed correctly."""
    pos_ret, _, _ = _synthetic_returns(n=120)
    corr, avg = compute_correlation_matrix(pos_ret, window=60)

    assert corr.shape == (2, 2)
    assert -1.0 <= avg <= 1.0


def test_high_avg_correlation_detected():
    """Average pairwise correlation > 0.5 detected."""
    rng = np.random.default_rng(42)
    # Highly correlated returns
    base = rng.normal(0, 0.01, 100)
    df = pd.DataFrame({
        "A": base + rng.normal(0, 0.001, 100),
        "B": base + rng.normal(0, 0.001, 100),
    }, index=pd.bdate_range("2024-01-01", periods=100))

    _, avg = compute_correlation_matrix(df, window=60)
    assert avg > 0.5


def test_correlation_insufficient_data():
    """< 2 positions or < 60 days -> empty result."""
    df = pd.DataFrame({"A": np.zeros(30)}, index=pd.bdate_range("2024-01-01", periods=30))
    corr, avg = compute_correlation_matrix(df, window=60)
    assert corr.empty


# -- 6.4 VaR Breach Check --


def test_var_historical_simulation():
    """VaR 95% and 99% computed via historical simulation."""
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.015, 500))
    var = compute_var_historical(returns)

    assert var["var_95"] is not None
    assert var["var_99"] is not None
    assert var["cvar_99"] is not None
    assert var["var_99"] >= var["var_95"]  # 99% VaR >= 95% VaR
    assert var["cvar_99"] >= var["var_99"]  # CVaR >= VaR


def test_var_insufficient_data():
    """< 50 returns -> None (no crash)."""
    returns = pd.Series([0.01, -0.01, 0.005])
    var = compute_var_historical(returns)
    assert var["var_95"] is None


# -- 6.5 Drawdown Cascade --


def test_dd_0_to_5_no_action():
    """0-5% DD -> normal operations."""
    level, event = check_drawdown_cascade(current_equity=97_000, peak_equity=100_000)
    assert level is None
    assert event is None


def test_dd_5_to_10_sizing_override():
    """5-10% DD -> RISK_SIZING_OVERRIDE."""
    level, event = check_drawdown_cascade(current_equity=93_000, peak_equity=100_000)
    assert level == "sizing_override"
    assert event == "RISK_SIZING_OVERRIDE"


def test_dd_10_to_15_entry_halt():
    """10-15% DD -> RISK_ENTRY_HALT."""
    level, event = check_drawdown_cascade(current_equity=88_000, peak_equity=100_000)
    assert level == "entry_halt"
    assert event == "RISK_ENTRY_HALT"


def test_dd_15_to_20_liquidation():
    """15-20% DD -> RISK_LIQUIDATION."""
    level, event = check_drawdown_cascade(current_equity=83_000, peak_equity=100_000)
    assert level == "liquidation"
    assert event == "RISK_LIQUIDATION"


def test_dd_over_20_emergency():
    """>20% DD -> RISK_EMERGENCY (kill switch)."""
    level, event = check_drawdown_cascade(current_equity=78_000, peak_equity=100_000)
    assert level == "emergency"
    assert event == "RISK_EMERGENCY"
