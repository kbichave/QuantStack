# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for QuantStack MCP risk tools.

All tools under test are pure computation — no DB, no context injection needed.
They return ``{"error": str(e)}`` on failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantstack.mcp.tools.qc_risk import (
    check_risk_limits,
    compute_max_drawdown,
    compute_position_size,
    compute_var,
    stress_test_portfolio,
)
from tests.quantstack.mcp.conftest import _fn


# ---------------------------------------------------------------------------
# compute_position_size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_position_size_happy_path():
    result = await _fn(compute_position_size)(
        equity=100_000, entry_price=100.0, stop_loss_price=95.0
    )
    assert "error" not in result

    assert "position" in result
    assert "shares" in result["position"]
    assert result["position"]["shares"] > 0

    assert "risk" in result
    assert "risk_amount" in result["risk"]
    assert result["risk"]["risk_per_share"] == 5.0

    assert "trade_details" in result
    assert result["trade_details"]["entry_price"] == 100.0
    assert result["trade_details"]["stop_loss"] == 95.0


@pytest.mark.asyncio
async def test_position_size_risk_amount():
    """Risk amount should be ~1% of equity (default risk_per_trade_pct)."""
    result = await _fn(compute_position_size)(
        equity=100_000, entry_price=100.0, stop_loss_price=95.0
    )
    assert "error" not in result
    # 1% of 100k = 1000; risk_amount should be close (alignment=1.0 -> multiplier=1.0)
    assert result["risk"]["risk_amount"] == pytest.approx(1000.0, rel=0.01)


@pytest.mark.asyncio
async def test_position_size_zero_risk_per_share():
    """When stop_loss == entry_price, risk per share is zero -> 0 shares."""
    result = await _fn(compute_position_size)(
        equity=100_000, entry_price=100.0, stop_loss_price=100.0
    )
    assert "error" not in result
    assert result["position"]["shares"] == 0


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_drawdown_trending_up():
    """Monotonically increasing equity has zero drawdown."""
    result = await _fn(compute_max_drawdown)(
        equity_curve=[100, 110, 120, 130, 140]
    )
    assert "error" not in result
    assert result["max_drawdown_pct"] == 0.0
    assert result["is_in_drawdown"] == False  # noqa: E712 — numpy bool


@pytest.mark.asyncio
async def test_max_drawdown_with_recovery():
    """Peak at 110, trough at 80 -> drawdown ~ -27.27%, then recovery."""
    curve = [100, 110, 90, 80, 100, 120]
    result = await _fn(compute_max_drawdown)(equity_curve=curve)
    assert "error" not in result

    assert "max_drawdown_pct" in result
    assert "peak_idx" in result
    assert "trough_value" in result
    assert "recovery_idx" in result

    # Peak is 110 (idx 1), trough is 80 (idx 3) -> dd = (80-110)/110 = -27.27%
    assert result["max_drawdown_pct"] == pytest.approx(-27.27, abs=0.1)
    assert result["peak_idx"] == 1
    assert result["trough_value"] == 80.0
    assert result["recovery_idx"] is not None
    # Curve recovers above 110 at idx 5 (value 120)
    assert result["recovery_idx"] == 5


@pytest.mark.asyncio
async def test_max_drawdown_no_recovery():
    """Peak then decline with no recovery -> recovery_idx is None."""
    curve = [100, 120, 100, 90, 85]
    result = await _fn(compute_max_drawdown)(equity_curve=curve)
    assert "error" not in result
    assert result["recovery_idx"] is None
    assert result["is_in_drawdown"] == True  # noqa: E712 — numpy bool


# ---------------------------------------------------------------------------
# compute_var
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_var_historical():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.01, 100).tolist()

    result = await _fn(compute_var)(returns=returns, method="historical")
    assert "error" not in result
    assert result["method"] == "historical"
    assert "95" in result["var"]
    assert "99" in result["var"]
    assert "95" in result["cvar"]
    assert "99" in result["cvar"]
    # CVaR should be >= VaR (both are positive loss numbers)
    assert result["cvar"]["95"] >= result["var"]["95"]


@pytest.mark.asyncio
async def test_var_parametric():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.01, 100).tolist()

    result = await _fn(compute_var)(returns=returns, method="parametric")
    assert "error" not in result
    assert result["method"] == "parametric"
    assert "95" in result["var"]
    assert "statistics" in result
    assert "volatility" in result["statistics"]


@pytest.mark.asyncio
async def test_var_insufficient_returns():
    """Fewer than 30 returns should produce an error."""
    result = await _fn(compute_var)(returns=[0.01] * 20, method="historical")
    assert "error" in result
    assert "30" in result["error"]


# ---------------------------------------------------------------------------
# stress_test_portfolio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stress_test_single_call():
    """stress_test_portfolio currently fails due to OptionsPosition schema mismatch.

    Tests stress test with a single call option position across all scenarios.
    """
    positions = [
        {
            "symbol": "SPY",
            "option_type": "call",
            "strike": 450,
            "expiry_days": 30,
            "quantity": 10,
            "current_price": 8.0,
            "underlying_price": 450,
            "iv": 0.20,
        }
    ]
    result = await _fn(stress_test_portfolio)(positions=positions)
    assert "error" not in result
    assert "all_scenarios" in result
    assert result["portfolio_size"] == 1
    assert result["scenarios_tested"] >= 1


@pytest.mark.asyncio
async def test_stress_test_empty_positions():
    result = await _fn(stress_test_portfolio)(positions=[])
    assert "error" in result
    assert result["error"] == "No positions provided"


# ---------------------------------------------------------------------------
# check_risk_limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_limits_normal():
    result = await _fn(check_risk_limits)(
        equity=100_000,
        daily_pnl=-500,
        open_trades=2,
        open_exposure_pct=30.0,
        drawdown_pct=-3.0,
    )
    assert "error" not in result
    assert result["status"] == "NORMAL"
    assert result["can_trade"] is True
    assert result["size_multiplier"] == 1.0
    assert result["breaches"] == []


@pytest.mark.asyncio
async def test_risk_limits_daily_loss_breach():
    """Daily loss exceeding limit -> HALTED, can_trade=False."""
    result = await _fn(check_risk_limits)(
        equity=100_000,
        daily_pnl=-2500,  # -2.5% > default 2%
        open_trades=1,
        open_exposure_pct=10.0,
        drawdown_pct=-1.0,
    )
    assert "error" not in result
    assert result["status"] == "HALTED"
    assert result["can_trade"] is False
    assert "daily_loss" in result["breaches"]


@pytest.mark.asyncio
async def test_risk_limits_drawdown_breach():
    """Drawdown exceeding limit -> HALTED."""
    result = await _fn(check_risk_limits)(
        equity=100_000,
        daily_pnl=-100,
        open_trades=1,
        open_exposure_pct=10.0,
        drawdown_pct=-12.0,  # > default 10%
    )
    assert "error" not in result
    assert result["status"] == "HALTED"
    assert result["can_trade"] is False
    assert "drawdown" in result["breaches"]


@pytest.mark.asyncio
async def test_risk_limits_position_count_breach():
    """Exceeding max concurrent trades -> RESTRICTED."""
    result = await _fn(check_risk_limits)(
        equity=100_000,
        daily_pnl=-100,
        open_trades=6,  # > default 5
        open_exposure_pct=30.0,
        drawdown_pct=-1.0,
    )
    assert "error" not in result
    assert result["status"] == "RESTRICTED"
    assert result["can_trade"] is False
    assert "position_count" in result["breaches"]


@pytest.mark.asyncio
async def test_risk_limits_multiple_breaches():
    """Daily loss + drawdown + position count all breached simultaneously."""
    result = await _fn(check_risk_limits)(
        equity=100_000,
        daily_pnl=-3000,  # -3% breach
        open_trades=7,  # breach
        open_exposure_pct=90.0,  # breach
        drawdown_pct=-15.0,  # breach
    )
    assert "error" not in result
    assert result["status"] == "HALTED"
    assert result["can_trade"] is False
    assert len(result["breaches"]) >= 3
    assert "daily_loss" in result["breaches"]
    assert "drawdown" in result["breaches"]
    assert "position_count" in result["breaches"]
    assert "exposure" in result["breaches"]
