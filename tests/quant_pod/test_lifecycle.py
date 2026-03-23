# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Phase 6: strategy lifecycle (promote, retire, performance),
RL status, and regime matrix performance analysis.
"""

from __future__ import annotations

import uuid

import pytest
from quantstack.context import create_trading_context
from quantstack.mcp.server import get_regime_strategies, get_rl_status, get_strategy, get_strategy_performance, promote_strategy, register_strategy, retire_strategy, set_regime_allocation, update_regime_matrix_from_performance, update_strategy
import quantstack.mcp._state as _mcp_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    context = create_trading_context(db_path=":memory:", session_id=str(uuid.uuid4()))
    yield context
    context.db.close()


@pytest.fixture
def _inject_ctx(ctx):
    original = _mcp_state._ctx
    _mcp_state._ctx = ctx
    yield ctx
    _mcp_state._ctx = original


def _fn(tool_obj):
    return tool_obj.fn if hasattr(tool_obj, "fn") else tool_obj


async def _create_strategy(name, status="draft", bt_sharpe=None, wf_oos_sharpe=None):
    """Helper to register and configure a test strategy."""
    r = await _fn(register_strategy)(
        name=name,
        parameters={"rsi_period": 14},
        entry_rules=[
            {"indicator": "rsi", "condition": "below", "value": 30, "direction": "long"}
        ],
        exit_rules=[],
    )
    sid = r["strategy_id"]

    updates = {"strategy_id": sid, "status": status}
    if bt_sharpe is not None:
        updates["backtest_summary"] = {
            "sharpe_ratio": bt_sharpe,
            "max_drawdown": 5.0,
            "symbol": "SPY",
        }
    if wf_oos_sharpe is not None:
        updates["walkforward_summary"] = {"oos_sharpe_mean": wf_oos_sharpe}

    await _fn(update_strategy)(**updates)
    return sid


# ---------------------------------------------------------------------------
# promote_strategy
# ---------------------------------------------------------------------------


class TestPromoteStrategy:
    @pytest.mark.asyncio
    async def test_promote_forward_testing_with_good_backtest(self, _inject_ctx):
        sid = await _create_strategy(
            "promo_good", status="forward_testing", bt_sharpe=1.5, wf_oos_sharpe=0.8
        )
        result = await _fn(promote_strategy)(
            strategy_id=sid, evidence="3 weeks forward test, Sharpe 1.2"
        )
        assert result["success"] is True
        assert result["new_status"] == "live"

        strat = await _fn(get_strategy)(strategy_id=sid)
        assert strat["strategy"]["status"] == "live"

    @pytest.mark.asyncio
    async def test_promote_fails_if_not_forward_testing(self, _inject_ctx):
        sid = await _create_strategy("promo_draft", status="draft", bt_sharpe=1.5)
        result = await _fn(promote_strategy)(strategy_id=sid, evidence="test")
        assert result["success"] is False
        assert "forward_testing" in result["failures"][0]

    @pytest.mark.asyncio
    async def test_promote_fails_if_no_backtest(self, _inject_ctx):
        sid = await _create_strategy("promo_no_bt", status="forward_testing")
        result = await _fn(promote_strategy)(strategy_id=sid, evidence="test")
        assert result["success"] is False
        assert any("backtest" in f.lower() for f in result["failures"])

    @pytest.mark.asyncio
    async def test_promote_fails_if_negative_sharpe(self, _inject_ctx):
        sid = await _create_strategy(
            "promo_neg_sharpe", status="forward_testing", bt_sharpe=-0.5
        )
        result = await _fn(promote_strategy)(strategy_id=sid, evidence="test")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# retire_strategy
# ---------------------------------------------------------------------------


class TestRetireStrategy:
    @pytest.mark.asyncio
    async def test_retire_updates_status(self, _inject_ctx):
        sid = await _create_strategy("retire_test", status="live", bt_sharpe=0.5)
        result = await _fn(retire_strategy)(
            strategy_id=sid, reason="4 weeks underperformance, Sharpe 0.1"
        )
        assert result["success"] is True
        assert result["new_status"] == "retired"

        strat = await _fn(get_strategy)(strategy_id=sid)
        assert strat["strategy"]["status"] == "retired"

    @pytest.mark.asyncio
    async def test_retire_removes_from_matrix(self, _inject_ctx, ctx):
        sid = await _create_strategy("retire_matrix", status="live")

        # Add to regime matrix
        await _fn(set_regime_allocation)(
            regime="trending_up",
            allocations=[{"strategy_id": sid, "allocation_pct": 0.20}],
        )

        # Verify it's in the matrix
        before = await _fn(get_regime_strategies)("trending_up")
        assert before["total"] == 1

        # Retire
        await _fn(retire_strategy)(strategy_id=sid, reason="test retirement")

        # Verify removed from matrix
        after = await _fn(get_regime_strategies)("trending_up")
        assert after["total"] == 0


# ---------------------------------------------------------------------------
# get_strategy_performance
# ---------------------------------------------------------------------------


class TestGetStrategyPerformance:
    @pytest.mark.asyncio
    async def test_no_trades_returns_zero(self, _inject_ctx):
        sid = await _create_strategy("perf_empty", bt_sharpe=1.0)
        result = await _fn(get_strategy_performance)(strategy_id=sid, lookback_days=30)
        assert result["success"] is True
        assert result["total_trades"] == 0

    @pytest.mark.asyncio
    async def test_not_found(self, _inject_ctx):
        result = await _fn(get_strategy_performance)(strategy_id="nonexistent")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# get_rl_status
# ---------------------------------------------------------------------------


class TestGetRlStatus:
    @pytest.mark.asyncio
    async def test_returns_config(self, _inject_ctx):
        result = await _fn(get_rl_status)()
        assert result["success"] is True
        assert "agents" in result
        assert "execution_rl" in result["agents"]
        assert "shadow_mode_enabled" in result


# ---------------------------------------------------------------------------
# update_regime_matrix_from_performance
# ---------------------------------------------------------------------------


class TestUpdateRegimeMatrixFromPerformance:
    @pytest.mark.asyncio
    async def test_no_trades_returns_note(self, _inject_ctx):
        result = await _fn(update_regime_matrix_from_performance)(lookback_days=30)
        assert result["success"] is True
        assert "No closed trades" in result.get("note", "")
