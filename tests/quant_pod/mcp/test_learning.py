# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for learning MCP tools — strategy lifecycle management.

Tools: promote_strategy, retire_strategy, get_strategy_performance,
       validate_strategy, update_regime_matrix_from_performance.

These tools use pg_conn() directly AND call inter-tool functions
(_get_strategy_impl, run_backtest) — key error class #1.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from quantstack.db import pg_conn
from quantstack.mcp.tools.learning import (
    get_strategy_performance,
    promote_strategy,
    retire_strategy,
    update_regime_matrix_from_performance,
    validate_strategy,
)
from quantstack.mcp.tools.strategy import register_strategy
import quantstack.mcp._state as _mcp_state
from tests.quantstack.mcp.conftest import _fn, assert_error_response, assert_standard_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_learning():
    """Clean strategy and trade tables."""
    def _clean():
        with pg_conn() as conn:
            conn.execute("DELETE FROM closed_trades")
            conn.execute("DELETE FROM regime_strategy_matrix")
            conn.execute("DELETE FROM strategies")
    _clean()
    yield
    _clean()


@pytest.fixture
async def forward_testing_strategy(inject_ctx, _clean_learning):
    """Register a strategy in 'forward_testing' status with a positive backtest."""
    result = await _fn(register_strategy)(
        name=f"test_ft_{uuid.uuid4().hex[:6]}",
        description="Test forward_testing strategy",
        parameters={"rsi_period": 14},
        entry_rules=[{"indicator": "rsi", "condition": "crosses_below", "value": 30}],
        exit_rules=[{"indicator": "rsi", "condition": "crosses_above", "value": 70}],
    )
    sid = result["strategy_id"]

    import json
    with pg_conn() as conn:
        conn.execute(
            "UPDATE strategies SET status = 'forward_testing', "
            "backtest_summary = ? WHERE strategy_id = ?",
            [json.dumps({"sharpe_ratio": 1.5, "max_drawdown": -0.12, "symbol": "SPY"}), sid],
        )
    return sid


# ---------------------------------------------------------------------------
# promote_strategy
# ---------------------------------------------------------------------------


class TestPromoteStrategy:
    @pytest.mark.asyncio
    async def test_promote_happy_path(self, forward_testing_strategy):
        sid = forward_testing_strategy
        result = await _fn(promote_strategy)(
            strategy_id=sid, evidence="Walk-forward OOS Sharpe > 1.0",
        )
        assert_standard_response(result)
        assert result["success"] is True
        assert result["new_status"] == "live"

    @pytest.mark.asyncio
    async def test_promote_wrong_status(self, inject_ctx, _clean_learning):
        """Cannot promote a draft strategy."""
        reg = await _fn(register_strategy)(
            name=f"test_draft_{uuid.uuid4().hex[:6]}",
            parameters={}, entry_rules=[], exit_rules=[],
        )
        result = await _fn(promote_strategy)(
            strategy_id=reg["strategy_id"], evidence="Trying anyway",
        )
        assert result["success"] is False
        assert "failures" in result

    @pytest.mark.asyncio
    async def test_promote_nonexistent(self, inject_ctx, _clean_learning):
        result = await _fn(promote_strategy)(
            strategy_id="nonexistent", evidence="test",
        )
        assert_error_response(result)

    @pytest.mark.asyncio
    async def test_promote_no_backtest(self, inject_ctx, _clean_learning):
        """Strategy without backtest should fail promotion."""
        reg = await _fn(register_strategy)(
            name=f"test_nobt_{uuid.uuid4().hex[:6]}",
            parameters={}, entry_rules=[], exit_rules=[],
        )
        import json
        with pg_conn() as conn:
            conn.execute(
                "UPDATE strategies SET status = 'forward_testing' WHERE strategy_id = ?",
                [reg["strategy_id"]],
            )
        result = await _fn(promote_strategy)(
            strategy_id=reg["strategy_id"], evidence="No backtest",
        )
        assert result["success"] is False
        assert any("backtest" in f.lower() for f in result["failures"])


# ---------------------------------------------------------------------------
# retire_strategy
# ---------------------------------------------------------------------------


class TestRetireStrategy:
    @pytest.mark.asyncio
    async def test_retire_removes_from_matrix(self, inject_ctx, _clean_learning):
        reg = await _fn(register_strategy)(
            name=f"test_retire_{uuid.uuid4().hex[:6]}",
            parameters={}, entry_rules=[], exit_rules=[],
        )
        sid = reg["strategy_id"]

        # Add to matrix
        with pg_conn() as conn:
            conn.execute(
                "INSERT INTO regime_strategy_matrix (regime, strategy_id, allocation_pct) "
                "VALUES ('trending_up', ?, 0.3)",
                [sid],
            )

        result = await _fn(retire_strategy)(
            strategy_id=sid, reason="Underperforming in current regime",
        )
        assert_standard_response(result)
        assert result["removed_from_matrix"] is True

        # Verify removed from matrix
        with pg_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM regime_strategy_matrix WHERE strategy_id = ?",
                [sid],
            ).fetchone()
            assert row[0] == 0


# ---------------------------------------------------------------------------
# get_strategy_performance
# ---------------------------------------------------------------------------


class TestGetStrategyPerformance:
    @pytest.mark.asyncio
    async def test_no_trades_returns_note(self, forward_testing_strategy):
        sid = forward_testing_strategy
        result = await _fn(get_strategy_performance)(strategy_id=sid)
        assert_standard_response(result)
        assert result["total_trades"] == 0
        assert "note" in result

    @pytest.mark.asyncio
    async def test_nonexistent_strategy(self, inject_ctx, _clean_learning):
        result = await _fn(get_strategy_performance)(strategy_id="bad_id")
        assert_error_response(result)

    @pytest.mark.asyncio
    async def test_with_closed_trades(self, forward_testing_strategy):
        sid = forward_testing_strategy
        from datetime import datetime
        with pg_conn() as conn:
            for i in range(5):
                pnl = 100.0 if i < 3 else -50.0
                conn.execute(
                    """INSERT INTO closed_trades
                       (id, symbol, side, quantity, entry_price, exit_price,
                        realized_pnl, opened_at, closed_at, holding_days, strategy_id)
                       VALUES (?, 'SPY', 'long', 10, 100, ?, ?, ?, ?, 1, ?)""",
                    [
                        10000 + i, 100 + pnl/10, pnl,
                        datetime.now(), datetime.now(), sid,
                    ],
                )

        result = await _fn(get_strategy_performance)(strategy_id=sid)
        assert result["success"] is True
        assert result["total_trades"] == 5
        assert result["win_rate"] == 60.0
        assert "live_sharpe" in result
        assert "degradation_pct" in result


# ---------------------------------------------------------------------------
# validate_strategy
# ---------------------------------------------------------------------------


class TestValidateStrategy:
    @pytest.mark.asyncio
    async def test_no_backtest_summary(self, inject_ctx, _clean_learning):
        reg = await _fn(register_strategy)(
            name=f"test_val_{uuid.uuid4().hex[:6]}",
            parameters={}, entry_rules=[], exit_rules=[],
        )
        result = await _fn(validate_strategy)(strategy_id=reg["strategy_id"])
        assert result["success"] is True
        assert result["still_valid"] is False
        assert "No backtest_summary" in result["reason"]

    @pytest.mark.asyncio
    async def test_nonexistent_strategy(self, inject_ctx, _clean_learning):
        result = await _fn(validate_strategy)(strategy_id="nonexistent")
        assert_error_response(result)


# ---------------------------------------------------------------------------
# update_regime_matrix_from_performance
# ---------------------------------------------------------------------------


class TestUpdateRegimeMatrixFromPerformance:
    @pytest.mark.asyncio
    async def test_no_trades(self, inject_ctx, _clean_learning):
        result = await _fn(update_regime_matrix_from_performance)()
        assert_standard_response(result)
        assert result["proposals"] == []
        assert "No closed trades" in result["note"]

    @pytest.mark.asyncio
    async def test_missing_context(self, _clean_learning):
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await _fn(update_regime_matrix_from_performance)()
            assert_error_response(result)
        finally:
            _mcp_state._ctx = original
