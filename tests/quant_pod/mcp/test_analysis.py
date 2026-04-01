# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the QuantStack MCP server tools.

Tests use an in-memory TradingContext — no file I/O, no LLM calls.
The run_analysis tool is tested with mocked crew execution.

FastMCP wraps tool functions as FunctionTool objects, so we call the
underlying function via the `.fn` attribute.
"""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from quantstack.context import create_trading_context
from quantstack.execution.portfolio_state import Position
from quantstack.audit.models import DecisionEvent
from quantstack.mcp.tools.analysis import get_portfolio_state, get_recent_decisions, get_regime, get_system_status
from quantstack.mcp.server import main
from quantstack.mcp._app import mcp
import quantstack.mcp._state as _mcp_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    """In-memory TradingContext for isolated testing."""
    context = create_trading_context(
        db_path=":memory:",
        initial_cash=100_000.0,
        session_id=str(uuid.uuid4()),
    )
    yield context
    context.db.close()


@pytest.fixture
def _inject_ctx(ctx):
    """Inject the test context into the MCP server module with a clean DB state."""
    original = _mcp_state._ctx
    _mcp_state._ctx = ctx
    # Clear tables that have pre-existing data from production DB
    ctx.portfolio.reset()
    ctx.db.execute("DELETE FROM fills")
    ctx.db.execute("DELETE FROM decision_events")
    ctx.db.execute("DELETE FROM strategies")
    ctx.db.execute("DELETE FROM regime_strategy_matrix")
    ctx.db.execute("DELETE FROM closed_trades")
    yield ctx
    _mcp_state._ctx = original
    ctx.db.execute("ROLLBACK")


# ---------------------------------------------------------------------------
# Helper to get the raw async function from a FunctionTool
# ---------------------------------------------------------------------------


def _get_fn(tool_obj):
    """Extract the underlying async function from a FastMCP FunctionTool."""
    if hasattr(tool_obj, "fn"):
        return tool_obj.fn
    return tool_obj


# ---------------------------------------------------------------------------
# get_portfolio_state
# ---------------------------------------------------------------------------


class TestGetPortfolioState:
    @pytest.mark.asyncio
    async def test_returns_success(self, _inject_ctx):
        result = await _get_fn(get_portfolio_state)()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_returns_snapshot(self, _inject_ctx):
        result = await _get_fn(get_portfolio_state)()
        snapshot = result["snapshot"]
        assert "cash" in snapshot
        assert "total_equity" in snapshot
        assert snapshot["cash"] == 100_000.0

    @pytest.mark.asyncio
    async def test_returns_empty_positions(self, _inject_ctx):
        result = await _get_fn(get_portfolio_state)()
        assert result["positions"] == []

    @pytest.mark.asyncio
    async def test_returns_positions_after_upsert(self, _inject_ctx, ctx):
        ctx.portfolio.upsert_position(
            Position(symbol="SPY", quantity=100, avg_cost=450.0, current_price=455.0)
        )
        result = await _get_fn(get_portfolio_state)()
        assert len(result["positions"]) == 1
        assert result["positions"][0]["symbol"] == "SPY"

    @pytest.mark.asyncio
    async def test_returns_context_string(self, _inject_ctx):
        result = await _get_fn(get_portfolio_state)()
        assert "context_string" in result
        assert "PORTFOLIO STATE" in result["context_string"]


# ---------------------------------------------------------------------------
# get_system_status
# ---------------------------------------------------------------------------


class TestGetSystemStatus:
    @pytest.mark.asyncio
    async def test_returns_success(self, _inject_ctx):
        result = await _get_fn(get_system_status)()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_kill_switch_not_active(self, _inject_ctx):
        result = await _get_fn(get_system_status)()
        assert result["kill_switch_active"] is False

    @pytest.mark.asyncio
    async def test_risk_not_halted(self, _inject_ctx):
        result = await _get_fn(get_system_status)()
        assert result["risk_halted"] is False

    @pytest.mark.asyncio
    async def test_broker_mode_is_paper(self, _inject_ctx):
        result = await _get_fn(get_system_status)()
        assert result["broker_mode"] == "paper"

    @pytest.mark.asyncio
    async def test_has_session_id(self, _inject_ctx):
        result = await _get_fn(get_system_status)()
        assert result["session_id"] != ""


# ---------------------------------------------------------------------------
# get_recent_decisions
# ---------------------------------------------------------------------------


class TestGetRecentDecisions:
    @pytest.mark.asyncio
    async def test_returns_empty_initially(self, _inject_ctx):
        result = await _get_fn(get_recent_decisions)()
        assert result["success"] is True
        assert result["decisions"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_recorded_events(self, _inject_ctx, ctx):
        event = DecisionEvent(
            event_id=str(uuid.uuid4()),
            session_id=ctx.session_id,
            event_type="ic_analysis",
            agent_name="TrendIC",
            agent_role="ic",
            symbol="SPY",
            action="analyze",
            confidence=0.72,
            output_summary="Strong uptrend detected",
        )
        ctx.audit.record(event)

        result = await _get_fn(get_recent_decisions)(symbol="SPY")
        assert result["success"] is True
        assert result["total"] == 1
        assert result["decisions"][0]["agent_name"] == "TrendIC"


# ---------------------------------------------------------------------------
# get_regime (mocked — no real market data)
# ---------------------------------------------------------------------------


class TestGetRegime:
    @pytest.mark.asyncio
    async def test_returns_regime_with_mock_data(self, _inject_ctx):
        mock_result = {
            "success": True,
            "symbol": "SPY",
            "trend_regime": "trending_up",
            "volatility_regime": "normal",
            "confidence": 0.8,
            "adx": 32.5,
        }

        with patch(
            "quantstack.agents.regime_detector.RegimeDetectorAgent.detect_regime",
            return_value=mock_result,
        ):
            result = await _get_fn(get_regime)("SPY")
            assert result["success"] is True
            assert result["trend_regime"] == "trending_up"

    @pytest.mark.asyncio
    async def test_returns_error_on_failure(self, _inject_ctx):
        with patch(
            "quantstack.agents.regime_detector.RegimeDetectorAgent.detect_regime",
            side_effect=ValueError("No data available"),
        ):
            result = await _get_fn(get_regime)("INVALID")
            assert result["success"] is False
            assert "error" in result


# ---------------------------------------------------------------------------
# _require_ctx guard
# ---------------------------------------------------------------------------


class TestRequireCtx:
    def test_raises_when_ctx_is_none(self):
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                _mcp_state.require_ctx()
        finally:
            _mcp_state._ctx = original


# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------


class TestServerDefinition:
    def test_mcp_server_has_tools(self):
        assert mcp is not None
        assert mcp.name == "QuantStack"

    def test_main_is_callable(self):
        assert callable(main)
