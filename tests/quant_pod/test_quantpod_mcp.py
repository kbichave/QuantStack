# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the QuantPod MCP server tools.

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

from quant_pod.context import create_trading_context
from quant_pod.execution.portfolio_state import Position


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
    """Inject the test context into the MCP server module."""
    import quant_pod.mcp.server as srv

    original = srv._ctx
    srv._ctx = ctx
    yield ctx
    srv._ctx = original


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
        from quant_pod.mcp.server import get_portfolio_state

        result = await _get_fn(get_portfolio_state)()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_returns_snapshot(self, _inject_ctx):
        from quant_pod.mcp.server import get_portfolio_state

        result = await _get_fn(get_portfolio_state)()
        snapshot = result["snapshot"]
        assert "cash" in snapshot
        assert "total_equity" in snapshot
        assert snapshot["cash"] == 100_000.0

    @pytest.mark.asyncio
    async def test_returns_empty_positions(self, _inject_ctx):
        from quant_pod.mcp.server import get_portfolio_state

        result = await _get_fn(get_portfolio_state)()
        assert result["positions"] == []

    @pytest.mark.asyncio
    async def test_returns_positions_after_upsert(self, _inject_ctx, ctx):
        from quant_pod.mcp.server import get_portfolio_state

        ctx.portfolio.upsert_position(
            Position(symbol="SPY", quantity=100, avg_cost=450.0, current_price=455.0)
        )
        result = await _get_fn(get_portfolio_state)()
        assert len(result["positions"]) == 1
        assert result["positions"][0]["symbol"] == "SPY"

    @pytest.mark.asyncio
    async def test_returns_context_string(self, _inject_ctx):
        from quant_pod.mcp.server import get_portfolio_state

        result = await _get_fn(get_portfolio_state)()
        assert "context_string" in result
        assert "PORTFOLIO STATE" in result["context_string"]


# ---------------------------------------------------------------------------
# get_system_status
# ---------------------------------------------------------------------------


class TestGetSystemStatus:
    @pytest.mark.asyncio
    async def test_returns_success(self, _inject_ctx):
        from quant_pod.mcp.server import get_system_status

        result = await _get_fn(get_system_status)()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_kill_switch_not_active(self, _inject_ctx):
        from quant_pod.mcp.server import get_system_status

        result = await _get_fn(get_system_status)()
        assert result["kill_switch_active"] is False

    @pytest.mark.asyncio
    async def test_risk_not_halted(self, _inject_ctx):
        from quant_pod.mcp.server import get_system_status

        result = await _get_fn(get_system_status)()
        assert result["risk_halted"] is False

    @pytest.mark.asyncio
    async def test_broker_mode_is_paper(self, _inject_ctx):
        from quant_pod.mcp.server import get_system_status

        result = await _get_fn(get_system_status)()
        assert result["broker_mode"] == "paper"

    @pytest.mark.asyncio
    async def test_has_session_id(self, _inject_ctx):
        from quant_pod.mcp.server import get_system_status

        result = await _get_fn(get_system_status)()
        assert result["session_id"] != ""


# ---------------------------------------------------------------------------
# get_recent_decisions
# ---------------------------------------------------------------------------


class TestGetRecentDecisions:
    @pytest.mark.asyncio
    async def test_returns_empty_initially(self, _inject_ctx):
        from quant_pod.mcp.server import get_recent_decisions

        result = await _get_fn(get_recent_decisions)()
        assert result["success"] is True
        assert result["decisions"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_recorded_events(self, _inject_ctx, ctx):
        from quant_pod.audit.models import DecisionEvent
        from quant_pod.mcp.server import get_recent_decisions

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
        from quant_pod.mcp.server import get_regime

        mock_result = {
            "success": True,
            "symbol": "SPY",
            "trend_regime": "trending_up",
            "volatility_regime": "normal",
            "confidence": 0.8,
            "adx": 32.5,
        }

        with patch(
            "quant_pod.agents.regime_detector.RegimeDetectorAgent.detect_regime",
            return_value=mock_result,
        ):
            result = await _get_fn(get_regime)("SPY")
            assert result["success"] is True
            assert result["trend_regime"] == "trending_up"

    @pytest.mark.asyncio
    async def test_returns_error_on_failure(self, _inject_ctx):
        from quant_pod.mcp.server import get_regime

        with patch(
            "quant_pod.agents.regime_detector.RegimeDetectorAgent.detect_regime",
            side_effect=ValueError("No data available"),
        ):
            result = await _get_fn(get_regime)("INVALID")
            assert result["success"] is False
            assert "error" in result


# ---------------------------------------------------------------------------
# run_analysis (mocked — no actual crew execution)
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    @pytest.mark.asyncio
    async def test_returns_daily_brief_with_mock(self, _inject_ctx):
        from quant_pod.crews.schemas import DailyBrief
        from quant_pod.mcp.server import run_analysis

        mock_brief = DailyBrief(
            date=date.today(),
            market_overview="SPY in strong uptrend",
            market_bias="bullish",
            market_conviction=0.75,
            risk_environment="normal",
            top_opportunities=["SPY"],
            key_risks=["FOMC meeting"],
            pods_reporting=5,
            total_analyses=10,
            overall_confidence=0.7,
        )

        mock_result = MagicMock()
        mock_result.pydantic = mock_brief

        mock_regime = {
            "success": True,
            "trend_regime": "trending_up",
            "volatility_regime": "normal",
            "confidence": 0.8,
        }

        with patch(
            "quant_pod.agents.regime_detector.RegimeDetectorAgent.detect_regime",
            return_value=mock_regime,
        ), patch(
            "quant_pod.crews.trading_crew.run_analysis_only",
            return_value=mock_result,
        ):
            result = await _get_fn(run_analysis)("SPY")
            assert result["success"] is True
            assert result["daily_brief"] is not None
            assert result["daily_brief"]["market_bias"] == "bullish"
            assert result["elapsed_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_uses_provided_regime(self, _inject_ctx):
        from quant_pod.mcp.server import run_analysis

        mock_result = MagicMock()
        mock_result.pydantic = None
        mock_result.json_dict = {"market_overview": "test"}

        custom_regime = {"trend": "ranging", "volatility": "high", "confidence": 0.6}

        with patch(
            "quant_pod.crews.trading_crew.run_analysis_only",
            return_value=mock_result,
        ):
            result = await _get_fn(run_analysis)("SPY", regime=custom_regime)
            assert result["success"] is True
            assert result["regime_used"] == custom_regime

    @pytest.mark.asyncio
    async def test_returns_error_on_crew_failure(self, _inject_ctx):
        from quant_pod.mcp.server import run_analysis

        with patch(
            "quant_pod.agents.regime_detector.RegimeDetectorAgent.detect_regime",
            return_value={"success": True, "trend_regime": "unknown"},
        ), patch(
            "quant_pod.crews.trading_crew.run_analysis_only",
            side_effect=RuntimeError("Crew failed to start"),
        ):
            result = await _get_fn(run_analysis)("SPY")
            assert result["success"] is False
            assert "Crew failed" in result["error"]


# ---------------------------------------------------------------------------
# _require_ctx guard
# ---------------------------------------------------------------------------


class TestRequireCtx:
    def test_raises_when_ctx_is_none(self):
        import quant_pod.mcp.server as srv

        original = srv._ctx
        srv._ctx = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                srv._require_ctx()
        finally:
            srv._ctx = original


# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------


class TestServerDefinition:
    def test_mcp_server_has_tools(self):
        from quant_pod.mcp.server import mcp

        assert mcp is not None
        assert mcp.name == "QuantPod Trading Intelligence"

    def test_main_is_callable(self):
        from quant_pod.mcp.server import main

        assert callable(main)
