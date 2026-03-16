# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Phase 3 execution MCP tools:
  - execute_trade — happy path, risk rejection, paper/live mode
  - close_position
  - get_fills, get_risk_metrics, get_audit_trail

All tests use in-memory TradingContext with pre-populated positions.
"""

from __future__ import annotations

import os
import uuid

import pytest

from quant_pod.context import create_trading_context
from quant_pod.execution.portfolio_state import Position


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
    import quant_pod.mcp.server as srv
    original = srv._ctx
    srv._ctx = ctx
    yield ctx
    srv._ctx = original


@pytest.fixture
def ctx_with_position(ctx):
    """
    Inject a small position so execute_trade can get a current_price.

    The position is deliberately small (5 shares @ $100 = $500 notional)
    to stay well within risk limits (10% of $100k = $10,000 max).
    """
    ctx.portfolio.upsert_position(
        Position(symbol="SPY", quantity=5, avg_cost=100.0, current_price=100.0, side="long")
    )
    return ctx


def _fn(tool_obj):
    return tool_obj.fn if hasattr(tool_obj, "fn") else tool_obj


# ---------------------------------------------------------------------------
# execute_trade — happy path
# ---------------------------------------------------------------------------


class TestExecuteTradeHappyPath:
    @pytest.mark.asyncio
    async def test_buy_fills_successfully(self, _inject_ctx, ctx_with_position):
        from quant_pod.mcp.server import execute_trade

        result = await _fn(execute_trade)(
            symbol="SPY",
            action="buy",
            reasoning="Strong uptrend, RSI pullback to 35",
            confidence=0.75,
            quantity=10,
        )
        assert result["success"] is True
        assert result["risk_approved"] is True
        assert result["filled_quantity"] > 0
        assert result["fill_price"] > 0
        assert result["slippage_bps"] >= 0
        assert result["broker_mode"] == "paper"

    @pytest.mark.asyncio
    async def test_sell_fills_successfully(self, _inject_ctx, ctx_with_position):
        from quant_pod.mcp.server import execute_trade

        result = await _fn(execute_trade)(
            symbol="SPY",
            action="sell",
            reasoning="Taking profit at resistance",
            confidence=0.8,
            quantity=10,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_fill_logged_in_audit(self, _inject_ctx, ctx_with_position, ctx):
        from quant_pod.mcp.server import execute_trade, get_audit_trail

        await _fn(execute_trade)(
            symbol="SPY",
            action="buy",
            reasoning="Audit trail test",
            confidence=0.6,
            quantity=5,
        )

        trail = await _fn(get_audit_trail)(symbol="SPY")
        assert trail["success"] is True
        assert trail["total"] >= 1
        # Find the execution event
        exec_events = [e for e in trail["events"] if e["event_type"] == "execution"]
        assert len(exec_events) >= 1
        assert "Audit trail test" in exec_events[0]["output_summary"]

    @pytest.mark.asyncio
    async def test_auto_calculates_quantity(self, _inject_ctx, ctx_with_position):
        from quant_pod.mcp.server import execute_trade

        result = await _fn(execute_trade)(
            symbol="SPY",
            action="buy",
            reasoning="Position sizing test",
            confidence=0.7,
            position_size="quarter",
        )
        assert result["success"] is True
        assert result["filled_quantity"] > 0


# ---------------------------------------------------------------------------
# execute_trade — risk gate rejection
# ---------------------------------------------------------------------------


class TestExecuteTradeRiskRejection:
    @pytest.mark.asyncio
    async def test_rejected_when_no_price(self, _inject_ctx, ctx):
        """No position = no current_price = rejection."""
        from quant_pod.mcp.server import execute_trade

        result = await _fn(execute_trade)(
            symbol="UNKNOWN",
            action="buy",
            reasoning="Test no price",
            confidence=0.5,
            quantity=10,
        )
        assert result["success"] is False
        assert "No current price" in result["error"]

    @pytest.mark.asyncio
    async def test_rejected_when_kill_switch_active(self, _inject_ctx, ctx_with_position, ctx):
        from quant_pod.mcp.server import execute_trade

        ctx.kill_switch.trigger("test halt")
        try:
            result = await _fn(execute_trade)(
                symbol="SPY",
                action="buy",
                reasoning="Kill switch test",
                confidence=0.5,
                quantity=5,
            )
            assert result["success"] is False
            assert "Kill switch" in result["error"] or "ACTIVE" in result["error"]
        finally:
            ctx.kill_switch.reset("test")

    @pytest.mark.asyncio
    async def test_risk_rejection_logs_audit_event(self, _inject_ctx, ctx_with_position, ctx):
        """Buying a huge position should trigger risk gate rejection + audit log."""
        from quant_pod.mcp.server import execute_trade, get_audit_trail

        # Try to buy way too much: 500 shares @ $100 = $50,000
        # Max position is 10% of $100k = $10,000, so this should be rejected or scaled
        result = await _fn(execute_trade)(
            symbol="SPY",
            action="buy",
            reasoning="Over-sized order test",
            confidence=0.5,
            quantity=500,
        )

        # Should either be rejected by risk gate or scaled down
        if not result["success"]:
            assert result["risk_approved"] is False
            assert len(result["risk_violations"]) > 0

            # Check audit trail for the rejection
            trail = await _fn(get_audit_trail)(symbol="SPY")
            rejection_events = [e for e in trail["events"] if e["event_type"] == "risk_rejection"]
            assert len(rejection_events) >= 1


# ---------------------------------------------------------------------------
# execute_trade — paper/live mode
# ---------------------------------------------------------------------------


class TestPaperModeDefault:
    @pytest.mark.asyncio
    async def test_paper_mode_is_default(self, _inject_ctx, ctx_with_position):
        from quant_pod.mcp.server import execute_trade

        # Default paper_mode=True should work
        result = await _fn(execute_trade)(
            symbol="SPY",
            action="buy",
            reasoning="Paper mode default test",
            confidence=0.6,
            quantity=5,
        )
        assert result["success"] is True
        assert result["broker_mode"] == "paper"

    @pytest.mark.asyncio
    async def test_live_mode_rejected_without_env_var(self, _inject_ctx, ctx_with_position):
        from quant_pod.mcp.server import execute_trade

        # Ensure USE_REAL_TRADING is not set
        old_val = os.environ.pop("USE_REAL_TRADING", None)
        try:
            result = await _fn(execute_trade)(
                symbol="SPY",
                action="buy",
                reasoning="Live mode rejection test",
                confidence=0.6,
                quantity=5,
                paper_mode=False,
            )
            assert result["success"] is False
            assert "USE_REAL_TRADING" in result["error"]
        finally:
            if old_val is not None:
                os.environ["USE_REAL_TRADING"] = old_val


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_existing_position(self, _inject_ctx, ctx_with_position):
        from quant_pod.mcp.server import close_position

        result = await _fn(close_position)(
            symbol="SPY",
            reasoning="Taking profit",
        )
        assert result["success"] is True
        assert result["filled_quantity"] > 0

    @pytest.mark.asyncio
    async def test_close_nonexistent_position(self, _inject_ctx, ctx):
        from quant_pod.mcp.server import close_position

        result = await _fn(close_position)(
            symbol="NOPOS",
            reasoning="Test",
        )
        assert result["success"] is False
        assert "No open position" in result["error"]


# ---------------------------------------------------------------------------
# get_fills
# ---------------------------------------------------------------------------


class TestGetFills:
    @pytest.mark.asyncio
    async def test_empty_initially(self, _inject_ctx):
        from quant_pod.mcp.server import get_fills

        result = await _fn(get_fills)()
        assert result["success"] is True
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_fills_after_trade(self, _inject_ctx, ctx_with_position):
        from quant_pod.mcp.server import execute_trade, get_fills

        await _fn(execute_trade)(
            symbol="SPY", action="buy", reasoning="fill test",
            confidence=0.7, quantity=5,
        )

        result = await _fn(get_fills)(symbol="SPY")
        assert result["success"] is True
        assert result["total"] >= 1


# ---------------------------------------------------------------------------
# get_risk_metrics
# ---------------------------------------------------------------------------


class TestGetRiskMetrics:
    @pytest.mark.asyncio
    async def test_returns_all_fields(self, _inject_ctx):
        from quant_pod.mcp.server import get_risk_metrics

        result = await _fn(get_risk_metrics)()
        assert result["success"] is True
        assert "cash" in result
        assert "total_equity" in result
        assert "daily_pnl" in result
        assert "daily_loss_limit_pct" in result
        assert "daily_headroom_pct" in result
        assert "gross_exposure" in result
        assert "max_gross_exposure_pct" in result
        assert "kill_switch_active" in result
        assert result["kill_switch_active"] is False

    @pytest.mark.asyncio
    async def test_shows_position_exposure(self, _inject_ctx, ctx_with_position):
        from quant_pod.mcp.server import get_risk_metrics

        result = await _fn(get_risk_metrics)()
        assert result["position_count"] == 1
        assert result["gross_exposure"] > 0


# ---------------------------------------------------------------------------
# get_audit_trail
# ---------------------------------------------------------------------------


class TestGetAuditTrail:
    @pytest.mark.asyncio
    async def test_empty_initially(self, _inject_ctx):
        from quant_pod.mcp.server import get_audit_trail

        result = await _fn(get_audit_trail)()
        assert result["success"] is True
        assert result["total"] == 0
