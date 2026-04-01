# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for equity alert MCP tools — full CRUD lifecycle.

Tools: create_equity_alert, get_equity_alerts, update_alert_status,
       create_exit_signal, add_alert_update.

All tools use pg_conn() directly, so we need both inject_ctx (for
live_db_or_error guard) and the real pg_conn (for actual writes).
"""

from __future__ import annotations

import pytest

from quantstack.mcp.tools.alerts import (
    add_alert_update,
    create_equity_alert,
    create_exit_signal,
    get_equity_alerts,
    update_alert_status,
)
import quantstack.mcp._state as _mcp_state
from tests.quant_pod.mcp.conftest import _fn, assert_error_response, assert_standard_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_alerts():
    """Clean alert tables before and after each test."""
    from quantstack.db import pg_conn
    def _clean():
        with pg_conn() as conn:
            conn.execute("DELETE FROM alert_exit_signals")
            conn.execute("DELETE FROM alert_updates")
            conn.execute("DELETE FROM equity_alerts")
    _clean()
    yield
    _clean()


# ---------------------------------------------------------------------------
# create_equity_alert
# ---------------------------------------------------------------------------


class TestCreateEquityAlert:
    @pytest.mark.asyncio
    async def test_create_alert_happy_path(self, inject_ctx, _clean_alerts):
        result = await _fn(create_equity_alert)(
            symbol="AAPL",
            action="buy",
            time_horizon="investment",
            thesis="Strong FCF growth, P/E below 5yr average",
            confidence=0.85,
            current_price=175.0,
            suggested_entry=170.0,
            stop_price=155.0,
            target_price=210.0,
        )
        assert_standard_response(result)
        assert result["success"] is True
        assert result["alert_id"] > 0
        assert result["symbol"] == "AAPL"
        assert result["deduplicated"] is False

    @pytest.mark.asyncio
    async def test_deduplication(self, inject_ctx, _clean_alerts):
        """Second alert for same symbol+horizon within 7 days returns existing."""
        first = await _fn(create_equity_alert)(
            symbol="MSFT", action="buy", time_horizon="swing",
            thesis="First alert",
        )
        assert first["success"] is True
        first_id = first["alert_id"]

        second = await _fn(create_equity_alert)(
            symbol="MSFT", action="buy", time_horizon="swing",
            thesis="Second alert — should be deduplicated",
        )
        assert second["success"] is True
        assert second["deduplicated"] is True
        assert second["alert_id"] == first_id

    @pytest.mark.asyncio
    async def test_invalid_action(self, inject_ctx, _clean_alerts):
        result = await _fn(create_equity_alert)(
            symbol="AAPL", action="hold", time_horizon="investment",
            thesis="Invalid",
        )
        assert_error_response(result)
        assert "Invalid action" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_time_horizon(self, inject_ctx, _clean_alerts):
        result = await _fn(create_equity_alert)(
            symbol="AAPL", action="buy", time_horizon="scalp",
            thesis="Invalid horizon",
        )
        assert_error_response(result)
        assert "Invalid time_horizon" in result["error"]

    @pytest.mark.asyncio
    async def test_risk_reward_computed(self, inject_ctx, _clean_alerts):
        """Risk/reward ratio should be computed when entry/stop/target provided."""
        result = await _fn(create_equity_alert)(
            symbol="GOOG", action="buy", time_horizon="swing",
            thesis="Test RR", suggested_entry=100.0,
            stop_price=90.0, target_price=120.0,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_missing_context_returns_error(self, _clean_alerts):
        """With no TradingContext, should return error dict."""
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await _fn(create_equity_alert)(
                symbol="AAPL", action="buy", time_horizon="investment",
                thesis="no ctx",
            )
            assert_error_response(result)
        finally:
            _mcp_state._ctx = original


# ---------------------------------------------------------------------------
# get_equity_alerts
# ---------------------------------------------------------------------------


class TestGetEquityAlerts:
    @pytest.mark.asyncio
    async def test_empty_initially(self, inject_ctx, _clean_alerts):
        result = await _fn(get_equity_alerts)()
        assert_standard_response(result)
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_returns_created_alert(self, inject_ctx, _clean_alerts):
        await _fn(create_equity_alert)(
            symbol="NVDA", action="buy", time_horizon="position",
            thesis="AI demand cycle",
        )
        result = await _fn(get_equity_alerts)(symbol="NVDA")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["alerts"][0]["symbol"] == "NVDA"

    @pytest.mark.asyncio
    async def test_filter_by_status(self, inject_ctx, _clean_alerts):
        create_result = await _fn(create_equity_alert)(
            symbol="AMZN", action="buy", time_horizon="swing",
            thesis="Breakout",
        )
        alert_id = create_result["alert_id"]

        # Default status is 'pending'
        result = await _fn(get_equity_alerts)(status="pending")
        assert result["count"] >= 1

        # No watching alerts yet
        result = await _fn(get_equity_alerts)(status="watching")
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_include_updates(self, inject_ctx, _clean_alerts):
        create_result = await _fn(create_equity_alert)(
            symbol="META", action="buy", time_horizon="investment",
            thesis="Reels monetization",
        )
        alert_id = create_result["alert_id"]

        result = await _fn(get_equity_alerts)(
            alert_id=alert_id, include_updates=True,
        )
        assert result["success"] is True
        assert result["count"] == 1
        # Should have the initial thesis_check update
        assert len(result["alerts"][0].get("updates", [])) >= 1


# ---------------------------------------------------------------------------
# update_alert_status
# ---------------------------------------------------------------------------


class TestUpdateAlertStatus:
    @pytest.mark.asyncio
    async def test_pending_to_watching(self, inject_ctx, _clean_alerts):
        create_result = await _fn(create_equity_alert)(
            symbol="TSLA", action="buy", time_horizon="swing",
            thesis="Volume breakout",
        )
        alert_id = create_result["alert_id"]

        result = await _fn(update_alert_status)(
            alert_id=alert_id, status="watching",
            status_reason="Added to watchlist",
        )
        assert_standard_response(result)
        assert result["new_status"] == "watching"

    @pytest.mark.asyncio
    async def test_invalid_status(self, inject_ctx, _clean_alerts):
        result = await _fn(update_alert_status)(
            alert_id=1, status="invalid_status",
        )
        assert_error_response(result)
        assert "Invalid status" in result["error"]


# ---------------------------------------------------------------------------
# create_exit_signal
# ---------------------------------------------------------------------------


class TestCreateExitSignal:
    @pytest.mark.asyncio
    async def test_create_exit_signal(self, inject_ctx, _clean_alerts):
        create_result = await _fn(create_equity_alert)(
            symbol="AMD", action="buy", time_horizon="swing",
            thesis="Semiconductor cycle", current_price=150.0,
        )
        alert_id = create_result["alert_id"]

        result = await _fn(create_exit_signal)(
            alert_id=alert_id,
            signal_type="stop_loss_hit",
            severity="critical",
            headline="AMD stop hit at $135 (-10%)",
            exit_price=135.0,
            pnl_pct=-10.0,
        )
        assert_standard_response(result)
        assert result["exit_signal_id"] > 0
        assert result["auto_closed"] is False

    @pytest.mark.asyncio
    async def test_auto_close_on_severity(self, inject_ctx, _clean_alerts):
        create_result = await _fn(create_equity_alert)(
            symbol="INTC", action="buy", time_horizon="investment",
            thesis="Turnaround play",
        )
        alert_id = create_result["alert_id"]

        result = await _fn(create_exit_signal)(
            alert_id=alert_id,
            signal_type="thesis_invalidated",
            severity="auto_close",
            headline="INTC foundry plan cancelled",
        )
        assert result["success"] is True
        assert result["auto_closed"] is True

        # Verify parent alert is now expired
        alerts = await _fn(get_equity_alerts)(alert_id=alert_id)
        assert alerts["alerts"][0]["status"] == "expired"

    @pytest.mark.asyncio
    async def test_invalid_signal_type(self, inject_ctx, _clean_alerts):
        result = await _fn(create_exit_signal)(
            alert_id=1, signal_type="bad_type",
            severity="info", headline="test",
        )
        assert_error_response(result)
        assert "Invalid signal_type" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_severity(self, inject_ctx, _clean_alerts):
        result = await _fn(create_exit_signal)(
            alert_id=1, signal_type="stop_loss_hit",
            severity="extreme", headline="test",
        )
        assert_error_response(result)
        assert "Invalid severity" in result["error"]


# ---------------------------------------------------------------------------
# add_alert_update
# ---------------------------------------------------------------------------


class TestAddAlertUpdate:
    @pytest.mark.asyncio
    async def test_add_thesis_check(self, inject_ctx, _clean_alerts):
        create_result = await _fn(create_equity_alert)(
            symbol="MSFT", action="buy", time_horizon="investment",
            thesis="Cloud growth",
        )
        alert_id = create_result["alert_id"]

        result = await _fn(add_alert_update)(
            alert_id=alert_id,
            update_type="thesis_check",
            commentary="Azure growth accelerating, thesis intact",
            thesis_status="strengthening",
        )
        assert_standard_response(result)
        assert result["thesis_status"] == "strengthening"

    @pytest.mark.asyncio
    async def test_broken_thesis_creates_exit_signal(self, inject_ctx, _clean_alerts):
        create_result = await _fn(create_equity_alert)(
            symbol="COIN", action="buy", time_horizon="swing",
            thesis="Crypto bull cycle",
        )
        alert_id = create_result["alert_id"]

        result = await _fn(add_alert_update)(
            alert_id=alert_id,
            update_type="fundamental_update",
            commentary="SEC enforcement action, revenue collapse",
            thesis_status="broken",
        )
        assert result["success"] is True
        assert result["thesis_status"] == "broken"

        # Verify exit signal was auto-created
        alerts = await _fn(get_equity_alerts)(
            alert_id=alert_id, include_exit_signals=True,
        )
        exit_signals = alerts["alerts"][0].get("exit_signals", [])
        assert len(exit_signals) >= 1
        assert exit_signals[0]["signal_type"] == "thesis_invalidated"

    @pytest.mark.asyncio
    async def test_invalid_update_type(self, inject_ctx, _clean_alerts):
        result = await _fn(add_alert_update)(
            alert_id=1, update_type="bad_type",
            commentary="test",
        )
        assert_error_response(result)
        assert "Invalid update_type" in result["error"]
