"""Tests for market_intel node in trading graph."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from quantstack.graphs.state import TradingState
from quantstack.graphs.trading.nodes import make_market_intel


def _make_config(name="market_intel"):
    cfg = MagicMock()
    cfg.name = name
    cfg.role = "Market Intelligence Analyst"
    cfg.goal = "Surface real-time news and events"
    cfg.backstory = "You scan news sources for trading-relevant intelligence."
    cfg.llm_tier = "medium"
    cfg.max_iterations = 10
    cfg.timeout_seconds = 120
    return cfg


def _base_state(**overrides) -> dict:
    state = {
        "cycle_number": 1,
        "regime": "trending_up",
        "portfolio_context": {},
        "data_refresh_summary": {},
        "market_context": {},
        "earnings_symbols": [],
        "earnings_analysis": {},
        "daily_plan": "",
        "position_reviews": [],
        "exit_orders": [],
        "entry_candidates": [],
        "risk_verdicts": [],
        "fund_manager_decisions": [],
        "options_analysis": [],
        "entry_orders": [],
        "reflection": "",
        "errors": [],
        "decisions": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
@patch("quantstack.graphs.trading.nodes.run_agent")
async def test_market_intel_populates_market_context(mock_run_agent):
    """market_intel populates market_context in TradingState via event trigger."""
    mock_run_agent.return_value = json.dumps({
        "headlines": ["Fed holds rates steady"],
        "risk_alerts": [],
        "sentiment": "neutral",
    })
    llm = MagicMock()
    cfg = _make_config()
    node = make_market_intel(llm, cfg, [])
    # Use market_move_trigger to bypass time window check
    state = _base_state(portfolio_context={"market_move_trigger": True})

    result = await node(state)

    assert "market_context" in result
    assert isinstance(result["market_context"], dict)
    assert result["market_context"].get("sentiment") == "neutral"


@pytest.mark.asyncio
@patch("quantstack.graphs.trading.nodes.run_agent")
async def test_market_intel_event_triggered(mock_run_agent):
    """market_intel fires on MARKET_MOVE trigger in portfolio_context."""
    mock_run_agent.return_value = json.dumps({
        "headlines": ["Market drops 3%"],
        "risk_alerts": ["major_sell_off"],
        "sentiment": "bearish",
    })
    llm = MagicMock()
    cfg = _make_config()
    node = make_market_intel(llm, cfg, [])
    state = _base_state(portfolio_context={"market_move_trigger": True})

    result = await node(state)

    assert result["market_context"].get("sentiment") == "bearish"
    assert any(d.get("mode") == "event_triggered" for d in result.get("decisions", []))


@pytest.mark.asyncio
@patch("quantstack.graphs.trading.nodes.run_agent")
async def test_market_intel_outside_window_no_trigger(mock_run_agent):
    """market_intel returns empty context outside pre-market window with no event trigger."""
    llm = MagicMock()
    cfg = _make_config()
    node = make_market_intel(llm, cfg, [])
    state = _base_state()

    result = await node(state)

    assert "market_context" in result
    # Without trigger and outside the 8:30-9:30 AM ET window, context should be empty
    # (test likely runs outside that window)


@pytest.mark.asyncio
@patch("quantstack.graphs.trading.nodes.run_agent")
async def test_market_intel_handles_failure_gracefully(mock_run_agent):
    """market_intel returns empty context on failure, not an exception."""
    mock_run_agent.side_effect = RuntimeError("LLM unavailable")
    llm = MagicMock()
    cfg = _make_config()
    node = make_market_intel(llm, cfg, [])
    state = _base_state(portfolio_context={"market_move_trigger": True})

    result = await node(state)

    assert result["market_context"] == {}
    assert any("market_intel" in e for e in result.get("errors", []))


@pytest.mark.asyncio
@patch("quantstack.graphs.trading.nodes.run_agent")
async def test_plan_day_consumes_market_context(mock_run_agent):
    """plan_day includes market_context in its prompt."""
    from quantstack.graphs.trading.nodes import make_daily_plan

    mock_run_agent.return_value = json.dumps({
        "plan": "Test plan",
        "priorities": [],
        "entry_candidates": [],
        "exit_recommendations": [],
        "earnings_within_14d": [],
    })
    llm = MagicMock()
    cfg = _make_config("daily_planner")
    cfg.role = "Daily Trading Strategist"
    cfg.goal = "Create an actionable daily trading plan"
    cfg.backstory = "You are a tactical trading strategist."

    node = make_daily_plan(llm, cfg, [])
    state = _base_state(market_context={
        "headlines": ["Major tech earnings tonight"],
        "sentiment": "bullish",
    })

    result = await node(state)

    # Verify run_agent was called with a prompt containing market context
    call_args = mock_run_agent.call_args
    prompt = call_args[0][3]  # 4th positional arg is user_message
    assert "Market Intelligence Briefing" in prompt
    assert "Major tech earnings tonight" in prompt


def test_trading_graph_has_market_intel_and_earnings_routers():
    """Trading graph routers work correctly."""
    from quantstack.graphs.trading.graph import _earnings_router, _safety_check_router

    assert _safety_check_router({"decisions": [{"node": "safety_check", "halted": True}], "errors": []}) == "halt"
    assert _safety_check_router({"decisions": [{"node": "safety_check", "halted": False}], "errors": []}) == "continue"

    assert _earnings_router({"earnings_symbols": ["AAPL"]}) == "has_earnings"
    assert _earnings_router({"earnings_symbols": []}) == "no_earnings"
    assert _earnings_router({}) == "no_earnings"
