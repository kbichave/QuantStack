"""Tests for earnings_analyst routing and node in trading graph."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from quantstack.graphs.trading.graph import _earnings_router
from quantstack.graphs.trading.nodes import make_earnings_analysis


def _make_config(name="earnings_analyst"):
    cfg = MagicMock()
    cfg.name = name
    cfg.role = "Earnings Event Specialist"
    cfg.goal = "Analyze earnings catalysts and recommend options structures"
    cfg.backstory = "You analyze earnings events."
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


def test_earnings_router_routes_to_analysis_when_earnings_present():
    """Conditional routing to earnings_analysis when earnings_symbols non-empty."""
    state = _base_state(earnings_symbols=["AAPL", "MSFT"])
    assert _earnings_router(state) == "has_earnings"


def test_earnings_router_skips_when_no_earnings():
    """No routing to earnings_analysis when earnings_symbols empty."""
    state = _base_state(earnings_symbols=[])
    assert _earnings_router(state) == "no_earnings"


def test_earnings_router_skips_when_field_missing():
    """No routing when earnings_symbols field not present."""
    state = _base_state()
    del state["earnings_symbols"]
    assert _earnings_router(state) == "no_earnings"


@pytest.mark.asyncio
@patch("quantstack.graphs.trading.nodes.run_agent")
async def test_earnings_analysis_produces_analysis(mock_run_agent):
    """earnings_analysis output contains analysis for earnings symbols."""
    mock_run_agent.return_value = json.dumps({
        "analyses": [{
            "symbol": "AAPL",
            "beat_rate": 0.85,
            "iv_premium_ratio": 1.3,
            "recommendation": "HOLD_THROUGH",
            "options_suggestion": "iron_condor",
            "reasoning": "High beat rate, moderate IV premium",
        }],
    })
    llm = MagicMock()
    cfg = _make_config()
    node = make_earnings_analysis(llm, cfg, [])
    state = _base_state(earnings_symbols=["AAPL"])

    result = await node(state)

    assert "earnings_analysis" in result
    analyses = result["earnings_analysis"].get("analyses", [])
    assert len(analyses) == 1
    assert analyses[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_earnings_analysis_noop_when_empty():
    """earnings_analysis is a no-op when earnings_symbols is empty."""
    llm = MagicMock()
    cfg = _make_config()
    node = make_earnings_analysis(llm, cfg, [])
    state = _base_state(earnings_symbols=[])

    result = await node(state)

    assert result["earnings_analysis"] == {}


@pytest.mark.asyncio
@patch("quantstack.graphs.trading.nodes.run_agent")
async def test_earnings_analysis_handles_failure(mock_run_agent):
    """earnings_analyst handles missing data gracefully."""
    mock_run_agent.side_effect = RuntimeError("fetch_earnings_data returned None")
    llm = MagicMock()
    cfg = _make_config()
    node = make_earnings_analysis(llm, cfg, [])
    state = _base_state(earnings_symbols=["AAPL"])

    result = await node(state)

    assert result["earnings_analysis"] == {}
    assert any("earnings_analysis" in e for e in result.get("errors", []))


@pytest.mark.asyncio
@patch("quantstack.graphs.trading.nodes.run_agent")
async def test_plan_day_detects_earnings_symbols(mock_run_agent):
    """plan_day populates earnings_symbols from LLM response."""
    from quantstack.graphs.trading.nodes import make_daily_plan

    mock_run_agent.return_value = json.dumps({
        "plan": "Test plan",
        "priorities": [],
        "entry_candidates": [],
        "exit_recommendations": [],
        "earnings_within_14d": ["AAPL", "MSFT"],
    })
    cfg = MagicMock()
    cfg.name = "daily_planner"
    cfg.role = "Daily Trading Strategist"
    cfg.goal = "Create an actionable daily trading plan"
    cfg.backstory = "You are a tactical trading strategist."
    llm = MagicMock()

    node = make_daily_plan(llm, cfg, [])
    state = _base_state()

    result = await node(state)

    assert result["earnings_symbols"] == ["AAPL", "MSFT"]
    assert isinstance(result["earnings_symbols"], list)
