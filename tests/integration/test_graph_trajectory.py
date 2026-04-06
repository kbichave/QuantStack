"""Graph trajectory integration tests (WI-2).

Verify safety-critical node ordering invariants by capturing
the actual node traversal from trading graph executions and
comparing against expected patterns.

Uses agentevals.graph_trajectory.strict for trajectory matching.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentevals.graph_trajectory.strict import graph_trajectory_strict_match
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRADING_AGENTS_YAML = """\
daily_planner:
  role: "Daily Trading Planner"
  goal: "Generate actionable daily trading plan."
  backstory: "Senior trading planner."
  llm_tier: medium
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - signal_brief

position_monitor:
  role: "Position Monitor"
  goal: "Review open positions."
  backstory: "Position monitoring specialist."
  llm_tier: medium
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - signal_brief

trade_debater:
  role: "Trade Entry Debater"
  goal: "Find entry candidates."
  backstory: "Market scanner."
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - signal_brief
    - fetch_market_data

risk_analyst:
  role: "Risk Analyst"
  goal: "Size positions and validate risk."
  backstory: "Risk management specialist."
  llm_tier: medium
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - compute_risk_metrics

fund_manager:
  role: "Fund Manager"
  goal: "Review proposed entries for portfolio risk."
  backstory: "Portfolio-level approval gate."
  llm_tier: heavy
  max_iterations: 10
  timeout_seconds: 180
  tools:
    - fetch_portfolio

options_analyst:
  role: "Options Analyst"
  goal: "Select optimal options structures."
  backstory: "Options specialist."
  llm_tier: heavy
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - fetch_portfolio

trade_reflector:
  role: "Trade Reflector"
  goal: "Analyze completed trades."
  backstory: "Post-trade analyst."
  llm_tier: medium
  max_iterations: 5
  timeout_seconds: 120
  tools:
    - signal_brief

market_intel:
  role: "Market Intelligence"
  goal: "Pre-market intelligence gathering."
  backstory: "Market intelligence specialist."
  llm_tier: medium
  max_iterations: 5
  timeout_seconds: 120
  tools:
    - signal_brief

earnings_analyst:
  role: "Earnings Analyst"
  goal: "Analyze earnings impact."
  backstory: "Earnings specialist."
  llm_tier: medium
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - signal_brief
"""

HEALTHY_STATE = {
    "cycle_number": 1,
    "regime": "trending_up",
    "portfolio_context": {
        "positions": [],
        "total_equity": 100000,
        "daily_pnl_pct": 0.0,
        "gross_exposure_pct": 0.5,
        "average_daily_volume": 1000000,
    },
    "errors": [],
    "decisions": [],
}


async def capture_trajectory(graph, input_state: dict, thread_id: str) -> list[str]:
    """Run graph via astream and return ordered list of node names that executed."""
    nodes: list[str] = []
    async for update in graph.astream(
        input_state,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        if isinstance(update, tuple):
            node_name = update[0]
        elif isinstance(update, dict):
            node_name = next(iter(update.keys()), None)
        else:
            continue
        if node_name and node_name not in ("__start__", "__end__"):
            nodes.append(node_name)
    return nodes


def _build_trajectory(steps: list[list[str]]) -> dict:
    """Build a GraphTrajectory dict for agentevals."""
    return {
        "inputs": None,
        "results": [{}] * len(steps),
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_watcher(tmp_path):
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(TRADING_AGENTS_YAML)
    from quantstack.graphs.config_watcher import ConfigWatcher
    watcher = ConfigWatcher(yaml_file)
    yield watcher
    watcher.stop()


def _make_happy_path_model():
    """Mock LLM that navigates the full happy path with risk approved."""
    call_count = 0

    async def staged(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # safety_check: healthy
        if call_count == 1:
            return AIMessage(content='{"halted": false, "reason": "healthy"}')
        # market_intel
        if call_count == 2:
            return AIMessage(content='{"summary": "market up", "vix": 15}')
        # plan_day
        if call_count == 3:
            return AIMessage(content='{"plan": "momentum focus", "priorities": ["SPY"]}')
        # position_review + entry_scan (parallel, order varies)
        if call_count in (4, 5):
            return AIMessage(content='[{"symbol": "SPY", "action": "HOLD", '
                             '"strategy": "swing", "signal_strength": 0.8, '
                             '"recommended_size_pct": 3.0, "reasoning": "ok", "confidence": 0.8}]')
        # execute_exits
        if call_count == 6:
            return AIMessage(content='[]')
        # risk_sizing — triggers SafetyGate which we'll patch
        if call_count == 7:
            return AIMessage(content='[{"symbol": "SPY", "recommended_size_pct": 3.0, '
                             '"reasoning": "half-Kelly", "confidence": 0.8}]')
        # portfolio_review
        if call_count == 8:
            return AIMessage(content='[{"symbol": "SPY", "decision": "APPROVED", "reason": "ok"}]')
        # analyze_options
        if call_count == 9:
            return AIMessage(content='[]')
        # execute_entries
        if call_count == 10:
            return AIMessage(content='[]')
        # reflection
        if call_count == 11:
            return AIMessage(content='{"reflection": "good cycle", "lessons": []}')
        return AIMessage(content='{"result": "ok"}')

    model = MagicMock()
    model.bind_tools = MagicMock(return_value=model)
    model.ainvoke = AsyncMock(side_effect=staged)
    return model


def _make_halted_model():
    """Mock LLM that triggers a halt at safety_check."""
    model = MagicMock()
    model.bind_tools = MagicMock(return_value=model)
    model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"halted": true, "reason": "daily loss halt triggered"}'
    ))
    return model


def _make_risk_rejected_model():
    """Mock LLM that gets to risk_sizing but produces empty verdicts (rejected)."""
    call_count = 0

    async def staged(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AIMessage(content='{"halted": false}')
        if call_count == 2:
            return AIMessage(content='{"summary": "mixed signals"}')
        if call_count == 3:
            return AIMessage(content='{"plan": "cautious"}')
        if call_count in (4, 5):
            return AIMessage(content='[{"symbol": "SPY", "action": "HOLD", '
                             '"strategy": "swing", "signal_strength": 0.5, '
                             '"recommended_size_pct": 40.0, "reasoning": "yolo", "confidence": 0.3}]')
        if call_count == 6:
            return AIMessage(content='[]')
        # risk_sizing: oversized position that SafetyGate rejects
        if call_count == 7:
            return AIMessage(content='[{"symbol": "SPY", "recommended_size_pct": 40.0, '
                             '"reasoning": "yolo", "confidence": 0.3}]')
        return AIMessage(content='{"result": "ok"}')

    model = MagicMock()
    model.bind_tools = MagicMock(return_value=model)
    model.ainvoke = AsyncMock(side_effect=staged)
    return model


def _build_graph(config_watcher, model):
    """Build a trading graph with mocked LLM and patched dependencies."""
    from quantstack.graphs.trading.graph import build_trading_graph
    with patch("quantstack.graphs.trading.graph.get_chat_model", return_value=model):
        return build_trading_graph(config_watcher, MemorySaver())


# ---------------------------------------------------------------------------
# Tests: Safety Invariants
# ---------------------------------------------------------------------------


class TestSafetyInvariants:
    """Safety-critical trajectory invariants."""

    @pytest.mark.asyncio
    async def test_safety_check_first(self, config_watcher):
        """safety_check is first non-data node in traversal."""
        model = _make_happy_path_model()
        # Patch SafetyGate to approve (avoid real risk gate logic)
        with patch("quantstack.graphs.trading.nodes.SafetyGate", return_value=_MockSafetyGate()):
            graph = _build_graph(config_watcher, model)
            trajectory = await capture_trajectory(graph, HEALTHY_STATE, "test-safety-first")

        assert len(trajectory) >= 2
        assert trajectory[0] == "data_refresh"
        assert trajectory[1] == "safety_check"

    @pytest.mark.asyncio
    async def test_halted_skips_trading(self, config_watcher):
        """Halted state produces trajectory [data_refresh, safety_check] only."""
        model = _make_halted_model()
        graph = _build_graph(config_watcher, model)
        trajectory = await capture_trajectory(graph, HEALTHY_STATE, "test-halted")

        assert trajectory == ["data_refresh", "safety_check"]

        # Also verify with agentevals strict match
        actual = _build_trajectory([[n] for n in trajectory])
        expected = _build_trajectory([["data_refresh"], ["safety_check"]])
        result = graph_trajectory_strict_match(
            outputs=actual, reference_outputs=expected
        )
        assert result["score"] is True

    @pytest.mark.asyncio
    async def test_risk_gate_before_execution(self, config_watcher):
        """risk_sizing appears before execute_entries with all intermediate nodes."""
        model = _make_happy_path_model()
        with patch("quantstack.graphs.trading.nodes.SafetyGate", return_value=_MockSafetyGate()):
            graph = _build_graph(config_watcher, model)
            with patch("quantstack.graphs.trading.nodes.db_conn"):
                trajectory = await capture_trajectory(
                    graph, HEALTHY_STATE, "test-risk-before-exec"
                )

        assert "risk_sizing" in trajectory
        assert "execute_entries" in trajectory
        risk_idx = trajectory.index("risk_sizing")
        exec_idx = trajectory.index("execute_entries")
        assert risk_idx < exec_idx, (
            f"risk_sizing at {risk_idx} must precede execute_entries at {exec_idx}"
        )
        # All intermediate nodes appear between risk_sizing and execute_entries
        between = set(trajectory[risk_idx + 1:exec_idx])
        assert "portfolio_construction" in between
        assert "portfolio_review" in between
        assert "analyze_options" in between

    @pytest.mark.asyncio
    async def test_reflect_after_execution(self, config_watcher):
        """reflect appears after execute_entries in non-halted paths."""
        model = _make_happy_path_model()
        with patch("quantstack.graphs.trading.nodes.SafetyGate", return_value=_MockSafetyGate()):
            graph = _build_graph(config_watcher, model)
            with patch("quantstack.graphs.trading.nodes.db_conn"):
                trajectory = await capture_trajectory(
                    graph, HEALTHY_STATE, "test-reflect-after"
                )

        assert "execute_entries" in trajectory
        assert "reflect" in trajectory
        assert trajectory.index("execute_entries") < trajectory.index("reflect")

    @pytest.mark.asyncio
    async def test_rejected_gate_skips_execution(self, config_watcher):
        """Rejected risk_verdicts produces trajectory without execute_entries."""
        model = _make_risk_rejected_model()
        graph = _build_graph(config_watcher, model)
        trajectory = await capture_trajectory(graph, HEALTHY_STATE, "test-rejected")

        assert "risk_sizing" in trajectory
        assert "execute_entries" not in trajectory
        assert "reflect" not in trajectory


# ---------------------------------------------------------------------------
# Helper: mock RiskDecision for approved paths
# ---------------------------------------------------------------------------


class _ApprovedVerdict:
    """Minimal mock for a SafetyGate.validate() approved result."""
    def __init__(self):
        self.approved = True
        self.violations = []


class _MockSafetyGate:
    """SafetyGate replacement that approves everything."""
    def validate(self, decision, portfolio_ctx):
        return _ApprovedVerdict()
