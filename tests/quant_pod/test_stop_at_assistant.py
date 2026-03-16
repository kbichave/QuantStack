# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the stop_at_assistant mode on TradingCrew.

Verifies that when stop_at_assistant=True:
  - The SuperTrader agent is NOT included in the agent list
  - The trade_decision_task is NOT included in the task list
  - The last task outputs DailyBrief (not TradeDecision)

LLM creation is patched in the crew_and_roster fixture, so no real
API key is required for these structural tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from quant_pod.crews.schemas import DailyBrief, TradeDecision

# ---------------------------------------------------------------------------
# Lightweight stubs
# ---------------------------------------------------------------------------


@dataclass
class StubAgent:
    role: str


@dataclass
class StubTask:
    description: str
    output_pydantic: type = None


# ---------------------------------------------------------------------------
# Fixture: TradingCrew with all factories stubbed
# ---------------------------------------------------------------------------


@pytest.fixture
def crew_and_roster():
    """
    Build a TradingCrew with all factory methods stubbed out.
    Returns (TradingCrew, PodSelection).
    """
    from quant_pod.crews.assembler import PodSelection
    from quant_pod.crews.trading_crew import TradingCrew

    # Patch create_llm at the call site so Agent construction doesn't require
    # a real API key (agent.core imports create_llm with a local binding).
    mock_llm = MagicMock()
    with patch("crewai.agent.core.create_llm", return_value=mock_llm):
        tc = TradingCrew()

    # Stub IC agent factories
    tc._ic_agent_factories = lambda: {
        "data_ingestion_ic": lambda: StubAgent("data_ingestion_ic"),
        "trend_momentum_ic": lambda: StubAgent("trend_momentum_ic"),
    }

    # Stub pod manager agent factories
    tc._pod_manager_factories = lambda: {
        "data_pod_manager": lambda: StubAgent("data_pod_manager"),
        "technicals_pod_manager": lambda: StubAgent("technicals_pod_manager"),
    }

    # Stub IC task factories
    tc._ic_task_factories = lambda: {
        "data_ingestion_ic": lambda: StubTask("fetch_data_task"),
        "trend_momentum_ic": lambda: StubTask("trend_momentum_task"),
    }

    # Stub pod task factories
    tc._pod_task_factories = lambda: {
        "data_pod_manager": lambda: StubTask("data_pod_compile_task"),
        "technicals_pod_manager": lambda: StubTask("technicals_compile_task"),
    }

    # Stub Layer 3 + Layer 4
    tc.trading_assistant = lambda: StubAgent("trading_assistant")
    tc.super_trader = lambda: StubAgent("super_trader")
    tc.assistant_synthesis_task = lambda: StubTask(
        "assistant_synthesis_task", output_pydantic=DailyBrief
    )
    tc.trade_decision_task = lambda: StubTask("trade_decision_task", output_pydantic=TradeDecision)

    roster = PodSelection(
        asset_class="equities",
        ic_agents=["data_ingestion_ic", "trend_momentum_ic"],
        pod_managers=["data_pod_manager", "technicals_pod_manager"],
    )

    return tc, roster


# ---------------------------------------------------------------------------
# Tests — _build_agents
# ---------------------------------------------------------------------------


class TestBuildAgents:
    def test_default_includes_super_trader(self, crew_and_roster):
        tc, roster = crew_and_roster
        agents = tc._build_agents(roster, stop_at_assistant=False)
        names = [a.role for a in agents]
        assert "super_trader" in names

    def test_default_includes_assistant(self, crew_and_roster):
        tc, roster = crew_and_roster
        agents = tc._build_agents(roster, stop_at_assistant=False)
        names = [a.role for a in agents]
        assert "trading_assistant" in names

    def test_stop_excludes_super_trader(self, crew_and_roster):
        tc, roster = crew_and_roster
        agents = tc._build_agents(roster, stop_at_assistant=True)
        names = [a.role for a in agents]
        assert "super_trader" not in names

    def test_stop_keeps_assistant(self, crew_and_roster):
        tc, roster = crew_and_roster
        agents = tc._build_agents(roster, stop_at_assistant=True)
        names = [a.role for a in agents]
        assert "trading_assistant" in names

    def test_stop_has_one_fewer_agent(self, crew_and_roster):
        tc, roster = crew_and_roster
        default = tc._build_agents(roster, stop_at_assistant=False)
        stopped = tc._build_agents(roster, stop_at_assistant=True)
        assert len(stopped) == len(default) - 1

    def test_stop_preserves_ic_and_pod_agents(self, crew_and_roster):
        tc, roster = crew_and_roster
        agents = tc._build_agents(roster, stop_at_assistant=True)
        names = [a.role for a in agents]
        assert "data_ingestion_ic" in names
        assert "trend_momentum_ic" in names
        assert "data_pod_manager" in names
        assert "technicals_pod_manager" in names


# ---------------------------------------------------------------------------
# Tests — _build_tasks
# ---------------------------------------------------------------------------


class TestBuildTasks:
    def test_default_last_task_is_trade_decision(self, crew_and_roster):
        tc, roster = crew_and_roster
        tasks = tc._build_tasks(roster, stop_at_assistant=False)
        assert tasks[-1].output_pydantic == TradeDecision

    def test_stop_excludes_trade_decision(self, crew_and_roster):
        tc, roster = crew_and_roster
        tasks = tc._build_tasks(roster, stop_at_assistant=True)
        for t in tasks:
            assert t.output_pydantic != TradeDecision

    def test_stop_last_task_outputs_daily_brief(self, crew_and_roster):
        tc, roster = crew_and_roster
        tasks = tc._build_tasks(roster, stop_at_assistant=True)
        assert tasks[-1].output_pydantic == DailyBrief

    def test_stop_has_one_fewer_task(self, crew_and_roster):
        tc, roster = crew_and_roster
        default = tc._build_tasks(roster, stop_at_assistant=False)
        stopped = tc._build_tasks(roster, stop_at_assistant=True)
        assert len(stopped) == len(default) - 1


# ---------------------------------------------------------------------------
# Tests — run_analysis_only (no API key needed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(False, reason="")  # Always run — no API key needed
class TestRunAnalysisOnly:
    def test_is_importable(self):
        from quant_pod.crews.trading_crew import run_analysis_only

        assert callable(run_analysis_only)

    def test_in_all(self):
        from quant_pod.crews import trading_crew

        assert "run_analysis_only" in trading_crew.__all__
