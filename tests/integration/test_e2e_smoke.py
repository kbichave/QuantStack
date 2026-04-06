"""E2E smoke tests: build and invoke each graph with minimal state.

Replaces the old CrewAI crew assembly test. Verifies that graphs compile,
accept initial state, execute all nodes, and return a valid final state.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def research_config_watcher(tmp_path):
    yaml_content = """
quant_researcher:
  role: "Quant Researcher"
  goal: "Discover trading strategies."
  backstory: "Senior quant."
  llm_tier: heavy
  tools: []
ml_scientist:
  role: "ML Scientist"
  goal: "Train ML models."
  backstory: "ML engineer."
  llm_tier: heavy
  tools: []
hypothesis_critic:
  role: "Hypothesis Critic"
  goal: "Critique hypotheses."
  backstory: "Critic."
  llm_tier: medium
  tools: []
"""
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)
    from quantstack.graphs.config_watcher import ConfigWatcher
    watcher = ConfigWatcher(yaml_file)
    yield watcher
    watcher.stop()


@pytest.fixture
def trading_config_watcher(tmp_path):
    yaml_content = """
daily_planner:
  role: "Planner"
  goal: "Plan."
  backstory: "Plans."
  llm_tier: medium
  tools: []
position_monitor:
  role: "Monitor"
  goal: "Monitor."
  backstory: "Monitors."
  llm_tier: medium
  tools: []
exit_evaluator:
  role: "Exit Evaluator"
  goal: "Evaluate exits."
  backstory: "Evaluates."
  llm_tier: medium
  tools: []
trade_debater:
  role: "Debater"
  goal: "Debate."
  backstory: "Debates."
  llm_tier: heavy
  tools: []
fund_manager:
  role: "FM"
  goal: "Manage."
  backstory: "Manages."
  llm_tier: heavy
  tools: []
options_analyst:
  role: "Options"
  goal: "Options."
  backstory: "Options."
  llm_tier: heavy
  tools: []
trade_reflector:
  role: "Reflector"
  goal: "Reflect."
  backstory: "Reflects."
  llm_tier: medium
  tools: []
market_intel:
  role: "Market Intel"
  goal: "Market intelligence."
  backstory: "Intel."
  llm_tier: medium
  tools: []
earnings_analyst:
  role: "Earnings"
  goal: "Earnings analysis."
  backstory: "Earnings."
  llm_tier: medium
  tools: []
"""
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)
    from quantstack.graphs.config_watcher import ConfigWatcher
    watcher = ConfigWatcher(yaml_file)
    yield watcher
    watcher.stop()


@pytest.fixture
def supervisor_config_watcher(tmp_path):
    yaml_content = """
health_monitor:
  role: "Health Monitor"
  goal: "Check health."
  backstory: "Checks."
  llm_tier: light
  tools: []
self_healer:
  role: "Self Healer"
  goal: "Heal."
  backstory: "Heals."
  llm_tier: medium
  tools: []
strategy_promoter:
  role: "Promoter"
  goal: "Promote."
  backstory: "Promotes."
  llm_tier: medium
  tools: []
"""
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)
    from quantstack.graphs.config_watcher import ConfigWatcher
    watcher = ConfigWatcher(yaml_file)
    yield watcher
    watcher.stop()


class TestResearchGraphSmoke:
    """Research graph builds and completes a minimal cycle."""

    def test_graph_compiles(self, research_config_watcher, mock_checkpointer):
        from quantstack.graphs.research.graph import build_research_graph
        with patch("quantstack.graphs.research.graph.get_chat_model", return_value=MagicMock()):
            graph = build_research_graph(research_config_watcher, mock_checkpointer)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_graph_completes_without_error(self, research_config_watcher, mock_checkpointer, mock_chat_model):
        from quantstack.graphs.research.graph import build_research_graph
        with patch("quantstack.graphs.research.graph.get_chat_model", return_value=mock_chat_model):
            graph = build_research_graph(research_config_watcher, mock_checkpointer)
        result = await graph.ainvoke(
            {"cycle_number": 1, "regime": "trending_up", "errors": [], "decisions": []},
            config={"configurable": {"thread_id": "smoke-research"}},
        )
        assert isinstance(result.get("errors"), list)
        assert isinstance(result.get("decisions"), list)


class TestTradingGraphSmoke:
    """Trading graph builds and completes a minimal cycle."""

    def test_graph_compiles(self, trading_config_watcher, mock_checkpointer):
        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model", return_value=MagicMock()):
            graph = build_trading_graph(trading_config_watcher, mock_checkpointer)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_graph_completes_without_error(self, trading_config_watcher, mock_checkpointer, mock_chat_model):
        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model", return_value=mock_chat_model):
            graph = build_trading_graph(trading_config_watcher, mock_checkpointer)
        result = await graph.ainvoke(
            {
                "cycle_number": 1, "regime": "trending_up",
                "portfolio_context": {"positions": [], "total_equity": 100000,
                                      "daily_pnl_pct": 0, "gross_exposure_pct": 0.5,
                                      "average_daily_volume": 1000000},
                "errors": [], "decisions": [],
            },
            config={"configurable": {"thread_id": "smoke-trading"}},
        )
        assert isinstance(result.get("errors"), list)
        assert isinstance(result.get("decisions"), list)


class TestSupervisorGraphSmoke:
    """Supervisor graph builds and completes a minimal cycle."""

    def test_graph_compiles(self, supervisor_config_watcher, mock_checkpointer):
        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        with patch("quantstack.graphs.supervisor.graph.get_chat_model", return_value=MagicMock()):
            graph = build_supervisor_graph(supervisor_config_watcher, mock_checkpointer)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_graph_completes_without_error(self, supervisor_config_watcher, mock_checkpointer, mock_chat_model):
        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        with patch("quantstack.graphs.supervisor.graph.get_chat_model", return_value=mock_chat_model):
            graph = build_supervisor_graph(supervisor_config_watcher, mock_checkpointer)
        result = await graph.ainvoke(
            {"cycle_number": 1, "errors": []},
            config={"configurable": {"thread_id": "smoke-supervisor"}},
        )
        assert isinstance(result.get("errors"), list)
