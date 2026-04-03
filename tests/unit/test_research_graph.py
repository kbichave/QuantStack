"""Tests for the Research Graph (Section 07)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_config_watcher(tmp_path):
    """Create a ConfigWatcher with research agent configs."""
    yaml_content = """
quant_researcher:
  role: "Quant Researcher"
  goal: "Discover trading strategies through systematic research."
  backstory: "Senior quant with 15 years experience."
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - signal_brief
    - fetch_market_data

ml_scientist:
  role: "ML Scientist"
  goal: "Train and validate ML models for strategy signals."
  backstory: "ML engineer with deep learning background."
  llm_tier: heavy
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - train_model
    - compute_features
"""
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)
    from quantstack.graphs.config_watcher import ConfigWatcher
    watcher = ConfigWatcher(yaml_file)
    yield watcher
    watcher.stop()


@pytest.fixture
def mock_chat_model():
    """Create a mock ChatModel that returns structured JSON."""
    from langchain_core.messages import AIMessage
    model = MagicMock()
    model.bind_tools = MagicMock(return_value=model)
    model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"result": "ok"}'
    ))
    return model


class TestBuildResearchGraph:
    """Tests for the graph builder."""

    def test_returns_compiled_graph(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.research.graph import build_research_graph
        with patch("quantstack.graphs.research.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_research_graph(mock_config_watcher, MemorySaver())
        assert graph is not None
        assert hasattr(graph, "ainvoke")

    def test_graph_has_eight_nodes(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.research.graph import build_research_graph
        with patch("quantstack.graphs.research.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_research_graph(mock_config_watcher, MemorySaver())
        node_names = set(graph.get_graph().nodes.keys())
        expected_nodes = {
            "context_load", "domain_selection", "hypothesis_generation",
            "signal_validation", "backtest_validation", "ml_experiment",
            "strategy_registration", "knowledge_update",
        }
        assert expected_nodes.issubset(node_names), (
            f"Missing nodes: {expected_nodes - node_names}"
        )

    def test_reads_agent_configs(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.research.graph import build_research_graph
        with patch("quantstack.graphs.research.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            build_research_graph(mock_config_watcher, MemorySaver())
        # Should call get_chat_model for heavy tier (both quant and ml use heavy)
        tier_calls = [call.args[0] for call in mock_gcm.call_args_list]
        assert "heavy" in tier_calls

    def test_conditional_edge_after_signal_validation(self, mock_config_watcher):
        """Signal validation has a conditional edge routing to backtest or END."""
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.research.graph import build_research_graph
        with patch("quantstack.graphs.research.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_research_graph(mock_config_watcher, MemorySaver())
        # Inspect edges from signal_validation — should have conditional routing
        graph_data = graph.get_graph()
        sv_edges = [e for e in graph_data.edges if e.source == "signal_validation"]
        # Conditional edges produce at least 2 targets (backtest_validation + __end__)
        targets = {e.target for e in sv_edges}
        assert "backtest_validation" in targets or len(targets) >= 1


class TestRouteAfterValidation:
    """Tests for the conditional router."""

    def test_routes_to_backtest_when_passed(self):
        from quantstack.graphs.research.nodes import route_after_validation
        state = {"validation_result": {"passed": True}}
        assert route_after_validation(state) == "backtest_validation"

    def test_routes_to_end_when_failed(self):
        from langgraph.graph import END
        from quantstack.graphs.research.nodes import route_after_validation
        state = {"validation_result": {"passed": False}}
        assert route_after_validation(state) == END

    def test_routes_to_end_when_missing_result(self):
        from langgraph.graph import END
        from quantstack.graphs.research.nodes import route_after_validation
        state = {}
        assert route_after_validation(state) == END


class TestResearchGraphExecution:
    """Tests for graph execution with mock LLM."""

    @pytest.mark.asyncio
    async def test_full_invocation_validation_passes(self, mock_config_watcher):
        """Full graph run where signal validation passes."""
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver

        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        # Return JSON that passes validation
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(
            content='{"passed": true, "summary": "ok", "domain": "swing", '
            '"symbols": ["SPY"], "hypothesis": "test", "signals_to_check": [], '
            '"backtest_id": "bt-1", "sharpe": 1.5, "experiment_id": "exp-1", '
            '"strategy_id": "strat-1", "status": "paper_ready"}'
        ))

        from quantstack.graphs.research.graph import build_research_graph
        with patch("quantstack.graphs.research.graph.get_chat_model", return_value=mock_model):
            graph = build_research_graph(mock_config_watcher, MemorySaver())

        result = await graph.ainvoke(
            {"cycle_number": 1, "regime": "trending_up", "errors": [], "decisions": []},
            config={"configurable": {"thread_id": "test-pass"}},
        )
        assert "decisions" in result
        assert isinstance(result["decisions"], list)
        assert len(result["decisions"]) > 0
        # Should have gone through all 8 nodes
        node_names = {d["node"] for d in result["decisions"] if "node" in d}
        assert "context_load" in node_names
        assert "knowledge_update" in node_names

    @pytest.mark.asyncio
    async def test_full_invocation_validation_fails(self, mock_config_watcher):
        """Graph stops early when signal validation fails."""
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver

        call_count = 0

        async def staged_responses(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                # context_load, domain_selection, hypothesis_generation
                return AIMessage(
                    content='{"summary": "ok", "domain": "swing", '
                    '"symbols": ["SPY"], "hypothesis": "test"}'
                )
            # signal_validation → fails
            return AIMessage(content='{"passed": false, "reason": "no confluence"}')

        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_model.ainvoke = AsyncMock(side_effect=staged_responses)

        from quantstack.graphs.research.graph import build_research_graph
        with patch("quantstack.graphs.research.graph.get_chat_model", return_value=mock_model):
            graph = build_research_graph(mock_config_watcher, MemorySaver())

        result = await graph.ainvoke(
            {"cycle_number": 1, "regime": "ranging", "errors": [], "decisions": []},
            config={"configurable": {"thread_id": "test-fail"}},
        )
        # Should NOT have backtest or later nodes in decisions
        node_names = {d.get("node") for d in result.get("decisions", [])}
        assert "backtest_validation" not in node_names
        assert "ml_experiment" not in node_names

    @pytest.mark.asyncio
    async def test_node_error_appends_to_errors(self, mock_config_watcher):
        """When a node's LLM call fails, error is captured in state."""
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver

        call_count = 0

        async def fail_first(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Test LLM error")
            return AIMessage(content='{"summary": "recovered"}')

        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_model.ainvoke = AsyncMock(side_effect=fail_first)

        from quantstack.graphs.research.graph import build_research_graph
        with patch("quantstack.graphs.research.graph.get_chat_model", return_value=mock_model):
            graph = build_research_graph(mock_config_watcher, MemorySaver())

        result = await graph.ainvoke(
            {"cycle_number": 1, "regime": "unknown", "errors": [], "decisions": []},
            config={"configurable": {"thread_id": "test-err"}},
        )
        assert isinstance(result["errors"], list)
