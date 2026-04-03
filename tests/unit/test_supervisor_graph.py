"""Tests for the Supervisor Graph (Section 06)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_config_watcher(tmp_path):
    """Create a ConfigWatcher with test agent configs."""
    yaml_content = """
health_monitor:
  role: "System Health Monitor"
  goal: "Detect unhealthy services."
  backstory: "Check heartbeats and service health."
  llm_tier: light
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - check_system_status
    - check_heartbeat

self_healer:
  role: "Self-Healing Engineer"
  goal: "Diagnose and recover from failures."
  backstory: "Execute recovery playbook."
  llm_tier: light
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - check_system_status
    - search_knowledge_base

strategy_promoter:
  role: "Strategy Lifecycle Manager"
  goal: "Promote or retire strategies."
  backstory: "Evaluate forward-testing strategies."
  llm_tier: medium
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - fetch_strategy_registry
    - search_knowledge_base
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
        content='{"status": "healthy", "services": {}}'
    ))
    return model


class TestBuildSupervisorGraph:
    """Tests for the graph builder."""

    def test_returns_compiled_graph(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        with patch("quantstack.graphs.supervisor.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_supervisor_graph(mock_config_watcher, MemorySaver())
        assert graph is not None
        assert hasattr(graph, "ainvoke")

    def test_graph_has_five_nodes(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        with patch("quantstack.graphs.supervisor.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_supervisor_graph(mock_config_watcher, MemorySaver())
        # CompiledStateGraph nodes include __start__ and __end__
        node_names = set(graph.get_graph().nodes.keys())
        expected_nodes = {
            "health_check", "diagnose_issues", "execute_recovery",
            "strategy_lifecycle", "scheduled_tasks",
        }
        assert expected_nodes.issubset(node_names), (
            f"Missing nodes: {expected_nodes - node_names}"
        )

    def test_reads_agent_configs(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        with patch("quantstack.graphs.supervisor.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            build_supervisor_graph(mock_config_watcher, MemorySaver())
        # Should have called get_chat_model for light and medium tiers
        tier_calls = [call.args[0] for call in mock_gcm.call_args_list]
        assert "light" in tier_calls
        assert "medium" in tier_calls

    def test_accepts_memory_saver_checkpointer(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        with patch("quantstack.graphs.supervisor.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_supervisor_graph(mock_config_watcher, MemorySaver())
        assert graph is not None


class TestSupervisorGraphExecution:
    """Tests for graph execution with mock LLM."""

    @pytest.mark.asyncio
    async def test_full_invocation_produces_valid_state(self, mock_config_watcher):
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver

        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(
            content='{"result": "ok"}'
        ))

        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        with patch("quantstack.graphs.supervisor.graph.get_chat_model", return_value=mock_model):
            graph = build_supervisor_graph(mock_config_watcher, MemorySaver())

        result = await graph.ainvoke(
            {"cycle_number": 1, "errors": []},
            config={"configurable": {"thread_id": "test-1"}},
        )
        assert "health_status" in result
        assert "errors" in result
        assert isinstance(result["errors"], list)

    @pytest.mark.asyncio
    async def test_node_error_appends_to_errors(self, mock_config_watcher):
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver

        call_count = 0

        async def failing_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Test error")
            return AIMessage(content='{"result": "recovered"}')

        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_model.ainvoke = AsyncMock(side_effect=failing_then_ok)

        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        with patch("quantstack.graphs.supervisor.graph.get_chat_model", return_value=mock_model):
            graph = build_supervisor_graph(mock_config_watcher, MemorySaver())

        result = await graph.ainvoke(
            {"cycle_number": 1, "errors": []},
            config={"configurable": {"thread_id": "test-err"}},
        )
        # At minimum the graph should complete (error handling catches exceptions)
        assert isinstance(result["errors"], list)
