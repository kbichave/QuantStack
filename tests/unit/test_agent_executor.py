"""Tests for agent executor tool search integration (Section 06)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from quantstack.graphs.agent_executor import (
    _AGENT_TEAMS,
    _TOOL_CATEGORIES,
    _is_server_tool_call,
    build_system_message,
    run_agent,
)
from quantstack.graphs.config import AgentConfig


def _make_config(name="test_agent", tools=(), always_loaded=()):
    return AgentConfig(
        name=name,
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm_tier="heavy",
        tools=tools,
        always_loaded_tools=always_loaded,
    )


def _make_fake_tool(name: str) -> MagicMock:
    t = MagicMock(spec=BaseTool)
    t.name = name
    t.ainvoke = AsyncMock(return_value=f"result from {name}")
    return t


def _make_ai_response(content="", tool_calls=None):
    msg = MagicMock(spec=AIMessage)
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


class TestServerToolUseFilter:

    def test_server_tool_call_detected(self):
        assert _is_server_tool_call({"name": "tool_search_tool", "args": {}}) is True

    def test_regular_tool_call_not_filtered(self):
        assert _is_server_tool_call({"name": "signal_brief", "args": {}}) is False

    def test_empty_name_not_filtered(self):
        assert _is_server_tool_call({"name": "", "args": {}}) is False


class TestBuildSystemMessage:

    def test_without_tool_search(self):
        config = _make_config()
        msg = build_system_message(config)
        assert "tool search" not in msg.content.lower()
        assert "categories" not in msg.content.lower()

    def test_with_tool_search_trading(self):
        config = _make_config(
            name="daily_planner",
            tools=("signal_brief",),
            always_loaded=("signal_brief",),
        )
        msg = build_system_message(config, graph_name="trading")
        assert "Signal & Analysis" in msg.content
        assert "Execution" in msg.content
        assert "tool search" in msg.content.lower()

    def test_with_tool_search_research(self):
        config = _make_config(
            name="quant_researcher",
            tools=("signal_brief",),
            always_loaded=("signal_brief",),
        )
        msg = build_system_message(config, graph_name="research")
        assert "ML Training" in msg.content
        assert "Backtesting" in msg.content

    def test_with_tool_search_supervisor(self):
        config = _make_config(
            name="health_monitor",
            tools=("check_system_status",),
            always_loaded=("check_system_status",),
        )
        msg = build_system_message(config, graph_name="supervisor")
        assert "System Health" in msg.content
        assert "Strategy Lifecycle" in msg.content

    def test_infers_graph_from_agent_name(self):
        config = _make_config(
            name="daily_planner",
            tools=("signal_brief",),
            always_loaded=("signal_brief",),
        )
        msg = build_system_message(config)  # No explicit graph_name
        assert "Signal & Analysis" in msg.content

    def test_categories_vary_by_graph(self):
        config = _make_config(
            name="test",
            tools=("signal_brief",),
            always_loaded=("signal_brief",),
        )
        trading_msg = build_system_message(config, graph_name="trading")
        research_msg = build_system_message(config, graph_name="research")
        # Trading has Execution, research has ML
        assert "Execution" in trading_msg.content
        assert "Execution" not in research_msg.content
        assert "ML Training" in research_msg.content


class TestRunAgent:

    @pytest.fixture
    def mock_publish(self):
        with patch("quantstack.dashboard.events.publish_event") as mock:
            yield mock

    async def test_executes_deferred_tool(self, mock_publish):
        """run_agent can execute tools that were deferred because the full
        configured tool set is passed as the tools parameter."""
        deferred_tool = _make_fake_tool("compute_risk_metrics")
        config = _make_config(tools=("compute_risk_metrics",))

        mock_llm = AsyncMock()
        # First response: call the deferred tool
        mock_llm.ainvoke.side_effect = [
            _make_ai_response(tool_calls=[
                {"name": "compute_risk_metrics", "args": {"symbol": "AAPL"}, "id": "call_1"},
            ]),
            # Second response: final answer
            _make_ai_response(content='{"risk": "low"}'),
        ]

        result = await run_agent(mock_llm, [deferred_tool], config, "Assess risk")
        assert "risk" in result
        deferred_tool.ainvoke.assert_called_once()

    async def test_skips_server_tool_use(self, mock_publish):
        """server_tool_use entries in tool_calls are skipped."""
        real_tool = _make_fake_tool("signal_brief")
        config = _make_config(tools=("signal_brief",))

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            _make_ai_response(tool_calls=[
                {"name": "tool_search_tool", "args": {}, "id": "srv_1"},
                {"name": "signal_brief", "args": {"symbol": "AAPL"}, "id": "call_1"},
            ]),
            _make_ai_response(content="done"),
        ]

        result = await run_agent(mock_llm, [real_tool], config, "Get signal")
        assert result == "done"
        real_tool.ainvoke.assert_called_once()

    async def test_mid_conversation_error_handling(self, mock_publish):
        """API errors mid-loop return graceful error instead of crashing."""
        tool = _make_fake_tool("signal_brief")
        config = _make_config(tools=("signal_brief",))

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            # Round 0: successful tool call
            _make_ai_response(tool_calls=[
                {"name": "signal_brief", "args": {"symbol": "AAPL"}, "id": "call_1"},
            ]),
            # Round 1: API error
            RuntimeError("beta header rejected"),
        ]

        result = await run_agent(mock_llm, [tool], config, "Get signal")
        parsed = json.loads(result)
        assert parsed["error"] == "agent_executor_mid_conversation_failure"
        assert parsed["round"] == 1

    async def test_first_round_error_propagates(self, mock_publish):
        """First-round API errors propagate (config problem, not transient)."""
        config = _make_config()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("auth failure")

        with pytest.raises(RuntimeError, match="auth failure"):
            await run_agent(mock_llm, [], config, "Test")

    async def test_force_final_with_deferred_results(self, mock_publish):
        """Force-final-answer path works with deferred tool results in history."""
        tool = _make_fake_tool("signal_brief")
        config = _make_config(tools=("signal_brief",))

        # All rounds return tool calls, forcing final answer path
        tool_response = _make_ai_response(tool_calls=[
            {"name": "signal_brief", "args": {"symbol": "AAPL"}, "id": "call_1"},
        ])
        final_response = _make_ai_response(content='{"answer": "forced"}')

        mock_llm = AsyncMock()
        # 2 rounds of tool calls + 1 forced final
        mock_llm.ainvoke.side_effect = [
            tool_response, tool_response, final_response,
        ]

        result = await run_agent(mock_llm, [tool], config, "Test", max_rounds=2)
        assert "forced" in result
