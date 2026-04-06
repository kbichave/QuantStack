"""Integration tests for tool search in the agent executor loop."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from quantstack.graphs.agent_executor import run_agent, build_system_message
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
    t.ainvoke = AsyncMock(return_value=f'{{"result": "from {name}"}}')
    return t


def _make_ai_response(content="", tool_calls=None):
    msg = MagicMock(spec=AIMessage)
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


class TestRunAgentWithDeferredTools:
    """run_agent() must execute any configured tool, even if it was deferred at bind time."""

    @pytest.fixture
    def mock_publish(self):
        with patch("quantstack.dashboard.events.publish_event") as mock:
            yield mock

    async def test_deferred_tool_executes_successfully(self, mock_publish):
        """When the LLM discovers and calls a deferred tool, run_agent() resolves
        it from the full tool map and executes it."""
        always_tool = _make_fake_tool("signal_brief")
        deferred_tool = _make_fake_tool("compute_risk_metrics")
        config = _make_config(
            tools=("signal_brief", "compute_risk_metrics"),
            always_loaded=("signal_brief",),
        )

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            _make_ai_response(tool_calls=[
                {"name": "compute_risk_metrics", "args": {"symbol": "AAPL"}, "id": "call_1"},
            ]),
            _make_ai_response(content='{"risk": "low"}'),
        ]

        result = await run_agent(mock_llm, [always_tool, deferred_tool], config, "Assess risk")
        assert "risk" in result
        deferred_tool.ainvoke.assert_called_once()

    async def test_tool_map_contains_all_configured_tools(self, mock_publish):
        """The tool_map inside run_agent() must contain every tool from the
        agent's 'tools' list, not just always_loaded_tools."""
        tools = [_make_fake_tool(n) for n in ["a", "b", "c", "d"]]
        config = _make_config(
            tools=("a", "b", "c", "d"),
            always_loaded=("a",),
        )

        mock_llm = AsyncMock()
        # Call each tool in sequence
        mock_llm.ainvoke.side_effect = [
            _make_ai_response(tool_calls=[
                {"name": "d", "args": {}, "id": "call_1"},
            ]),
            _make_ai_response(content="done"),
        ]

        result = await run_agent(mock_llm, tools, config, "Test")
        # Tool "d" (deferred) was called successfully
        tools[3].ainvoke.assert_called_once()


class TestServerToolUseFiltering:

    @pytest.fixture
    def mock_publish(self):
        with patch("quantstack.dashboard.events.publish_event") as mock:
            yield mock

    async def test_mixed_server_and_regular_tool_calls(self, mock_publish):
        """When response.tool_calls has both server_tool_use and regular tool_use,
        only regular tool_use blocks are executed."""
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


class TestFallbackPathIntegration:

    def test_fallback_on_bind_failure(self):
        """When bind_tools raises, bind_tools_to_llm falls back to full loading."""
        from quantstack.graphs.tool_binding import bind_tools_to_llm

        config = _make_config(
            tools=("signal_brief",),
            always_loaded=("signal_brief",),
        )
        mock_llm = MagicMock()
        # Make bind() raise to trigger fallback
        mock_llm.bind.side_effect = Exception("beta header rejected")
        mock_llm.bind_tools.return_value = mock_llm

        with patch("quantstack.graphs.tool_binding.get_tools_for_agent_with_search") as mock_search, \
             patch("quantstack.graphs.tool_binding.get_tools_for_agent") as mock_full:
            mock_search.side_effect = Exception("beta header rejected")
            mock_full.return_value = [_make_fake_tool("signal_brief")]

            bound_llm, tools, fallback_mode = bind_tools_to_llm(mock_llm, config)
            assert fallback_mode is True
            assert len(tools) == 1

    def test_fallback_sets_fallback_mode_flag(self):
        """On fallback, fallback_mode=True is returned."""
        from quantstack.graphs.tool_binding import bind_tools_to_llm

        config = _make_config(
            tools=("signal_brief",),
            always_loaded=("signal_brief",),
        )
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        with patch("quantstack.graphs.tool_binding.get_tools_for_agent_with_search") as mock_search, \
             patch("quantstack.graphs.tool_binding.get_tools_for_agent") as mock_full:
            mock_search.side_effect = Exception("API error")
            mock_full.return_value = [_make_fake_tool("signal_brief")]

            _, _, fallback_mode = bind_tools_to_llm(mock_llm, config)
            assert fallback_mode is True
