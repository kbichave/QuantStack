"""Tests for consolidated bind_tools_to_llm (Section 05)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from quantstack.graphs.config import AgentConfig


def _make_config(tools=(), always_loaded=()):
    return AgentConfig(
        name="test_agent",
        role="Test",
        goal="Test",
        backstory="Test",
        llm_tier="heavy",
        tools=tools,
        always_loaded_tools=always_loaded,
    )


def _make_fake_tool(name: str) -> MagicMock:
    t = MagicMock(spec=BaseTool)
    t.name = name
    return t


class TestBindToolsToLlm:

    def test_backward_compat_no_always_loaded(self):
        """When always_loaded_tools is empty, loads all tools via bind_tools."""
        from quantstack.graphs.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        fake_tools = [_make_fake_tool("signal_brief"), _make_fake_tool("fetch_portfolio")]

        config = _make_config(tools=("signal_brief", "fetch_portfolio"))

        with patch("quantstack.graphs.tool_binding.get_tools_for_agent", return_value=fake_tools):
            bound_llm, tools, fallback = bind_tools_to_llm(mock_llm, config)

        mock_llm.bind_tools.assert_called_once_with(fake_tools)
        assert len(tools) == 2
        assert fallback is False

    def test_no_tools_returns_immediately(self):
        """When config.tools is empty, returns (llm, [], False) with no binding."""
        from quantstack.graphs.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        config = _make_config(tools=())

        bound_llm, tools, fallback = bind_tools_to_llm(mock_llm, config)

        assert bound_llm is mock_llm
        assert tools == []
        assert fallback is False
        mock_llm.bind_tools.assert_not_called()

    def test_deferred_loading_path(self):
        """When always_loaded_tools is set, uses get_tools_for_agent_with_search."""
        from quantstack.graphs.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        mock_llm.bind.return_value = mock_llm
        fake_api_tools = [{"name": "signal_brief"}, {"type": "tool_search_bm25_2025_04_15"}]
        fake_exec_tools = [_make_fake_tool("signal_brief"), _make_fake_tool("fetch_portfolio")]

        config = _make_config(
            tools=("signal_brief", "fetch_portfolio"),
            always_loaded=("signal_brief",),
        )

        with patch(
            "quantstack.graphs.tool_binding.get_tools_for_agent_with_search",
            return_value=(fake_api_tools, fake_exec_tools),
        ):
            bound_llm, tools, fallback = bind_tools_to_llm(mock_llm, config)

        # Uses llm.bind() not bind_tools() for deferred path
        mock_llm.bind.assert_called_once_with(tools=fake_api_tools)
        assert tools == fake_exec_tools
        assert len(tools) == 2  # All configured tools, not just always-loaded
        assert fallback is False

    def test_fallback_on_deferred_loading_error(self):
        """When deferred loading fails, falls back to full loading."""
        from quantstack.graphs.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        fake_tools = [_make_fake_tool("signal_brief")]

        config = _make_config(
            tools=("signal_brief",),
            always_loaded=("signal_brief",),
        )

        with (
            patch(
                "quantstack.graphs.tool_binding.get_tools_for_agent_with_search",
                side_effect=RuntimeError("API unavailable"),
            ),
            patch(
                "quantstack.graphs.tool_binding.get_tools_for_agent",
                return_value=fake_tools,
            ),
        ):
            bound_llm, tools, fallback = bind_tools_to_llm(mock_llm, config)

        assert fallback is True
        assert len(tools) == 1
        mock_llm.bind_tools.assert_called_once_with(fake_tools)

    def test_fallback_on_registry_error(self):
        """When get_tools_for_agent_with_search raises, falls back."""
        from quantstack.graphs.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        fake_tools = [_make_fake_tool("signal_brief")]

        config = _make_config(
            tools=("signal_brief",),
            always_loaded=("signal_brief",),
        )

        with (
            patch(
                "quantstack.graphs.tool_binding.get_tools_for_agent_with_search",
                side_effect=KeyError("unknown tool"),
            ),
            patch(
                "quantstack.graphs.tool_binding.get_tools_for_agent",
                return_value=fake_tools,
            ),
        ):
            bound_llm, tools, fallback = bind_tools_to_llm(mock_llm, config)

        assert fallback is True

    def test_backward_compat_error_returns_no_tools(self):
        """When backward-compat path itself fails, returns (llm, [])."""
        from quantstack.graphs.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        config = _make_config(tools=("bad_tool",))

        with patch(
            "quantstack.graphs.tool_binding.get_tools_for_agent",
            side_effect=KeyError("bad_tool not found"),
        ):
            bound_llm, tools, fallback = bind_tools_to_llm(mock_llm, config)

        assert bound_llm is mock_llm
        assert tools == []

    def test_graph_files_no_local_bind_tools(self):
        """No graph file defines a local _bind_tools_to_llm anymore."""
        import quantstack.graphs.trading.graph as trading
        import quantstack.graphs.research.graph as research
        import quantstack.graphs.supervisor.graph as supervisor

        for module in [trading, research, supervisor]:
            assert not hasattr(module, "_bind_tools_to_llm"), (
                f"{module.__name__} still has local _bind_tools_to_llm"
            )
