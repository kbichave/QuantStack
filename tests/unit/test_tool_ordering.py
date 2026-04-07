"""Tests for deterministic tool ordering in registry.py."""

from unittest.mock import patch, MagicMock

from langchain_core.tools import BaseTool


def _make_mock_tool(name: str) -> BaseTool:
    """Create a mock BaseTool with a given name."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.description = f"Description for {name}"
    return tool


def _make_registry(names: list[str]) -> dict[str, BaseTool]:
    return {n: _make_mock_tool(n) for n in names}


_UNSORTED_NAMES = ["zebra_tool", "alpha_tool", "mid_tool"]
_SORTED_NAMES = ["alpha_tool", "mid_tool", "zebra_tool"]


class TestGetToolsForAgent:

    @patch("quantstack.tools.registry.TOOL_REGISTRY", _make_registry(_UNSORTED_NAMES))
    def test_returns_tools_sorted_by_name(self):
        from quantstack.tools.registry import get_tools_for_agent

        result = get_tools_for_agent(_UNSORTED_NAMES)
        names = [t.name for t in result]
        assert names == _SORTED_NAMES

    @patch("quantstack.tools.registry.TOOL_REGISTRY", _make_registry(_UNSORTED_NAMES))
    def test_sorting_is_stable_across_calls(self):
        from quantstack.tools.registry import get_tools_for_agent

        result1 = [t.name for t in get_tools_for_agent(_UNSORTED_NAMES)]
        result2 = [t.name for t in get_tools_for_agent(_UNSORTED_NAMES)]
        assert result1 == result2 == _SORTED_NAMES


class TestGetToolsForAgentWithSearch:

    @patch("quantstack.tools.registry.tool_to_anthropic_dict")
    @patch("quantstack.tools.registry.TOOL_REGISTRY", _make_registry(_UNSORTED_NAMES))
    def test_tools_for_api_sorted_by_name(self, mock_convert):
        from quantstack.tools.registry import get_tools_for_agent_with_search

        mock_convert.side_effect = lambda t, defer=False: {
            "name": t.name,
            "defer": defer,
        }

        tools_for_api, tools_for_execution = get_tools_for_agent_with_search(
            _UNSORTED_NAMES, always_loaded=["alpha_tool"]
        )

        # Exclude TOOL_SEARCH_TOOL (last item) from name check
        api_names = [t["name"] for t in tools_for_api[:-1]]
        assert api_names == _SORTED_NAMES

        exec_names = [t.name for t in tools_for_execution]
        assert exec_names == _SORTED_NAMES

    @patch("quantstack.tools.registry.tool_to_anthropic_dict")
    @patch("quantstack.tools.registry.TOOL_REGISTRY", _make_registry(_UNSORTED_NAMES))
    def test_tool_search_tool_remains_last(self, mock_convert):
        from quantstack.tools.registry import get_tools_for_agent_with_search, TOOL_SEARCH_TOOL

        mock_convert.side_effect = lambda t, defer=False: {
            "name": t.name,
            "defer": defer,
        }

        tools_for_api, _ = get_tools_for_agent_with_search(
            _UNSORTED_NAMES, always_loaded=["alpha_tool"]
        )

        assert tools_for_api[-1] is TOOL_SEARCH_TOOL

    @patch("quantstack.tools.registry.tool_to_anthropic_dict")
    @patch("quantstack.tools.registry.TOOL_REGISTRY", _make_registry(_UNSORTED_NAMES))
    def test_tool_search_not_sorted_into_middle(self, mock_convert):
        from quantstack.tools.registry import get_tools_for_agent_with_search, TOOL_SEARCH_TOOL

        mock_convert.side_effect = lambda t, defer=False: {
            "name": t.name,
            "defer": defer,
        }

        tools_for_api, _ = get_tools_for_agent_with_search(
            _UNSORTED_NAMES, always_loaded=["alpha_tool"]
        )

        # TOOL_SEARCH_TOOL should only appear once, at the end
        search_tool_indices = [
            i for i, t in enumerate(tools_for_api) if t is TOOL_SEARCH_TOOL
        ]
        assert search_tool_indices == [len(tools_for_api) - 1]
