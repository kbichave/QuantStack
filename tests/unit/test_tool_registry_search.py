"""Tests for get_tools_for_agent_with_search() (Section 04)."""

import pytest
from langchain_core.tools import BaseTool

from quantstack.graphs.tool_search_compat import TOOL_SEARCH_TOOL
from quantstack.tools.registry import get_tools_for_agent_with_search


# Use real tool names from the registry
SAMPLE_TOOLS = ("signal_brief", "fetch_portfolio", "compute_risk_metrics")


class TestGetToolsForAgentWithSearch:

    def test_always_loaded_subset_validation(self):
        """Raises ValueError when always_loaded contains a name not in tool_names."""
        with pytest.raises(ValueError, match="always_loaded.*not in tool_names"):
            get_tools_for_agent_with_search(
                tool_names=["signal_brief"],
                always_loaded=["fetch_portfolio"],
            )

    def test_defer_loading_flags_correct(self):
        """Always-loaded tools have no defer_loading; others have defer_loading=True."""
        api_tools, _ = get_tools_for_agent_with_search(
            tool_names=SAMPLE_TOOLS,
            always_loaded=["signal_brief"],
        )

        tool_dicts = {t["name"]: t for t in api_tools if "name" in t and t.get("type") != TOOL_SEARCH_TOOL["type"]}
        assert tool_dicts["signal_brief"].get("defer_loading") is not True
        assert tool_dicts["fetch_portfolio"]["defer_loading"] is True
        assert tool_dicts["compute_risk_metrics"]["defer_loading"] is True

    def test_bm25_search_tool_included(self):
        """tools_for_api includes the BM25 search tool definition."""
        api_tools, _ = get_tools_for_agent_with_search(
            tool_names=SAMPLE_TOOLS,
            always_loaded=["signal_brief"],
        )
        search_tools = [t for t in api_tools if t.get("type") == TOOL_SEARCH_TOOL["type"]]
        assert len(search_tools) == 1
        assert search_tools[0]["name"] == TOOL_SEARCH_TOOL["name"]

    def test_tools_for_execution_contains_all(self):
        """tools_for_execution has all configured tools as BaseTool instances."""
        _, exec_tools = get_tools_for_agent_with_search(
            tool_names=SAMPLE_TOOLS,
            always_loaded=["signal_brief"],
        )
        assert len(exec_tools) == len(SAMPLE_TOOLS)
        for t in exec_tools:
            assert isinstance(t, BaseTool)
        exec_names = {t.name for t in exec_tools}
        for name in SAMPLE_TOOLS:
            assert name in exec_names

    def test_unknown_tool_raises_key_error(self):
        """Unknown tool name raises KeyError (preserves get_tools_for_agent behavior)."""
        with pytest.raises(KeyError, match="nonexistent_tool_xyz"):
            get_tools_for_agent_with_search(
                tool_names=["nonexistent_tool_xyz"],
                always_loaded=[],
            )

    def test_empty_always_loaded_defers_all(self):
        """When always_loaded is empty, every tool has defer_loading=True."""
        api_tools, _ = get_tools_for_agent_with_search(
            tool_names=SAMPLE_TOOLS,
            always_loaded=[],
        )
        regular_tools = [t for t in api_tools if t.get("type") != TOOL_SEARCH_TOOL["type"]]
        for t in regular_tools:
            assert t["defer_loading"] is True, f"{t['name']} should be deferred"

    def test_all_always_loaded_defers_none(self):
        """When all tools are always-loaded, none have defer_loading=True."""
        api_tools, _ = get_tools_for_agent_with_search(
            tool_names=SAMPLE_TOOLS,
            always_loaded=list(SAMPLE_TOOLS),
        )
        regular_tools = [t for t in api_tools if t.get("type") != TOOL_SEARCH_TOOL["type"]]
        for t in regular_tools:
            assert t.get("defer_loading") is not True, f"{t['name']} should not be deferred"

    def test_api_tool_dicts_have_required_fields(self):
        """Each API tool dict has name, description, and input_schema."""
        api_tools, _ = get_tools_for_agent_with_search(
            tool_names=SAMPLE_TOOLS,
            always_loaded=["signal_brief"],
        )
        for t in api_tools:
            if t.get("type") == TOOL_SEARCH_TOOL["type"]:
                continue
            assert "name" in t
            assert "description" in t
            assert "input_schema" in t
