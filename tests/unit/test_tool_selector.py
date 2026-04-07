"""Tests for the tool-binding selector."""

from __future__ import annotations

from quantstack.meta.tool_selector import (
    find_high_demand_deferred,
    find_unused_tools,
    recommend_tool_changes,
)


def test_never_used_tools_removed():
    usage = {"tool_a": 50, "tool_b": 0, "tool_c": 0}
    bound = ["tool_a", "tool_b", "tool_c"]
    unused = find_unused_tools("agent_x", usage, bound)
    assert "tool_b" in unused
    assert "tool_c" in unused
    assert "tool_a" not in unused


def test_frequently_searched_deferred_promoted():
    search_log = {"deferred_x": 12, "deferred_y": 3, "tool_a": 5}
    bound = ["tool_a"]
    promoted = find_high_demand_deferred("agent_x", search_log, bound)
    assert "deferred_x" in promoted
    assert "deferred_y" in promoted
    # Already-bound tool should NOT appear in the add list.
    assert "tool_a" not in promoted


def test_tool_changes_output_structure():
    usage = {"tool_a": 10, "tool_b": 0}
    search_log = {"new_tool": 5}
    bound = ["tool_a", "tool_b"]
    result = recommend_tool_changes("agent_x", usage, search_log, bound)
    assert "add" in result
    assert "remove" in result
    assert isinstance(result["add"], list)
    assert isinstance(result["remove"], list)
    assert "new_tool" in result["add"]
    assert "tool_b" in result["remove"]
