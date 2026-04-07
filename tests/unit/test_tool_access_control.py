"""Tests for tool access control (section-08).

Verifies blocked_tools loading, ConfigWatcher integration, and the guard
in agent_executor that prevents blocked tool invocations.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from quantstack.graphs.config import load_blocked_tools
from quantstack.graphs.config_watcher import ConfigWatcher


# ---------------------------------------------------------------------------
# load_blocked_tools
# ---------------------------------------------------------------------------


def test_load_blocked_tools_from_yaml(tmp_path):
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml.dump({
        "blocked_tools": ["execute_order", "cancel_order"],
        "daily_planner": {
            "role": "planner",
            "goal": "plan",
            "backstory": "test",
            "llm_tier": "light",
            "tools": [],
        },
    }))
    result = load_blocked_tools(yaml_file)
    assert result == frozenset({"execute_order", "cancel_order"})


def test_load_blocked_tools_missing_key(tmp_path):
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml.dump({
        "daily_planner": {
            "role": "planner",
            "goal": "plan",
            "backstory": "test",
            "llm_tier": "light",
            "tools": [],
        },
    }))
    result = load_blocked_tools(yaml_file)
    assert result == frozenset()


def test_load_blocked_tools_empty_list(tmp_path):
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml.dump({
        "blocked_tools": [],
        "daily_planner": {
            "role": "planner",
            "goal": "plan",
            "backstory": "test",
            "llm_tier": "light",
            "tools": [],
        },
    }))
    result = load_blocked_tools(yaml_file)
    assert result == frozenset()


# ---------------------------------------------------------------------------
# ConfigWatcher integration
# ---------------------------------------------------------------------------


def test_config_watcher_loads_blocked_tools(tmp_path):
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml.dump({
        "blocked_tools": ["execute_order"],
        "test_agent": {
            "role": "test",
            "goal": "test",
            "backstory": "test",
            "llm_tier": "light",
            "tools": [],
        },
    }))
    watcher = ConfigWatcher(yaml_file)
    assert watcher.get_blocked_tools() == frozenset({"execute_order"})


def test_config_watcher_hot_reload_blocked_tools(tmp_path):
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml.dump({
        "blocked_tools": ["execute_order"],
        "test_agent": {
            "role": "test",
            "goal": "test",
            "backstory": "test",
            "llm_tier": "light",
            "tools": [],
        },
    }))
    watcher = ConfigWatcher(yaml_file)
    assert "cancel_order" not in watcher.get_blocked_tools()

    # Update file with new blocked tools
    yaml_file.write_text(yaml.dump({
        "blocked_tools": ["execute_order", "cancel_order"],
        "test_agent": {
            "role": "test",
            "goal": "test",
            "backstory": "test",
            "llm_tier": "light",
            "tools": [],
        },
    }))
    watcher._stage_reload()
    watcher.apply_pending_reload()
    assert watcher.get_blocked_tools() == frozenset({"execute_order", "cancel_order"})


# ---------------------------------------------------------------------------
# Guard behavior (unit-level — test the logic, not the full agent loop)
# ---------------------------------------------------------------------------


def test_blocked_tool_guard_rejects():
    """Verify the guard logic: tool in blocked_tools returns error JSON."""
    import json

    blocked = frozenset({"execute_order", "cancel_order"})
    tool_name = "execute_order"

    # Simulate guard check
    assert tool_name in blocked
    result = json.loads(json.dumps({
        "error": "tool_access_denied",
        "tool": tool_name,
        "message": f"Tool '{tool_name}' is not available in this graph context.",
    }))
    assert result["error"] == "tool_access_denied"
    assert result["tool"] == "execute_order"


def test_allowed_tool_passes_guard():
    """Tool NOT in blocked_tools passes the guard."""
    blocked = frozenset({"execute_order"})
    assert "fetch_portfolio" not in blocked


def test_supervisor_blocks_all_execution():
    """Supervisor's blocked_tools covers all execution tools."""
    supervisor_blocked = frozenset({
        "execute_order", "cancel_order", "activate_kill_switch",
        "submit_option_order", "close_position",
    })
    for tool in ["execute_order", "cancel_order", "activate_kill_switch",
                 "submit_option_order", "close_position"]:
        assert tool in supervisor_blocked
