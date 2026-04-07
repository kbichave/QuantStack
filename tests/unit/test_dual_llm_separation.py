# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests verifying dual-LLM separation: research agents cannot access execution tools.

The trading system's safety model requires that research agents (which process
external data and generate hypotheses) never have the ability to execute trades.
Execution tools are reserved for the trading graph's executor agent.
"""

import yaml
import pytest
from pathlib import Path

# Execution tools that must NEVER appear in research agent configs.
# These tools submit orders, close positions, or modify live broker state.
EXECUTION_TOOLS = frozenset({
    "execute_order",
    "execute_options_trade",
    "close_position",
    "update_position_stops",
})

# Broader set: execution-adjacent tools that research agents should not need.
EXECUTION_ADJACENT_TOOLS = EXECUTION_TOOLS | frozenset({
    "check_broker_connection",
    "create_exit_signal",
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_AGENTS_YAML = PROJECT_ROOT / "src" / "quantstack" / "graphs" / "research" / "config" / "agents.yaml"
TRADING_AGENTS_YAML = PROJECT_ROOT / "src" / "quantstack" / "graphs" / "trading" / "config" / "agents.yaml"


def _load_agents(yaml_path: Path) -> dict:
    """Load agent configs from a YAML file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def research_agents() -> dict:
    return _load_agents(RESEARCH_AGENTS_YAML)


@pytest.fixture(scope="module")
def trading_agents() -> dict:
    return _load_agents(TRADING_AGENTS_YAML)


class TestResearchAgentsHaveNoExecutionTools:
    """Research agents must never be able to submit orders or modify positions."""

    def test_no_execution_tools_in_research_agents(self, research_agents):
        violations = []
        for agent_name, config in research_agents.items():
            tools = set(config.get("tools", []))
            # Also check domain_tool_sets if present (e.g., domain_researcher)
            for domain, domain_tools in config.get("domain_tool_sets", {}).items():
                tools.update(domain_tools)
            overlap = tools & EXECUTION_TOOLS
            if overlap:
                violations.append(f"{agent_name}: {sorted(overlap)}")

        assert not violations, (
            f"Research agents have execution tools (safety violation):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_no_execution_adjacent_tools_in_research_agents(self, research_agents):
        violations = []
        for agent_name, config in research_agents.items():
            tools = set(config.get("tools", []))
            for domain, domain_tools in config.get("domain_tool_sets", {}).items():
                tools.update(domain_tools)
            overlap = tools & EXECUTION_ADJACENT_TOOLS
            if overlap:
                violations.append(f"{agent_name}: {sorted(overlap)}")

        assert not violations, (
            f"Research agents have execution-adjacent tools:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_no_execution_tools_in_always_loaded(self, research_agents):
        """always_loaded_tools is a subset of tools, but verify separately."""
        violations = []
        for agent_name, config in research_agents.items():
            always_loaded = set(config.get("always_loaded_tools", []))
            overlap = always_loaded & EXECUTION_TOOLS
            if overlap:
                violations.append(f"{agent_name}: {sorted(overlap)}")

        assert not violations, (
            f"Research agents have execution tools in always_loaded_tools:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


class TestTradingAgentsCanHaveExecutionTools:
    """The executor agent in the trading graph legitimately needs execution tools."""

    def test_executor_has_execution_tools(self, trading_agents):
        executor = trading_agents.get("executor")
        assert executor is not None, "executor agent missing from trading config"
        tools = set(executor.get("tools", []))
        assert EXECUTION_TOOLS & tools, (
            "executor agent should have at least one execution tool"
        )

    def test_non_executor_trading_agents_lack_order_tools(self, trading_agents):
        """Trading agents other than executor should not have execute_order."""
        order_tools = frozenset({"execute_order", "execute_options_trade"})
        violations = []
        for agent_name, config in trading_agents.items():
            if agent_name == "executor":
                continue
            tools = set(config.get("tools", []))
            overlap = tools & order_tools
            if overlap:
                violations.append(f"{agent_name}: {sorted(overlap)}")

        assert not violations, (
            f"Non-executor trading agents have order submission tools:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


class TestToolRegistrySeparation:
    """Verify the tool registry correctly registers execution tools."""

    def test_execution_tools_exist_in_registry(self):
        from quantstack.tools.registry import TOOL_REGISTRY

        for tool_name in EXECUTION_TOOLS:
            assert tool_name in TOOL_REGISTRY, (
                f"Execution tool '{tool_name}' missing from TOOL_REGISTRY"
            )

    def test_get_tools_for_agent_rejects_unknown(self):
        from quantstack.tools.registry import get_tools_for_agent

        with pytest.raises(KeyError):
            get_tools_for_agent(["nonexistent_tool_xyz"])
