"""Regression tests: risk_check is never deferred and risk gate behavior unchanged."""

import pathlib

import pytest
import yaml

from quantstack.graphs.config import AgentConfig, load_agent_configs

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "quantstack"

AGENT_YAMLS = [
    SRC_ROOT / "graphs" / "trading" / "config" / "agents.yaml",
    SRC_ROOT / "graphs" / "research" / "config" / "agents.yaml",
    SRC_ROOT / "graphs" / "supervisor" / "config" / "agents.yaml",
]

# Tools that must never be deferred (always-loaded when present)
RISK_TOOLS = {"risk_check", "check_risk_limits"}


class TestRiskCheckNeverDeferred:
    """risk_check / check_risk_limits must appear in always_loaded_tools
    for every agent that has it in tools and uses deferred loading."""

    def test_risk_tools_in_always_loaded_for_all_agents(self):
        """For every agent across all three graphs: if a risk tool is in tools
        and always_loaded_tools is non-empty, the risk tool must also be in
        always_loaded_tools."""
        violations = []
        for yaml_path in AGENT_YAMLS:
            if not yaml_path.exists():
                continue
            configs = load_agent_configs(yaml_path)
            for name, cfg in configs.items():
                if not cfg.always_loaded_tools:
                    continue  # No deferred loading, no risk
                tools_set = set(cfg.tools)
                always_set = set(cfg.always_loaded_tools)
                for risk_tool in RISK_TOOLS:
                    if risk_tool in tools_set and risk_tool not in always_set:
                        violations.append(
                            f"{yaml_path.parent.parent.name}/{name}: "
                            f"'{risk_tool}' in tools but not in always_loaded_tools"
                        )
        assert not violations, (
            f"Risk tools must never be deferred:\n" + "\n".join(violations)
        )

    def test_config_validation_rejects_deferred_risk_check(self):
        """Construct an AgentConfig where risk_check is in tools but NOT in
        always_loaded_tools. Config validation must raise ValueError."""
        with pytest.raises(ValueError, match="risk_check.*never be deferred"):
            AgentConfig(
                name="test_agent",
                role="Tester",
                goal="Test",
                backstory="Test",
                llm_tier="heavy",
                tools=("risk_check", "signal_brief"),
                always_loaded_tools=("signal_brief",),
            )

    def test_config_validation_allows_risk_check_in_always_loaded(self):
        """When risk_check IS in always_loaded_tools, validation passes."""
        cfg = AgentConfig(
            name="test_agent",
            role="Tester",
            goal="Test",
            backstory="Test",
            llm_tier="heavy",
            tools=("risk_check", "signal_brief"),
            always_loaded_tools=("risk_check", "signal_brief"),
        )
        assert "risk_check" in cfg.always_loaded_tools


class TestRiskGateBehaviorUnchanged:
    """Tool search must not alter risk gate behavior."""

    def test_risk_tools_always_in_tool_map(self):
        """When bind_tools_to_llm() is called for agents with risk tools,
        those tools must be in the returned tool list."""
        from unittest.mock import MagicMock, patch

        from langchain_core.tools import BaseTool

        from quantstack.graphs.tool_binding import bind_tools_to_llm

        config = AgentConfig(
            name="risk_analyst",
            role="Risk Analyst",
            goal="Assess risk",
            backstory="You assess risk",
            llm_tier="heavy",
            tools=("check_risk_limits", "compute_risk_metrics", "signal_brief"),
            always_loaded_tools=("check_risk_limits", "compute_risk_metrics"),
        )

        def _make_tool(name):
            t = MagicMock(spec=BaseTool)
            t.name = name
            return t

        all_tools = [_make_tool(n) for n in config.tools]

        with patch("quantstack.graphs.tool_binding.get_tools_for_agent_with_search") as mock_search:
            mock_search.return_value = (
                [{"type": "function", "name": t.name} for t in all_tools],
                all_tools,
            )
            mock_llm = MagicMock()
            mock_llm.bind.return_value = mock_llm

            _, tools, fallback = bind_tools_to_llm(mock_llm, config)

        tool_names = {t.name for t in tools}
        assert "check_risk_limits" in tool_names
        assert "compute_risk_metrics" in tool_names
        assert not fallback

    def test_all_yaml_configs_load_without_error(self):
        """All three agents.yaml files load and pass config validation."""
        for yaml_path in AGENT_YAMLS:
            if not yaml_path.exists():
                continue
            configs = load_agent_configs(yaml_path)
            assert len(configs) > 0, f"No agents loaded from {yaml_path}"
