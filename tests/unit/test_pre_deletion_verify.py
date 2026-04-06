"""Pre-deletion verification tests.

These tests gate Section 8 (MCP deletion). ALL must pass before any MCP code
is removed.
"""

import importlib
import pathlib

import pytest

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "quantstack"


class TestToolRegistryCompleteness:
    def test_registry_count_meets_baseline(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        BASELINE_COUNT = 19  # pre-migration count
        assert len(TOOL_REGISTRY) >= BASELINE_COUNT

    def test_all_registry_tools_are_basetool(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        from langchain_core.tools import BaseTool
        for name, tool in TOOL_REGISTRY.items():
            assert isinstance(tool, BaseTool), f"TOOL_REGISTRY['{name}'] is not a BaseTool"


class TestNoBridgeReferences:
    def test_no_get_bridge_in_tools(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-r", "get_bridge", str(SRC_ROOT / "tools")],
            capture_output=True, text=True,
        )
        # Filter out mcp_bridge directory itself (still exists pre-deletion)
        violations = [
            line for line in result.stdout.strip().splitlines()
            if line and "mcp_bridge" not in line
        ]
        assert violations == [], f"Found get_bridge references in tools/:\n" + "\n".join(violations)

    def test_no_bridge_in_langchain_or_functions(self):
        import subprocess
        for subdir in ["tools/langchain", "tools/functions"]:
            result = subprocess.run(
                ["grep", "-r", "get_bridge\\|call_quantcore\\|MCPBridge", str(SRC_ROOT / subdir)],
                capture_output=True, text=True,
            )
            assert result.stdout.strip() == "", f"Found bridge refs in {subdir}:\n{result.stdout}"


class TestGraphImportSmokeTests:
    def test_trading_graph_importable(self):
        importlib.import_module("quantstack.graphs.trading.graph")

    def test_research_graph_importable(self):
        importlib.import_module("quantstack.graphs.research.graph")

    def test_supervisor_graph_importable(self):
        importlib.import_module("quantstack.graphs.supervisor.graph")


class TestAgentToolBindingResolution:
    @pytest.fixture
    def tool_registry(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        return TOOL_REGISTRY

    @staticmethod
    def _load_agent_configs(yaml_path):
        import yaml
        with open(yaml_path) as f:
            configs = yaml.safe_load(f)
        return {
            name: cfg.get("tools", [])
            for name, cfg in (configs or {}).items()
            if isinstance(cfg, dict)
        }

    def test_trading_agent_tools_resolve(self, tool_registry):
        configs = self._load_agent_configs(SRC_ROOT / "graphs/trading/config/agents.yaml")
        for agent, tools in configs.items():
            for t in tools:
                assert t in tool_registry, f"Trading '{agent}' -> '{t}' not in TOOL_REGISTRY"

    def test_research_agent_tools_resolve(self, tool_registry):
        configs = self._load_agent_configs(SRC_ROOT / "graphs/research/config/agents.yaml")
        for agent, tools in configs.items():
            for t in tools:
                assert t in tool_registry, f"Research '{agent}' -> '{t}' not in TOOL_REGISTRY"

    def test_supervisor_agent_tools_resolve(self, tool_registry):
        configs = self._load_agent_configs(SRC_ROOT / "graphs/supervisor/config/agents.yaml")
        for agent, tools in configs.items():
            for t in tools:
                assert t in tool_registry, f"Supervisor '{agent}' -> '{t}' not in TOOL_REGISTRY"


class TestNoCircularImports:
    def test_tools_do_not_import_graphs(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "from quantstack.graphs", str(SRC_ROOT / "tools"), "--include=*.py"],
            capture_output=True, text=True,
        )
        # tool_search_compat is a shared utility (no graph logic) imported by registry
        allowed = {"tool_search_compat"}
        violations = [
            line for line in result.stdout.strip().splitlines()
            if not any(a in line for a in allowed)
        ]
        assert not violations, f"tools/ imports from graphs/:\n" + "\n".join(violations)
