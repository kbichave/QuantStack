"""Integration tests for TOOL_REGISTRY completeness and agent binding.

Validates that the registry is self-consistent, that every agent config
resolves, and that no bridge/MCP references remain in the tool layer.
"""

import pathlib

import pytest
import yaml

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "quantstack"


class TestToolRegistryCompleteness:

    def test_registry_is_nonempty(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        assert len(TOOL_REGISTRY) > 0

    def test_all_tools_are_base_tool_instances(self):
        from langchain_core.tools import BaseTool
        from quantstack.tools.registry import TOOL_REGISTRY

        for name, tool_obj in TOOL_REGISTRY.items():
            assert isinstance(tool_obj, BaseTool), (
                f"Tool '{name}' is {type(tool_obj)}, expected BaseTool"
            )

    def test_no_duplicate_tool_names(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        assert len(TOOL_REGISTRY) == len(set(TOOL_REGISTRY.keys()))

    def test_all_tools_have_descriptions(self):
        from quantstack.tools.registry import TOOL_REGISTRY

        for name, tool_obj in TOOL_REGISTRY.items():
            assert tool_obj.description and len(tool_obj.description.strip()) > 0, (
                f"Tool '{name}' has empty description"
            )

    def test_all_tools_are_async(self):
        from quantstack.tools.registry import TOOL_REGISTRY

        for name, tool_obj in TOOL_REGISTRY.items():
            assert tool_obj.coroutine is not None, (
                f"Tool '{name}' is synchronous — must be async for run_agent()"
            )


class TestAgentBindingResolution:

    AGENT_YAMLS = [
        SRC_ROOT / "graphs" / "trading" / "config" / "agents.yaml",
        SRC_ROOT / "graphs" / "research" / "config" / "agents.yaml",
        SRC_ROOT / "graphs" / "supervisor" / "config" / "agents.yaml",
    ]

    @pytest.fixture
    def all_agent_configs(self):
        configs = {}
        for yaml_path in self.AGENT_YAMLS:
            if yaml_path.exists():
                with open(yaml_path) as f:
                    data = yaml.safe_load(f) or {}
                for agent_name, agent_cfg in data.items():
                    if isinstance(agent_cfg, dict):
                        configs[f"{yaml_path.parent.parent.name}/{agent_name}"] = agent_cfg
        return configs

    def test_all_agent_tool_names_resolve(self, all_agent_configs):
        from quantstack.tools.registry import get_tools_for_agent

        failures = []
        for agent_key, cfg in all_agent_configs.items():
            tool_names = cfg.get("tools", [])
            if not tool_names:
                continue
            try:
                tools = get_tools_for_agent(tool_names)
                assert len(tools) == len(tool_names)
            except KeyError as exc:
                failures.append(f"{agent_key}: {exc}")

        assert not failures, "Agent binding failures:\n" + "\n".join(failures)

    def test_no_duplicate_tools_within_agent(self, all_agent_configs):
        duplicates = []
        for agent_key, cfg in all_agent_configs.items():
            tool_names = cfg.get("tools", [])
            if len(tool_names) != len(set(tool_names)):
                seen = set()
                dupes = [t for t in tool_names if t in seen or seen.add(t)]
                duplicates.append(f"{agent_key}: {dupes}")

        assert not duplicates, "Duplicate tools:\n" + "\n".join(duplicates)


class TestNoBridgeOrMCPReferences:

    TOOLS_DIR = SRC_ROOT / "tools"

    def _scan_python_files(self, directory: pathlib.Path, pattern: str) -> list[str]:
        hits = []
        for py_file in directory.rglob("*.py"):
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                if pattern in line and not line.strip().startswith("#"):
                    hits.append(f"{py_file.relative_to(SRC_ROOT)}:{i}: {line.strip()}")
        return hits

    def test_no_get_bridge_in_tools(self):
        hits = self._scan_python_files(self.TOOLS_DIR, "get_bridge")
        assert not hits, "get_bridge references:\n" + "\n".join(hits)

    def test_no_call_quantcore_in_tools(self):
        hits = self._scan_python_files(self.TOOLS_DIR, "call_quantcore")
        assert not hits, "call_quantcore references:\n" + "\n".join(hits)

    def test_no_mcp_imports_in_tools(self):
        hits = self._scan_python_files(self.TOOLS_DIR, "from quantstack.mcp")
        assert not hits, "quantstack.mcp imports:\n" + "\n".join(hits)

    def test_no_mcp_imports_in_graphs(self):
        graphs_dir = SRC_ROOT / "graphs"
        hits = self._scan_python_files(graphs_dir, "from quantstack.mcp")
        assert not hits, "quantstack.mcp imports in graphs/:\n" + "\n".join(hits)
