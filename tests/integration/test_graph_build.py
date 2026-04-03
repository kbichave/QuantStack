"""Integration tests for graph construction after MCP removal.

Verifies that all three StateGraphs can be imported without errors, confirming
the full import chain: graph builder -> node factories -> tool imports -> registry.
"""

import importlib


class TestGraphBuildSmoke:

    def test_trading_graph_importable(self):
        mod = importlib.import_module("quantstack.graphs.trading.graph")
        assert hasattr(mod, "build_trading_graph")

    def test_research_graph_importable(self):
        mod = importlib.import_module("quantstack.graphs.research.graph")
        assert hasattr(mod, "build_research_graph")

    def test_supervisor_graph_importable(self):
        mod = importlib.import_module("quantstack.graphs.supervisor.graph")
        assert hasattr(mod, "build_supervisor_graph")


class TestGraphToolBindings:

    def test_trading_agent_tools_resolve(self):
        import yaml
        from quantstack.tools.registry import get_tools_for_agent

        yaml_path = "src/quantstack/graphs/trading/config/agents.yaml"
        with open(yaml_path) as f:
            configs = yaml.safe_load(f) or {}

        for agent, cfg in configs.items():
            if not isinstance(cfg, dict):
                continue
            tool_names = cfg.get("tools", [])
            if tool_names:
                tools = get_tools_for_agent(tool_names)
                assert len(tools) == len(tool_names), (
                    f"Trading agent '{agent}' tool count mismatch"
                )

    def test_research_agent_tools_resolve(self):
        import yaml
        from quantstack.tools.registry import get_tools_for_agent

        yaml_path = "src/quantstack/graphs/research/config/agents.yaml"
        with open(yaml_path) as f:
            configs = yaml.safe_load(f) or {}

        for agent, cfg in configs.items():
            if not isinstance(cfg, dict):
                continue
            tool_names = cfg.get("tools", [])
            if tool_names:
                tools = get_tools_for_agent(tool_names)
                assert len(tools) == len(tool_names)

    def test_supervisor_agent_tools_resolve(self):
        import yaml
        from quantstack.tools.registry import get_tools_for_agent

        yaml_path = "src/quantstack/graphs/supervisor/config/agents.yaml"
        with open(yaml_path) as f:
            configs = yaml.safe_load(f) or {}

        for agent, cfg in configs.items():
            if not isinstance(cfg, dict):
                continue
            tool_names = cfg.get("tools", [])
            if tool_names:
                tools = get_tools_for_agent(tool_names)
                assert len(tools) == len(tool_names)
