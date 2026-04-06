"""Tests for the tool layer (Section 05: LangGraph migration).

Validates LLM-facing tools, node-callable functions, and the TOOL_REGISTRY.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src" / "quantstack"


class TestLLMFacingToolContract:
    """Every tool in tools/langchain/ must satisfy the LangChain @tool contract."""

    def _get_tool_modules(self):
        import quantstack.tools.langchain as pkg
        return [
            importlib.import_module(f"{pkg.__name__}.{info.name}")
            for info in pkgutil.iter_modules(pkg.__path__)
            if not info.name.startswith("_")
        ]

    def _get_tools_from_module(self, mod):
        from langchain_core.tools import BaseTool
        tools = []
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, BaseTool):
                tools.append(obj)
        return tools

    def test_all_llm_tools_have_tool_decorator(self):
        """All public callables in tools/langchain/ must be BaseTool instances."""
        from langchain_core.tools import BaseTool
        for mod in self._get_tool_modules():
            tools = self._get_tools_from_module(mod)
            assert len(tools) > 0, f"{mod.__name__} has no BaseTool instances"
            for t in tools:
                assert isinstance(t, BaseTool), f"{t.name} is not a BaseTool"

    def test_all_llm_tools_have_nonempty_description(self):
        """LLM-facing tools need descriptions so the model knows when to call them."""
        for mod in self._get_tool_modules():
            for t in self._get_tools_from_module(mod):
                assert t.description, f"{t.name} has no description"
                assert len(t.description) > 10, f"{t.name} description too short"

    def test_all_llm_tools_are_async(self):
        """LangChain @tool supports async natively. All tools must be async."""
        for mod in self._get_tool_modules():
            for t in self._get_tools_from_module(mod):
                assert t.coroutine is not None or hasattr(t, 'afunc'), (
                    f"{t.name} is not async"
                )


class TestNodeCallableFunctionContract:
    """Every function in tools/functions/ must be a plain async function with type hints."""

    def _get_function_modules(self):
        import quantstack.tools.functions as pkg
        return [
            importlib.import_module(f"{pkg.__name__}.{info.name}")
            for info in pkgutil.iter_modules(pkg.__path__)
            if not info.name.startswith("_")
        ]

    def _get_public_functions(self, mod):
        return [
            (name, obj)
            for name, obj in inspect.getmembers(mod, inspect.isfunction)
            if not name.startswith("_") and obj.__module__ == mod.__name__
        ]

    def test_all_node_functions_are_async(self):
        """Node-callable functions must be coroutine functions."""
        for mod in self._get_function_modules():
            for name, fn in self._get_public_functions(mod):
                assert inspect.iscoroutinefunction(fn), (
                    f"{mod.__name__}.{name} is not async"
                )

    def test_all_node_functions_have_return_annotation(self):
        """Node functions must have return type annotations."""
        for mod in self._get_function_modules():
            for name, fn in self._get_public_functions(mod):
                hints = inspect.get_annotations(fn)
                assert "return" in hints, (
                    f"{mod.__name__}.{name} missing return annotation"
                )

    def test_no_tool_decorator_on_node_functions(self):
        """Node-callable functions must NOT have @tool decorator."""
        from langchain_core.tools import BaseTool
        for mod in self._get_function_modules():
            for name in dir(mod):
                obj = getattr(mod, name)
                assert not isinstance(obj, BaseTool), (
                    f"{mod.__name__}.{name} should not be a BaseTool"
                )


class TestNoCrewAIImports:
    """After migration, no new tool file should reference crewai."""

    def test_no_crewai_imports_in_langchain_tools(self):
        tools_dir = SRC_ROOT / "tools" / "langchain"
        for py_file in tools_dir.glob("*.py"):
            content = py_file.read_text()
            assert "crewai" not in content.lower(), (
                f"{py_file.name} references crewai"
            )

    def test_no_crewai_imports_in_functions(self):
        funcs_dir = SRC_ROOT / "tools" / "functions"
        for py_file in funcs_dir.glob("*.py"):
            content = py_file.read_text()
            assert "crewai" not in content.lower(), (
                f"{py_file.name} references crewai"
            )


class TestToolRegistry:
    """TOOL_REGISTRY maps string names to tool objects."""

    def test_registry_contains_all_yaml_referenced_tools(self):
        """Every tool name referenced in agents.yaml must exist in TOOL_REGISTRY."""
        import yaml
        from quantstack.tools.registry import TOOL_REGISTRY

        yaml_dirs = [
            SRC_ROOT / "graphs" / g / "config" / "agents.yaml"
            for g in ("research", "trading", "supervisor")
        ]
        referenced_tools = set()
        for yaml_path in yaml_dirs:
            if yaml_path.exists():
                with open(yaml_path) as f:
                    configs = yaml.safe_load(f)
                for agent_cfg in configs.values():
                    for tool_name in agent_cfg.get("tools", []):
                        referenced_tools.add(tool_name)

        missing = referenced_tools - set(TOOL_REGISTRY.keys())
        assert not missing, f"Missing tools in registry: {missing}"

    def test_registry_values_are_callable(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        for name, tool_obj in TOOL_REGISTRY.items():
            # LangChain tools have .invoke() method but may not be directly callable
            assert callable(tool_obj) or hasattr(tool_obj, "invoke"), (
                f"{name} is not callable and has no invoke method"
            )

    def test_signal_brief_in_registry(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        assert "signal_brief" in TOOL_REGISTRY
        assert "multi_signal_brief" in TOOL_REGISTRY

    def test_risk_tools_in_registry(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        assert "compute_risk_metrics" in TOOL_REGISTRY

    def test_data_tools_in_registry(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        assert "fetch_market_data" in TOOL_REGISTRY
        assert "fetch_earnings_data" in TOOL_REGISTRY

    def test_get_tools_for_agent(self):
        from quantstack.tools.registry import get_tools_for_agent
        tools = get_tools_for_agent(["signal_brief", "fetch_portfolio"])
        assert len(tools) == 2

    def test_get_tools_for_agent_raises_on_missing(self):
        from quantstack.tools.registry import get_tools_for_agent
        with pytest.raises(KeyError, match="nonexistent_tool"):
            get_tools_for_agent(["signal_brief", "nonexistent_tool"])

    def test_registry_count(self):
        """Registry should have at least 18 tools (all YAML-referenced ones)."""
        from quantstack.tools.registry import TOOL_REGISTRY
        assert len(TOOL_REGISTRY) >= 18
