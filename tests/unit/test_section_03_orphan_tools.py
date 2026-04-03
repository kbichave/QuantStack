"""Tests for Section 3: Triage orphan MCP tools and migrate live ones."""

import importlib
import pathlib

import pytest


def test_no_bridge_imports_in_any_langchain_tool():
    """No langchain tool module should import from mcp_bridge."""
    langchain_dir = pathlib.Path("src/quantstack/tools/langchain")
    for py_file in langchain_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        text = py_file.read_text()
        assert "mcp_bridge" not in text, f"{py_file.name} still imports mcp_bridge"
        assert "get_bridge" not in text, f"{py_file.name} still uses get_bridge"


def test_system_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.system_tools")
    assert hasattr(mod, "check_system_status")
    assert hasattr(mod, "check_heartbeat")


EXPECTED_NEW_MODULES = [
    "quantstack.tools.langchain.system_tools",
    "quantstack.tools.langchain.alert_tools",
    "quantstack.tools.langchain.attribution_tools",
    "quantstack.tools.langchain.capitulation_tools",
    "quantstack.tools.langchain.cross_domain_tools",
    "quantstack.tools.langchain.decoder_tools",
    "quantstack.tools.langchain.feedback_tools",
    "quantstack.tools.langchain.institutional_tools",
    "quantstack.tools.langchain.intraday_tools",
    "quantstack.tools.langchain.macro_tools",
    "quantstack.tools.langchain.nlp_tools",
    "quantstack.tools.langchain.options_execution_tools",
    "quantstack.tools.langchain.meta_tools",
    "quantstack.tools.langchain.finrl_tools",
    "quantstack.tools.langchain.qc_acquisition_tools",
    "quantstack.tools.langchain.qc_backtesting_tools",
    "quantstack.tools.langchain.qc_indicator_tools",
    "quantstack.tools.langchain.qc_research_tools",
]


@pytest.mark.parametrize("mod_name", EXPECTED_NEW_MODULES)
def test_new_modules_importable(mod_name):
    importlib.import_module(mod_name)


def test_no_tool_name_collisions():
    from quantstack.tools.registry import TOOL_REGISTRY
    assert len(TOOL_REGISTRY) == len(set(TOOL_REGISTRY.keys()))


def test_registry_has_minimum_tools():
    from quantstack.tools.registry import TOOL_REGISTRY
    assert len(TOOL_REGISTRY) >= 80, f"Expected >=80 tools, got {len(TOOL_REGISTRY)}"


def test_registry_has_system_tools():
    from quantstack.tools.registry import TOOL_REGISTRY
    assert "check_system_status" in TOOL_REGISTRY
    assert "check_heartbeat" in TOOL_REGISTRY


def test_registry_defines_no_inline_tools():
    """registry.py must not define any @tool functions inline."""
    import ast
    registry_path = pathlib.Path("src/quantstack/tools/registry.py")
    tree = ast.parse(registry_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # _try_import and get_tools_for_agent are helpers, not @tool definitions
            if node.name not in ("_try_import", "get_tools_for_agent"):
                assert False, f"registry.py defines function '{node.name}' — move it to a langchain module"
