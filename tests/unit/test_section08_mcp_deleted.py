"""Post-deletion validation tests for Section 8.

These tests verify the MCP infrastructure has been completely removed
and the remaining system still functions.
"""

import importlib
import os
import subprocess

import pytest

SRC_ROOT = os.path.join("src", "quantstack")


def test_mcp_module_not_importable():
    """After deletion, `import quantstack.mcp` must raise ImportError."""
    with pytest.raises(ImportError):
        importlib.import_module("quantstack.mcp")


def test_mcp_bridge_not_importable():
    """After deletion, `import quantstack.tools.mcp_bridge` must raise ImportError."""
    with pytest.raises(ImportError):
        importlib.import_module("quantstack.tools.mcp_bridge")


def test_no_mcp_directory_exists():
    """The src/quantstack/mcp/ directory must not exist."""
    mcp_dir = os.path.join(SRC_ROOT, "mcp")
    assert not os.path.exists(mcp_dir), f"{mcp_dir} still exists"


def test_no_mcp_bridge_directory_exists():
    """The src/quantstack/tools/mcp_bridge/ directory must not exist."""
    bridge_dir = os.path.join(SRC_ROOT, "tools", "mcp_bridge")
    assert not os.path.exists(bridge_dir), f"{bridge_dir} still exists"


def test_core_mcp_server_deleted():
    """src/quantstack/core/mcp_server.py must not exist."""
    path = os.path.join(SRC_ROOT, "core", "mcp_server.py")
    assert not os.path.exists(path), f"{path} still exists"


def test_no_fastmcp_in_pyproject():
    """pyproject.toml must not reference fastmcp as a dependency."""
    with open("pyproject.toml") as f:
        content = f.read()
    assert "fastmcp" not in content.lower(), "fastmcp still in pyproject.toml"


def test_no_mcp_entry_points_in_pyproject():
    """pyproject.toml must not contain MCP server entry points."""
    with open("pyproject.toml") as f:
        content = f.read()
    assert "quantstack-mcp" not in content, "quantstack-mcp entry point still in pyproject.toml"


def test_no_mcp_imports_in_source():
    """No Python file in src/quantstack/ should import from quantstack.mcp."""
    result = subprocess.run(
        ["grep", "-r", "from quantstack.mcp", SRC_ROOT],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", (
        f"Remaining MCP imports found:\n{result.stdout}"
    )


def test_no_bridge_references_in_source():
    """No source file should reference get_bridge, MCPBridge, or call_quantcore."""
    result = subprocess.run(
        ["grep", "-rE", "get_bridge|MCPBridge|call_quantcore", SRC_ROOT],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", (
        f"Remaining bridge references found:\n{result.stdout}"
    )


def test_no_fastmcp_references_in_source():
    """No source file should reference fastmcp or FastMCP."""
    result = subprocess.run(
        ["grep", "-rE", "fastmcp|FastMCP", SRC_ROOT],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", (
        f"Remaining FastMCP references found:\n{result.stdout}"
    )


def test_tools_registry_still_importable():
    """The tool registry must still be importable after MCP deletion."""
    mod = importlib.import_module("quantstack.tools.registry")
    assert hasattr(mod, "TOOL_REGISTRY")


def test_trading_graph_still_importable():
    """The trading graph must still build after MCP deletion."""
    importlib.import_module("quantstack.graphs.trading.graph")
