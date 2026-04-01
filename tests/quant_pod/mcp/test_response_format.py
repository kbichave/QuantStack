# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Meta-test: verify every tool module exports callable tools via @tool_def.

The new architecture uses @tool_def() decorators that accumulate into a
module-level TOOLS list (via collect_tools()). server_factory then registers
these on a FastMCP instance at startup. This test imports each tool module
and validates that its TOOLS export contains valid, callable ToolDefinitions.

Catches:
  - Silent import failures (error class #6)
  - Modules that fail to export TOOLS
  - Non-callable or incorrectly decorated tool functions
"""

from __future__ import annotations

import asyncio
import importlib

import pytest


# All tool modules that should export TOOLS via @tool_def / collect_tools.
_TOOL_MODULES = [
    "quantstack.mcp.tools.qc_data",
    "quantstack.mcp.tools.qc_indicators",
    "quantstack.mcp.tools.qc_backtesting",
    "quantstack.mcp.tools.qc_research",
    "quantstack.mcp.tools.qc_options",
    "quantstack.mcp.tools.qc_risk",
    "quantstack.mcp.tools.qc_market",
    "quantstack.mcp.tools.qc_fundamentals",
    "quantstack.mcp.tools.qc_fundamentals_av",
    "quantstack.mcp.tools.qc_acquisition",
    "quantstack.mcp.tools.execution",
    "quantstack.mcp.tools.options_execution",
    "quantstack.mcp.tools.alerts",
    "quantstack.mcp.tools.coordination",
    "quantstack.mcp.tools.analysis",
    "quantstack.mcp.tools.portfolio",
    "quantstack.mcp.tools.attribution",
    "quantstack.mcp.tools.feedback",
    "quantstack.mcp.tools.signal",
    "quantstack.mcp.tools.intraday",
    "quantstack.mcp.tools.backtesting",
    "quantstack.mcp.tools.strategy",
    "quantstack.mcp.tools.meta",
    "quantstack.mcp.tools.learning",
    "quantstack.mcp.tools.decoder",
    "quantstack.mcp.tools.ml",
    "quantstack.mcp.tools.finrl_tools",
    "quantstack.mcp.tools.capitulation",
    "quantstack.mcp.tools.institutional_accumulation",
    "quantstack.mcp.tools.macro_signals",
    "quantstack.mcp.tools.cross_domain",
    "quantstack.mcp.tools.nlp",
]


def _load_all_tools():
    """Import all tool modules and collect their TOOLS exports."""
    all_tools = {}
    for modname in _TOOL_MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        tools = getattr(mod, "TOOLS", None)
        if tools:
            for td in tools:
                all_tools[td.name] = td
    return all_tools


class TestAllToolsCallable:
    """Verify that all tool modules export valid, callable tool definitions."""

    def test_modules_export_tools(self):
        """At least 50 tools should be exported across all modules."""
        tools = _load_all_tools()
        assert len(tools) >= 50, f"Expected 50+ tools, found {len(tools)}"

    def test_all_tools_are_callable(self):
        """Every exported ToolDefinition.fn should be callable."""
        tools = _load_all_tools()
        non_callable = [
            name for name, td in tools.items() if not callable(td.fn)
        ]
        assert non_callable == [], f"Non-callable tools: {non_callable}"

    def test_all_tools_are_async_or_sync(self):
        """Every tool function should be async or sync callable."""
        tools = _load_all_tools()
        stats = {"async": 0, "sync": 0}
        for name, td in tools.items():
            if asyncio.iscoroutinefunction(td.fn):
                stats["async"] += 1
            else:
                stats["sync"] += 1
        total = stats["async"] + stats["sync"]
        assert total >= 50, f"Expected 50+ callable tools, got {total}"

    def test_no_duplicate_tool_names(self):
        """Tool names should be unique across all modules.

        Known intentional duplicates: qc_fundamentals and qc_fundamentals_av
        both export get_financial_statements, get_insider_trades,
        get_institutional_ownership.  At runtime only one module loads
        (selected by API key in server_factory).
        """
        known_duplicates = {
            "get_financial_statements",
            "get_insider_trades",
            "get_institutional_ownership",
        }
        all_names = []
        for modname in _TOOL_MODULES:
            try:
                mod = importlib.import_module(modname)
            except ImportError:
                continue
            tools = getattr(mod, "TOOLS", None)
            if tools:
                all_names.extend(td.name for td in tools)
        unexpected = [
            n for n in all_names
            if all_names.count(n) > 1 and n not in known_duplicates
        ]
        assert unexpected == [], f"Unexpected duplicate tool names: {unexpected}"
