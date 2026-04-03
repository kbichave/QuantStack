# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tools module for LangGraph agent tools.

Tool Categories:
- langchain/: LLM-facing tools with @tool decorator (used by agent nodes)
- functions/: Node-callable async functions (called directly by graph nodes)
- registry.py: TOOL_REGISTRY mapping YAML tool names to tool objects
- mcp_bridge/: MCPBridge class for MCP server communication (internal)
"""

from quantstack.tools.mcp_bridge._bridge import MCPBridge, get_bridge
from quantstack.tools.registry import TOOL_REGISTRY, get_tools_for_agent

__all__ = [
    "MCPBridge",
    "get_bridge",
    "TOOL_REGISTRY",
    "get_tools_for_agent",
]
