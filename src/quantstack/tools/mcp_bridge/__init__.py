# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
MCP Bridge - Communication layer to MCP servers.

Provides MCPBridge for calling QuantCore and eTrade MCP servers.
Tool class wrappers (tools_*.py) are legacy CrewAI-style tools — new code
should use tools/langchain/ (LLM-facing) or tools/functions/ (node-callable).
"""

from ._bridge import MCPBridge, get_bridge

__all__ = [
    "MCPBridge",
    "get_bridge",
]
