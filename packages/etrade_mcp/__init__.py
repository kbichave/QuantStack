# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
eTrade MCP (Model Context Protocol) Server Module.

Exposes eTrade trading functionality as MCP tools for AI assistants.
Supports OAuth 1.0a authentication, account management, market data,
and order execution.
"""

try:
    from etrade_mcp.server import mcp, main
except ImportError:
    # Handle case where server not fully initialized
    mcp = None
    main = None

__all__ = ["mcp", "main"]
