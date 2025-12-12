# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP (Model Context Protocol) Server Module.

Exposes QuantCore functionality as MCP tools for AI assistants.
"""

from quantcore.mcp.server import mcp, main

__all__ = ["mcp", "main"]
