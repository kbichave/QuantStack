# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP Server - Entry Point.

Usage:
    python -m quantcore.mcp_server

Or run directly:
    python src/quantcore/mcp_server.py
"""

from quantcore.mcp.server import mcp, main

if __name__ == "__main__":
    main()
