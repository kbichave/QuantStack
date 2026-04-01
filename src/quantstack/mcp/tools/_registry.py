# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Domain registry for MCP tools.

Provides the ``@domain()`` decorator that tags tool functions with the
server domain(s) they belong to.  The server factory reads ``TOOL_DOMAINS``
at startup to register only the tools for its target domain.

Usage::

    from quantstack.mcp.tools._registry import domain
    from quantstack.mcp.tools._tool_def import tool_def
    from quantstack.mcp.domains import Domain

    @domain(Domain.ML)
    @tool_def()
    async def train_ml_model(symbol: str, ...) -> dict:
        ...

    # Cross-cutting tool — registered in multiple servers:
    @domain(Domain.SIGNALS, Domain.INTEL, Domain.RESEARCH)
    @tool_def()
    async def get_regime(symbol: str) -> dict:
        ...
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from quantstack.mcp.domains import Domain

F = TypeVar("F", bound=Callable[..., Any])

# Global registry: function_name -> combined Domain flags.
# Populated by @domain() at import time, read by server_factory at startup.
TOOL_DOMAINS: dict[str, Domain] = {}


def domain(*domains: Domain) -> Callable[[F], F]:
    """Tag a tool function with its domain affinity.

    Can be stacked with ``@tool_def()`` in any order — this decorator is
    purely a metadata annotation that does not modify the function.

    Args:
        *domains: One or more :class:`Domain` flags.  The tool will be
            registered in every server whose target domain overlaps.
    """
    combined = Domain(0)
    for d in domains:
        combined |= d

    def decorator(fn: F) -> F:
        # Handle both raw functions and FastMCP FunctionTool wrappers
        name = getattr(fn, "__name__", None) or getattr(fn, "name", None)
        if name:
            TOOL_DOMAINS[name] = combined
        return fn

    return decorator
