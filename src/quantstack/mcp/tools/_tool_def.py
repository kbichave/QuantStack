# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight tool definition decorator for explicit MCP tool collection.

Replaces the import-time ``@mcp.tool()`` pattern with a metadata-only
decorator.  Tool modules use ``@tool_def()`` to annotate async functions;
the server factory then collects them via the module-level ``TOOLS`` export
and registers them explicitly on the target server.

Key property: ``@tool_def()`` returns the **original function** unchanged.
Functions remain plain async callables — no FunctionTool wrapper, no global
singleton reference, no import-time side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata for a single MCP tool."""

    fn: Callable[..., Any]
    name: str
    description: str | None = None


# Module-scoped accumulator — each tool module appends here at import time,
# then calls ``collect_tools()`` at the bottom to drain into its ``TOOLS``.
_MODULE_TOOLS: list[ToolDefinition] = []


def tool_def(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that records tool metadata without modifying the function.

    Args:
        name: MCP tool name.  Defaults to ``fn.__name__``.
        description: Tool description shown to the LLM.  Defaults to ``fn.__doc__``.

    Returns:
        The original function, unmodified.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _MODULE_TOOLS.append(
            ToolDefinition(
                fn=fn,
                name=name or fn.__name__,
                description=description or fn.__doc__,
            )
        )
        return fn  # returns ORIGINAL function — not a wrapper

    return decorator


def collect_tools() -> list[ToolDefinition]:
    """Drain the module-scoped accumulator and return collected definitions.

    Call this once at the bottom of each tool module to export ``TOOLS``.
    Clears the accumulator so the next imported module starts fresh.
    """
    tools = list(_MODULE_TOOLS)
    _MODULE_TOOLS.clear()
    return tools
