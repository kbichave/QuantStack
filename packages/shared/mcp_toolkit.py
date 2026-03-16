# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Shared MCP server boilerplate.

Provides:
- ``mcp_tool_response``  — standard {success, error, ...} envelope
- ``mcp_tool_safe``      — decorator that wraps tool functions with error handling
- ``require_resource``   — guard that raises if a server resource is None
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

from loguru import logger


def mcp_tool_response(
    success: bool,
    error: str | None = None,
    **data: Any,
) -> dict[str, Any]:
    """Build a standard MCP tool response envelope.

    Every tool should return this shape so consumers (Claude, test harness)
    can rely on ``response["success"]`` as the single boolean discriminant.
    """
    resp: dict[str, Any] = {"success": success}
    if error is not None:
        resp["error"] = error
    resp.update(data)
    return resp


def mcp_tool_safe(fn):
    """Decorator: catch unhandled exceptions and return a standard error envelope.

    Works for both sync and async tool functions.

    Usage::

        @mcp.tool()
        @mcp_tool_safe
        async def my_tool(...) -> dict:
            ...
            return mcp_tool_response(True, data=result)
    """
    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                logger.error(f"[mcp_tool] {fn.__name__} failed: {exc}")
                return mcp_tool_response(False, error=str(exc))

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.error(f"[mcp_tool] {fn.__name__} failed: {exc}")
            return mcp_tool_response(False, error=str(exc))

    return sync_wrapper


def require_resource(resource: Any, name: str) -> Any:
    """Return *resource* if it is not None, otherwise raise RuntimeError.

    Replaces the per-server ``_require_broker()``, ``_require_ctx()``, etc.
    pattern with a single generic guard.

    Args:
        resource: The server resource to check (broker client, context, etc.).
        name: Human-readable label used in the error message.

    Raises:
        RuntimeError: When *resource* is None.
    """
    if resource is None:
        raise RuntimeError(
            f"{name} not initialized. Check env vars and server startup logs."
        )
    return resource
