# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantPod MCP shared state — module-level variables and guard helpers.

All tool modules import from here rather than maintaining their own state.
The ``mcp`` singleton lives in ``server.py``; this module holds mutable
state (context, degraded mode, cache) and the functions that guard it.
"""

import asyncio
import functools
import inspect
import time
from pathlib import Path
from typing import Any

from loguru import logger

from quantstack.context import TradingContext, create_trading_context
from quantstack.shared.cache import TTLCache
from quantstack.shared.serializers import serialize_for_json


def release_db() -> None:
    """Release the DB connection lock so other processes can access the DB.

    Called automatically after every MCP tool via the ``auto_release_db``
    decorator.  Safe to call even if no connection is open.
    """
    if _ctx is not None and hasattr(_ctx, "db") and _ctx.db is not None:
        try:
            _ctx.db.release()
        except Exception:
            pass


def auto_release_db(fn):
    """Decorator: release the DuckDB file lock after the tool function returns.

    Wraps sync or async functions.  On success or failure, the connection is
    released so other processes (scripts, autonomous runner) can access the DB
    between MCP tool calls.
    """
    if inspect.iscoroutinefunction(fn) or asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            finally:
                release_db()

        return wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            release_db()

    return wrapper


# =============================================================================
# Mutable State
# =============================================================================

_ctx: TradingContext | None = None
_degraded_mode: bool = False
_degraded_reason: str = ""

# IC output cache — keyed "{symbol}::{ic_name}", 30-minute TTL.
ic_cache = TTLCache(ttl_seconds=1800)


# =============================================================================
# State Setters (called from lifespan)
# =============================================================================


def set_ctx(ctx: TradingContext) -> None:
    global _ctx
    _ctx = ctx


def set_degraded(mode: bool, reason: str = "") -> None:
    global _degraded_mode, _degraded_reason
    _degraded_mode = mode
    _degraded_reason = reason


def get_degraded_reason() -> str:
    return _degraded_reason


def is_degraded() -> bool:
    return _degraded_mode


# =============================================================================
# IC Cache Helpers
# =============================================================================


def ic_cache_set(symbol: str, ic_name: str, output: str) -> None:
    """Cache raw IC output text for later retrieval."""
    ic_cache.set(f"{symbol}::{ic_name}", output)


def ic_cache_get(symbol: str, ic_name: str) -> str | None:
    """Retrieve cached IC output, returning None if absent or expired."""
    return ic_cache.get(f"{symbol}::{ic_name}")


# =============================================================================
# Context Guards
# =============================================================================


def require_ctx() -> TradingContext:
    """Get the trading context, raising if the server hasn't started."""
    if _ctx is None:
        raise RuntimeError("QuantPod MCP server not initialized — call lifespan first")
    return _ctx


def _try_recover_from_degraded() -> bool:
    """Attempt to recover from degraded mode by retrying the DB connection."""
    global _ctx, _degraded_mode, _degraded_reason
    try:
        from quantstack.db import reset_connection

        reset_connection()
        new_ctx = create_trading_context()
        _ctx = new_ctx
        _degraded_mode = False
        _degraded_reason = ""
        logger.info("[MCP] Recovered from degraded mode — DB connection restored")
        return True
    except Exception as exc:
        logger.debug(f"[MCP] Recovery attempt failed: {exc}")
        return False


def require_live_db() -> TradingContext:
    """Get the trading context, raising if in degraded mode (DB locked).

    Tools that need persistent DB state (portfolio, fills, audit trail) use this.
    Purely computational tools (run_analysis, get_regime) should use require_ctx().
    """
    if _degraded_mode:
        if _try_recover_from_degraded():
            return _ctx
        raise RuntimeError(
            f"QuantPod is running in degraded mode — the persistent DB is locked. "
            f"Portfolio state and trade execution are unavailable. "
            f"Analysis tools (run_analysis, get_regime) still work. "
            f"Reason: {_degraded_reason}"
        )
    return require_ctx()


def live_db_or_error() -> tuple[TradingContext | None, dict | None]:
    """Get context with read-only fallback in degraded mode.

    Returns (ctx, None) on success, (None, error_dict) on failure.
    """
    try:
        return require_live_db(), None
    except RuntimeError:
        pass

    try:
        from quantstack.db import open_db_readonly

        ro_conn = open_db_readonly()
        ctx = require_ctx()
        if not hasattr(ctx, "_original_db"):
            ctx._original_db = ctx.db
        ctx.db = ro_conn
        logger.debug("[MCP] Using read-only DB fallback for degraded mode")
        return ctx, None
    except Exception as exc:
        logger.debug(f"[MCP] Read-only fallback failed: {exc}")
        return None, {
            "success": False,
            "error": (
                f"QuantPod is in degraded mode and read-only fallback failed. "
                f"Degraded reason: {_degraded_reason}. "
                f"Read-only error: {exc}"
            ),
            "degraded_mode": True,
        }


# =============================================================================
# Serialization Helpers
# =============================================================================


def _serialize(obj: Any) -> Any:
    """Convert Pydantic models, dataclasses, and datetime to JSON-safe dicts."""
    return serialize_for_json(obj)


def _read_memory_file(filename: str, max_chars: int = 2000) -> str:
    """Read a .claude/memory/*.md file and return its content (truncated)."""
    candidates = [
        Path(__file__).parents[4] / ".claude" / "memory" / filename,
        Path.home() / ".claude" / "memory" / filename,
    ]
    for path in candidates:
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                return content[:max_chars] if len(content) > max_chars else content
            except OSError:
                pass
    return ""
