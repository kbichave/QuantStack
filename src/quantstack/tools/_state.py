# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Shared state for tool implementations — module-level variables and guard helpers.

All tool modules import from here rather than maintaining their own state.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from quantstack.context import TradingContext

from quantstack.shared.cache import TTLCache
from quantstack.shared.files import read_memory_file
from quantstack.shared.serializers import serialize_for_json


# =============================================================================
# Mutable State
# =============================================================================

_ctx: TradingContext | None = None

# IC output cache — keyed "{symbol}::{ic_name}", 30-minute TTL.
ic_cache = TTLCache(ttl_seconds=1800)


# =============================================================================
# State Setters (called from lifespan)
# =============================================================================


def set_ctx(ctx: TradingContext) -> None:
    global _ctx
    _ctx = ctx


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
    """Get the trading context, auto-initializing if needed.

    When tools are invoked as direct Python imports, ``_ctx`` may be None.
    Rather than crashing, we create a minimal context on demand.
    """
    global _ctx
    if _ctx is None:
        logger.debug("[state] Auto-initializing TradingContext for direct-import mode")
        from quantstack.context import create_trading_context
        _ctx = create_trading_context(run_migrations_on_init=False)
    return _ctx


def require_live_db() -> TradingContext:
    """Get the trading context.

    PostgreSQL is always available — there is no degraded mode.
    This function is kept for call-site compatibility; it delegates to
    ``require_ctx()``.
    """
    return require_ctx()


def live_db_or_error() -> tuple[TradingContext | None, dict | None]:
    """Get context, returning (ctx, None) on success or (None, error_dict) on failure."""
    try:
        return require_live_db(), None
    except RuntimeError as exc:
        return None, {
            "success": False,
            "error": str(exc),
        }


# =============================================================================
# Serialization Helpers
# =============================================================================


def _serialize(obj: Any) -> Any:
    """Convert Pydantic models, dataclasses, and datetime to JSON-safe dicts."""
    return serialize_for_json(obj)


def _read_memory_file(filename: str, max_chars: int = 2000) -> str:
    """Read a .claude/memory/*.md file and return its content (truncated)."""
    return read_memory_file(filename, max_chars=max_chars)
