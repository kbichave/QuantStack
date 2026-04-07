# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""TTL cache for SignalBrief results.

Avoids re-running 22 collectors when the same symbol is analyzed
multiple times within a research cycle. Uses the shared TTLCache
(same infra as the IC output cache).

Default TTL: 1 hour (configurable via SIGNAL_ENGINE_CACHE_TTL env var).
Disable entirely with SIGNAL_ENGINE_CACHE_ENABLED=false.
"""

from __future__ import annotations

import os

from loguru import logger

from quantstack.shared.cache import TTLCache
from quantstack.signal_engine.brief import SignalBrief

_ttl = int(os.environ.get("SIGNAL_ENGINE_CACHE_TTL", "3600"))
_enabled = os.environ.get("SIGNAL_ENGINE_CACHE_ENABLED", "true").lower() == "true"

_cache = TTLCache(ttl_seconds=_ttl)

# Counters for observability (read via /status or Langfuse metrics).
hits = 0
misses = 0


def get(symbol: str) -> SignalBrief | None:
    """Return cached brief if fresh and caching is enabled, else None."""
    global hits, misses
    if not _enabled:
        return None
    result = _cache.get(symbol.upper())
    if result is not None:
        hits += 1
        logger.debug(f"[signal_cache] HIT {symbol} (hits={hits}, misses={misses})")
    else:
        misses += 1
    return result


def put(symbol: str, brief: SignalBrief, ttl: int | None = None) -> None:
    """Store brief with configured TTL, or override with per-entry TTL."""
    if _enabled:
        _cache.set(symbol.upper(), brief, ttl=ttl)


def invalidate(symbol: str) -> None:
    """Remove a specific symbol's cached brief."""
    _cache.delete(symbol.upper())


def clear() -> None:
    """Drop all cached briefs (e.g., after market data refresh)."""
    _cache.clear()


def stats() -> dict[str, int]:
    """Return cache hit/miss counters."""
    return {"hits": hits, "misses": misses, "size": len(_cache)}
