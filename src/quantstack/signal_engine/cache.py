# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""TTL cache for SignalBrief results.

Avoids re-running 22 collectors when the same symbol is analyzed
multiple times within a research cycle. Uses the shared TTLCache
(same infra as the IC output cache).

Default TTL: 1 hour (configurable via SIGNAL_ENGINE_CACHE_TTL env var).
Disable entirely with SIGNAL_ENGINE_CACHE_ENABLED=false.

P01 §1.3: When FEEDBACK_SIGNAL_DECAY=true, cached briefs have their
conviction/confidence exponentially decayed based on age. Half-life
configurable via SIGNAL_DECAY_HALF_LIFE (default 1800s = 30 min).
"""

from __future__ import annotations

import math
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

# P01 §1.3: Exponential decay parameters
_DECAY_HALF_LIFE = int(os.environ.get("SIGNAL_DECAY_HALF_LIFE", "1800"))
_DECAY_LAMBDA = math.log(2) / _DECAY_HALF_LIFE if _DECAY_HALF_LIFE > 0 else 0.0


def get(symbol: str) -> SignalBrief | None:
    """Return cached brief if fresh and caching is enabled, else None.

    When signal decay is enabled (FEEDBACK_SIGNAL_DECAY=true), applies
    exponential decay to conviction/confidence based on the brief's age.
    """
    global hits, misses
    if not _enabled:
        return None

    result = _cache.get_with_age(symbol.upper())
    if result is None:
        misses += 1
        return None

    brief, age_seconds = result
    hits += 1
    logger.debug(f"[signal_cache] HIT {symbol} (hits={hits}, misses={misses}, age={age_seconds:.0f}s)")

    # P01 §1.3: Apply conviction decay if enabled and brief is > 60s old
    from quantstack.config.feedback_flags import signal_decay_enabled
    if signal_decay_enabled() and age_seconds > 60 and _DECAY_LAMBDA > 0:
        decay_factor = math.exp(-_DECAY_LAMBDA * age_seconds)
        brief = _apply_decay(brief, decay_factor)

    return brief


def _apply_decay(brief: SignalBrief, factor: float) -> SignalBrief:
    """Return a copy of brief with decayed conviction/confidence.

    Only reduces values, never amplifies. The original brief in the
    cache remains unchanged (Pydantic model_dump creates a copy).
    """
    data = brief.model_dump()
    data["overall_confidence"] = round(data.get("overall_confidence", 0.5) * factor, 3)
    for sb in data.get("symbol_briefs", []):
        sb["consensus_conviction"] = round(sb.get("consensus_conviction", 0.5) * factor, 3)
    return SignalBrief.model_validate(data)


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
