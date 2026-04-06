"""
Focus list — the subset of the universe that agents actively research and trade.

Reads from ``focus.yaml`` in the same directory. Cached after first load.
Override at runtime via ``QUANTSTACK_FOCUS_SYMBOLS=SPY,QQQ,AAPL`` env var.

Usage:
    from quantstack.config.focus import get_focus_symbols

    symbols = get_focus_symbols()          # -> ["AAL", "AAPL", ..., "XME"]
    if "TSLA" in get_focus_symbols():
        ...
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_FOCUS_YAML = Path(__file__).with_name("focus.yaml")


@lru_cache(maxsize=1)
def get_focus_symbols() -> tuple[str, ...]:
    """Return the focused symbol list, sorted and deduplicated.

    Resolution order:
      1. QUANTSTACK_FOCUS_SYMBOLS env var (comma-separated)
      2. focus.yaml (stocks + etfs sections)
    """
    env_override = os.environ.get("QUANTSTACK_FOCUS_SYMBOLS", "").strip()
    if env_override:
        return tuple(sorted({s.strip().upper() for s in env_override.split(",") if s.strip()}))

    if not _FOCUS_YAML.exists():
        return ()

    with open(_FOCUS_YAML) as f:
        data = yaml.safe_load(f) or {}

    stocks = [s.upper() for s in (data.get("stocks") or [])]
    etfs = [s.upper() for s in (data.get("etfs") or [])]
    return tuple(sorted(set(stocks + etfs)))
