# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
WatchlistLoader — derives the symbol list for AutonomousRunner.

Symbol priority order:
1. AUTONOMOUS_WATCHLIST env var (comma-separated): "XOM,MSFT,SPY"
2. Symbols inferred from live + forward_testing strategies
   (strategies with a `symbols` field in their parameters)
3. DEFAULT_SYMBOLS fallback (broad liquid ETFs)

Synchronous by design — called via asyncio.to_thread() from the runner.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

from loguru import logger


DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "XOM", "MSFT", "AAPL", "JPM", "GLD"]

# Hard cap: even if strategies specify more symbols, we cap at this number
# to keep the per-pass latency bounded.
_MAX_SYMBOLS = 20


class WatchlistLoader:
    """Load and deduplicate the symbol list for one runner pass."""

    def load(self) -> list[str]:
        """Return deduplicated, capped symbol list."""
        # 1. Env override — takes full precedence
        env_list = _load_from_env()
        if env_list:
            return env_list[:_MAX_SYMBOLS]

        # 2. Derive from strategy registry
        strategy_symbols = _load_from_strategies()
        if strategy_symbols:
            symbols = list(dict.fromkeys(strategy_symbols))  # preserve order, deduplicate
            return symbols[:_MAX_SYMBOLS]

        # 3. Fallback
        logger.info("[WatchlistLoader] using DEFAULT_SYMBOLS fallback")
        return DEFAULT_SYMBOLS[:_MAX_SYMBOLS]


def _load_from_env() -> list[str]:
    """Parse AUTONOMOUS_WATCHLIST env var."""
    raw = os.getenv("AUTONOMOUS_WATCHLIST", "").strip()
    if not raw:
        return []
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if symbols:
        logger.info(f"[WatchlistLoader] from env: {symbols}")
    return symbols


def _load_from_strategies() -> list[str]:
    """
    Extract symbols from the `symbols` field inside strategy parameters.

    Strategies may optionally specify:
        parameters.symbols: ["XOM", "CVX"]

    This is not required — strategies without a symbols field are regime-based,
    not symbol-specific. Those strategies do not contribute to the watchlist here;
    the runner will apply them to whatever symbols are in the watchlist.
    """
    try:
        from quant_pod.db import open_db_readonly

        conn = open_db_readonly()
        rows = conn.execute(
            """
            SELECT parameters
            FROM strategies
            WHERE status IN ('live', 'forward_testing')
            """
        ).fetchall()
        conn.close()

        symbols: list[str] = []
        for (params_raw,) in rows:
            if not params_raw:
                continue
            try:
                params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
                syms = params.get("symbols", [])
                if isinstance(syms, list):
                    symbols.extend(s.strip().upper() for s in syms if isinstance(s, str) and s.strip())
            except (ValueError, TypeError):
                continue

        if symbols:
            unique = list(dict.fromkeys(symbols))
            logger.info(f"[WatchlistLoader] from strategies: {unique}")
            return unique
        return []

    except Exception as exc:
        logger.debug(f"[WatchlistLoader] could not read strategies: {exc}")
        return []
