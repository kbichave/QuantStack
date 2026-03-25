# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
WatchlistLoader — derives the symbol list for AutonomousRunner.

Symbol priority order:
1. AUTONOMOUS_WATCHLIST env var (comma-separated): "XOM,MSFT,SPY"
2. Screener results from ``screener_results`` DuckDB table (tiered)
3. Symbols inferred from live + forward_testing strategies
   (strategies with a ``symbols`` field in their parameters)
4. DEFAULT_SYMBOLS fallback (broad liquid ETFs)

Tiered loading (when USE_TIERED_WATCHLIST=true):
  Tier 1 (top 15): Full treatment — SignalEngine + ML + Groq
  Tier 2 (next 20): SignalEngine + rule-based only
  Tier 3 (next 15): Monitored, not actively traded by runner

Synchronous by design — called via asyncio.to_thread() from the runner.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

from loguru import logger

from quantstack.db import open_db, open_db_readonly
from quantstack.data.universe import WATCHLIST_DEFAULT

DEFAULT_SYMBOLS = list(WATCHLIST_DEFAULT)

# Tier sizes
_MAX_TIER_1 = 15
_MAX_TIER_2 = 20
_MAX_TIER_3 = 15

# Legacy hard cap (used when tiered mode is off)
_MAX_SYMBOLS = 20


def _get_db_connection():
    """Get a DB connection, preferring the existing write connection."""
    try:
        # If the write connection is already open, reuse it (avoids DuckDB
        # "different configuration" error when both RW and RO are attempted
        # on the same file in the same process).
        return open_db()
    except Exception:
        pass
    return open_db_readonly()


class WatchlistLoader:
    """Load and deduplicate the symbol list for one runner pass."""

    def load(self, tier: int | None = None) -> list[str]:
        """
        Return deduplicated, capped symbol list.

        Args:
            tier: If set, return only symbols from that tier (1, 2, or 3).
                  If None, return Tier 1 + Tier 2 (the tradeable set).
        """
        # 1. Env override — takes full precedence
        env_list = _load_from_env()
        if env_list:
            return env_list[:_MAX_SYMBOLS]

        # 2. Tiered loading from screener results
        if _use_tiered_watchlist():
            tiered = self.load_tiered()
            if tiered and any(tiered.values()):
                if tier is not None:
                    return tiered.get(tier, [])
                # Default: return T1 + T2 (tradeable symbols)
                return tiered.get(1, []) + tiered.get(2, [])

        # 3. Derive from strategy registry
        strategy_symbols = _load_from_strategies()
        if strategy_symbols:
            symbols = list(dict.fromkeys(strategy_symbols))
            return symbols[:_MAX_SYMBOLS]

        # 4. Fallback
        logger.info("[WatchlistLoader] using DEFAULT_SYMBOLS fallback")
        return DEFAULT_SYMBOLS[:_MAX_SYMBOLS]

    def load_tiered(self) -> dict[int, list[str]]:
        """
        Return tiered symbol lists from the screener_results table.

        Returns:
            {1: [top 15], 2: [next 20], 3: [next 15]}
            Empty dict if screener hasn't run yet.
        """
        try:
            conn = _get_db_connection()

            # Get the latest screened_at timestamp
            row = conn.execute(
                "SELECT MAX(screened_at) FROM screener_results"
            ).fetchone()

            if not row or not row[0]:
                logger.debug("[WatchlistLoader] No screener results found")
                return {}

            latest_ts = row[0]

            # Load symbols by tier for the latest screening
            rows = conn.execute(
                """
                SELECT symbol, tier
                FROM screener_results
                WHERE screened_at = ?
                ORDER BY composite_score DESC
                """,
                [latest_ts],
            ).fetchall()

            tiered: dict[int, list[str]] = {1: [], 2: [], 3: []}
            for symbol, tier in rows:
                if tier in tiered and len(tiered[tier]) < {
                    1: _MAX_TIER_1,
                    2: _MAX_TIER_2,
                    3: _MAX_TIER_3,
                }.get(tier, 0):
                    tiered[tier].append(symbol)

            total = sum(len(v) for v in tiered.values())
            if total > 0:
                logger.info(
                    f"[WatchlistLoader] from screener: "
                    f"T1={len(tiered[1])} T2={len(tiered[2])} T3={len(tiered[3])}"
                )
            return tiered

        except Exception as exc:
            logger.debug(f"[WatchlistLoader] could not read screener results: {exc}")
            return {}


def _use_tiered_watchlist() -> bool:
    """Check if tiered watchlist mode is enabled via env var."""
    return os.getenv("USE_TIERED_WATCHLIST", "false").lower() in ("true", "1", "yes")


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
    Extract symbols from the ``symbols`` field inside strategy parameters.

    Strategies may optionally specify:
        parameters.symbols: ["XOM", "CVX"]

    This is not required — strategies without a symbols field are regime-based,
    not symbol-specific. Those strategies do not contribute to the watchlist here;
    the runner will apply them to whatever symbols are in the watchlist.
    """
    try:
        conn = open_db_readonly()
        rows = conn.execute(
            """
            SELECT parameters
            FROM strategies
            WHERE status IN ('live', 'forward_testing')
            """
        ).fetchall()

        symbols: list[str] = []
        for (params_raw,) in rows:
            if not params_raw:
                continue
            try:
                params = (
                    json.loads(params_raw)
                    if isinstance(params_raw, str)
                    else params_raw
                )
                syms = params.get("symbols", [])
                if isinstance(syms, list):
                    symbols.extend(
                        s.strip().upper()
                        for s in syms
                        if isinstance(s, str) and s.strip()
                    )
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
