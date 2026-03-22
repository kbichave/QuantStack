# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Autonomous execution hooks — fire-and-forget callbacks wired into
the execution pipeline. Non-blocking: failures are logged, never fatal.

Hooks:
  - on_trade_close: fires ReflectionManager + ReflexionMemory after every position close
  - on_daily_close: fires daily reflection summary at market close
  - find_similar_situations: SQL query for pre-trade context (regime + symbol + strategy)
  - get_reflexion_episodes: structured episodic memory for debate filter injection
"""

from __future__ import annotations

from datetime import date
from typing import Any

from loguru import logger

# Module-level singletons — initialized lazily on first use
_reflection_mgr = None
_reflexion_mem = None


def _get_reflection_manager():
    """Lazy-init ReflectionManager with a ManagedConnection."""
    global _reflection_mgr
    if _reflection_mgr is None:
        try:
            from quantstack.autonomous.reflection import ReflectionManager
            from quantstack.db import open_db

            conn = open_db()
            _reflection_mgr = ReflectionManager(conn)
        except Exception as exc:
            logger.warning(f"[hooks] Failed to init ReflectionManager: {exc}")
            return None
    return _reflection_mgr


def _get_reflexion_memory():
    """Lazy-init ReflexionMemory with a ManagedConnection."""
    global _reflexion_mem
    if _reflexion_mem is None:
        try:
            from quantstack.optimization.reflexion_memory import ReflexionMemory
            from quantstack.db import open_db

            conn = open_db()
            _reflexion_mem = ReflexionMemory(conn)
        except Exception as exc:
            logger.warning(f"[hooks] Failed to init ReflexionMemory: {exc}")
            return None
    return _reflexion_mem


def on_trade_close(
    symbol: str,
    strategy_id: str,
    action: str,
    entry_price: float,
    exit_price: float,
    realized_pnl_pct: float,
    holding_days: int = 0,
    regime_at_entry: str = "unknown",
    regime_at_exit: str = "unknown",
    conviction: float = 0.0,
    signals_summary: str = "",
) -> None:
    """Fire after every trade close. Non-blocking, best-effort.

    Called from PortfolioState.close_position or execute_trade.
    Records outcome in ReflectionManager (raw journal) and, for losses > 1%,
    creates a classified ReflexionEpisode (structured episodic memory).
    """
    # 1. Raw reflection (always)
    try:
        mgr = _get_reflection_manager()
        if mgr is None:
            return
        ref = mgr.record_outcome(
            symbol=symbol,
            strategy_id=strategy_id,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pnl_pct=realized_pnl_pct,
            holding_days=holding_days,
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
            conviction=conviction,
            signals_entry=signals_summary,
        )
    except Exception as exc:
        logger.debug(f"[hooks] on_trade_close reflection failed (non-critical): {exc}")
        return

    # 2. Structured reflexion episode (losses only)
    if realized_pnl_pct < -1.0:
        try:
            mem = _get_reflexion_memory()
            if mem is not None:
                mem.record_episode(ref)
        except Exception as exc:
            logger.debug(f"[hooks] on_trade_close reflexion episode failed (non-critical): {exc}")


def on_daily_close(
    snapshot_date: date,
    daily_pnl: float = 0.0,
    daily_return_pct: float = 0.0,
    closed_trades: list[dict] | None = None,
) -> None:
    """Fire at market close (after equity snapshot). Non-blocking, best-effort.

    Called from EquityTracker.snapshot_daily.
    """
    try:
        mgr = _get_reflection_manager()
        if mgr is None:
            return
        mgr.daily_reflection(
            snapshot_date=snapshot_date,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            closed_trades=closed_trades,
        )
    except Exception as exc:
        logger.debug(f"[hooks] on_daily_close failed (non-critical): {exc}")


def find_similar_situations(
    symbol: str,
    regime: str,
    signals: str,
    top_k: int = 3,
) -> list[dict]:
    """Query past trade situations similar to current context.

    Use before entering a trade to surface relevant lessons.
    Returns list of dicts with keys: symbol, pnl_pct, lesson, regime.
    """
    try:
        mgr = _get_reflection_manager()
        if mgr is None:
            return []
        matches = mgr.find_similar(symbol, regime, signals, top_k=top_k)
        return [
            {
                "symbol": m.symbol,
                "pnl_pct": m.realized_pnl_pct,
                "lesson": m.lesson,
                "regime": m.regime_at_entry,
                "strategy": m.strategy_id,
            }
            for m in matches
            if m.lesson  # Only return entries with lessons
        ]
    except Exception as exc:
        logger.debug(f"[hooks] find_similar_situations failed: {exc}")
        return []


def get_reflexion_episodes(
    regime: str,
    strategy_id: str = "",
    symbol: str = "",
    k: int = 3,
) -> list:
    """Retrieve structured reflexion episodes for debate filter injection.

    Returns list of ReflexionEpisode objects (or empty list on failure).
    Used by AutonomousRunner to condition the debate filter on past failures.
    """
    try:
        mem = _get_reflexion_memory()
        if mem is None:
            return []
        return mem.get_relevant(regime=regime, strategy_id=strategy_id, symbol=symbol, k=k)
    except Exception as exc:
        logger.debug(f"[hooks] get_reflexion_episodes failed: {exc}")
        return []
