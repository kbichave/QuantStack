# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Enhancement 5 — Execution Feedback Loop tools.

Tools:
  - get_fill_quality    — assess execution quality for a fill vs VWAP
  - get_position_monitor — comprehensive position status: price, P&L, stop proximity, regime
"""

import asyncio
from datetime import datetime as _dt
from typing import Any

from loguru import logger

from quantstack.agents.regime_detector import RegimeDetectorAgent
from quantstack.data.storage import DataStore  # noqa: F401
from quantstack.config.timeframes import Timeframe as _TF
from quantstack.mcp._helpers import _get_reader
from quantstack.mcp._state import (
    _serialize,
    live_db_or_error,
    require_ctx,
    require_live_db,
)
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain



# =============================================================================
# MCP Tools
# =============================================================================


@domain(Domain.PORTFOLIO, Domain.SIGNALS)
@mcp.tool()
async def get_fill_quality(order_id: str) -> dict[str, Any]:
    """
    Assess execution quality for a completed fill.

    Compares the fill price to VWAP at fill time and returns slippage analysis.
    Use during /reflect sessions to track execution quality over time.

    Args:
        order_id: Order ID from get_fills output.

    Returns:
        Dict with fill_price, slippage_bps, vwap, fill_vs_vwap_bps, quality_note.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        row = ctx.db.execute(
            """
            SELECT order_id, symbol, side, fill_price, filled_quantity,
                   slippage_bps, commission, filled_at
            FROM fills
            WHERE order_id = ? AND rejected = FALSE
            """,
            [order_id],
        ).fetchone()

        if not row:
            return {
                "success": False,
                "error": f"Fill not found for order_id={order_id}",
            }

        (
            oid,
            symbol,
            side,
            fill_price,
            filled_qty,
            recorded_slippage,
            commission,
            filled_at,
        ) = row

        # Attempt VWAP comparison via QuantCore data store
        vwap: float | None = None
        fill_vs_vwap_bps: float | None = None
        try:
            store = _get_reader()
            df = store.load_ohlcv(symbol, _TF.D1)
            if df is not None and not df.empty and "vwap" in df.columns:
                fill_date = str(filled_at)[:10]
                day_rows = df[df.index.astype(str).str.startswith(fill_date)]
                if not day_rows.empty:
                    vwap = float(day_rows["vwap"].iloc[-1])
                    if vwap > 0 and fill_price and fill_price > 0:
                        fill_vs_vwap_bps = round((fill_price - vwap) / vwap * 10_000, 1)
        except Exception:
            pass

        direction_label = "above" if (fill_vs_vwap_bps or 0) > 0 else "below"
        quality_note = f"Recorded slippage: {(recorded_slippage or 0):.1f} bps. " + (
            f"Fill was {abs(fill_vs_vwap_bps):.1f} bps {direction_label} VWAP."
            if fill_vs_vwap_bps is not None
            else "VWAP data unavailable for comparison."
        )

        return {
            "success": True,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "fill_price": fill_price,
            "filled_quantity": filled_qty,
            "slippage_bps": recorded_slippage,
            "vwap": vwap,
            "fill_vs_vwap_bps": fill_vs_vwap_bps,
            "commission": commission,
            "filled_at": str(filled_at) if filled_at else None,
            "quality_note": quality_note,
        }

    except Exception as e:
        logger.error(f"[quantpod_mcp] get_fill_quality({order_id}) failed: {e}")
        return {"success": False, "error": str(e), "order_id": order_id}


@domain(Domain.PORTFOLIO, Domain.SIGNALS)
@mcp.tool()
async def get_position_monitor(symbol: str) -> dict[str, Any]:
    """
    Comprehensive position status for an open position.

    Returns price, unrealized P&L, ATR-based stop distance, days held,
    and current vs entry regime.  Designed for /review position checks.

    Args:
        symbol: Ticker symbol of the open position.

    Returns:
        Dict with price, pnl, days_held, current_regime, flags, recommended_action.
        Returns has_position=False if no open position exists.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        pos = ctx.portfolio.get_position(symbol)
        if not pos:
            return {
                "success": True,
                "symbol": symbol,
                "has_position": False,
                "note": "No open position found.",
            }

        current_price = pos.current_price or 0.0
        avg_cost = pos.avg_cost or 0.0
        quantity = pos.quantity or 0
        unrealized_pnl = pos.unrealized_pnl or 0.0

        pnl_pct = 0.0
        if avg_cost > 0 and current_price > 0:
            pnl_pct = round((current_price - avg_cost) / avg_cost * 100, 2)

        # Time held
        days_held: int | None = None
        entry_time: str | None = None
        try:
            row = ctx.db.execute(
                "SELECT opened_at FROM positions WHERE symbol = ?",
                [symbol],
            ).fetchone()
            if row and row[0]:
                opened_at = row[0]
                if isinstance(opened_at, str):
                    opened_at = _dt.fromisoformat(opened_at)
                days_held = (_dt.now() - opened_at).days
                entry_time = str(row[0])
        except Exception:
            pass

        # Current regime
        current_regime = "unknown"
        atr: float = 0.0
        try:
            detector = RegimeDetectorAgent(symbols=[symbol])
            r = await asyncio.get_event_loop().run_in_executor(
                None, detector.detect_regime, symbol
            )
            current_regime = r.get("trend_regime", "unknown")
            atr = float(r.get("atr", 0))
        except Exception:
            pass

        # ATR-based stop proximity
        near_stop = False
        atr_stop_distance_pct: float | None = None
        if atr > 0 and avg_cost > 0 and current_price > 0:
            atr_stop_distance_pct = round(atr / avg_cost * 100, 2)
            # Flag if within 30% of a 2-ATR stop
            stop_level = avg_cost - 2 * atr
            range_to_stop = avg_cost - stop_level  # = 2 * ATR
            if range_to_stop > 0:
                pct_to_stop = (current_price - stop_level) / range_to_stop
                near_stop = pct_to_stop < 0.30

        # Approaching target (>80% of a 3R move)
        near_target = False
        if atr > 0 and avg_cost > 0 and current_price > 0:
            target_level = avg_cost + 3 * atr
            range_to_target = target_level - avg_cost
            if range_to_target > 0:
                pct_to_target = (current_price - avg_cost) / range_to_target
                near_target = pct_to_target >= 0.80

        flags = {
            "near_stop": near_stop,
            "near_target": near_target,
            "pnl_positive": unrealized_pnl > 0,
        }

        if near_stop:
            recommended_action = "TIGHTEN STOP — price approaching 2-ATR stop level"
        elif near_target:
            recommended_action = "CONSIDER PARTIAL EXIT — 80% of 3R target reached"
        else:
            recommended_action = "HOLD — within normal parameters"

        return {
            "success": True,
            "symbol": symbol,
            "has_position": True,
            "quantity": quantity,
            "avg_cost": round(avg_cost, 4),
            "current_price": round(current_price, 4),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "pnl_pct": pnl_pct,
            "days_held": days_held,
            "entry_time": entry_time,
            "current_regime": current_regime,
            "atr_stop_distance_pct": atr_stop_distance_pct,
            "flags": flags,
            "recommended_action": recommended_action,
        }

    except Exception as e:
        logger.error(f"[quantpod_mcp] get_position_monitor({symbol}) failed: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}
