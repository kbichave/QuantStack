# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1 MCP tools — read-only analysis and system introspection.

Tools:
  - get_portfolio_state — positions, cash, equity, P&L
  - get_regime         — ADX/ATR regime classification
  - get_recent_decisions — audit trail query
  - get_system_status  — kill switch, risk halt, broker mode

Note: run_analysis (CrewAI-based) was removed in v0.6.0.
      Use ``get_signal_brief`` (SignalEngine) instead.
"""

import asyncio
from typing import Any

from loguru import logger

from quant_pod.mcp.server import mcp
from quant_pod.mcp._state import (
    require_ctx,
    live_db_or_error,
    _serialize,
)
from quant_pod.execution.broker_factory import get_broker_mode


# =============================================================================
# TOOL 2: get_portfolio_state
# =============================================================================


@mcp.tool()
async def get_portfolio_state() -> dict[str, Any]:
    """
    Return the current portfolio state: positions, cash, equity, and P&L.

    Returns:
        Dict with keys: snapshot, positions, context_string.
        - snapshot: cash, positions_value, total_equity, daily_pnl, etc.
        - positions: list of open positions with symbol, quantity, avg_cost, etc.
        - context_string: human-readable markdown summary.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        snapshot = ctx.portfolio.get_snapshot()
        positions = ctx.portfolio.get_positions()
        context_str = ctx.portfolio.as_context_string()
        return {
            "success": True,
            "snapshot": _serialize(snapshot),
            "positions": [_serialize(p) for p in positions],
            "context_string": context_str,
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_portfolio_state failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOL 3: get_regime
# =============================================================================


@mcp.tool()
async def get_regime(symbol: str) -> dict[str, Any]:
    """
    Detect the current market regime for a symbol.

    Uses ADX for trend strength/direction and ATR percentile for volatility.
    Deterministic — no LLM calls.

    Args:
        symbol: Ticker symbol (e.g., "SPY").

    Returns:
        Dict with keys: success, symbol, trend_regime, volatility_regime,
        confidence, adx, atr, atr_percentile, error.
    """
    try:
        from quant_pod.agents.regime_detector import RegimeDetectorAgent

        detector = RegimeDetectorAgent(symbols=[symbol])
        result = await asyncio.get_event_loop().run_in_executor(
            None, detector.detect_regime, symbol
        )
        return result
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_regime({symbol}) failed: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


# =============================================================================
# TOOL 4: get_recent_decisions
# =============================================================================


@mcp.tool()
async def get_recent_decisions(
    symbol: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Query recent audit trail entries.

    Args:
        symbol: Filter by ticker symbol.  None returns all symbols.
        limit: Maximum number of entries to return.

    Returns:
        Dict with keys: decisions (list of summaries), total.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        from quant_pod.audit.models import AuditQuery

        query = AuditQuery(symbol=symbol or "", limit=limit)
        events = ctx.audit.query(query)
        summaries = [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "agent_name": e.agent_name,
                "symbol": e.symbol,
                "action": e.action,
                "confidence": e.confidence,
                "output_summary": e.output_summary[:200] if e.output_summary else "",
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ]
        return {"success": True, "decisions": summaries, "total": len(summaries)}
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_recent_decisions failed: {e}")
        return {"success": False, "error": str(e), "decisions": [], "total": 0}


# =============================================================================
# TOOL 5: get_system_status
# =============================================================================


@mcp.tool()
async def get_system_status() -> dict[str, Any]:
    """
    Return system health: kill switch state, risk halt, broker mode, session ID.

    Returns:
        Dict with keys: kill_switch_active, kill_switch_reason, risk_halted,
        broker_mode, session_id.
    """
    ctx = require_ctx()
    try:
        ks_status = ctx.kill_switch.status()
        return {
            "success": True,
            "kill_switch_active": ks_status.active,
            "kill_switch_reason": ks_status.reason,
            "risk_halted": ctx.risk_gate.is_halted(),
            "broker_mode": get_broker_mode(),
            "session_id": ctx.session_id,
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_system_status failed: {e}")
        return {"success": False, "error": str(e)}
