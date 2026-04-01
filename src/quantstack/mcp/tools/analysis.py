# Copyright 2024 QuantStack Contributors
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

from quantstack.agents.regime_detector import RegimeDetectorAgent
from quantstack.audit.models import AuditQuery
from quantstack.execution.broker_factory import get_broker_mode
from quantstack.mcp._state import (
    _serialize,
    live_db_or_error,
    require_ctx,
)
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.tools._registry import domain
from quantstack.mcp.domains import Domain


# =============================================================================
# TOOL 2: get_portfolio_state
# =============================================================================


@domain(Domain.PORTFOLIO, Domain.EXECUTION, Domain.RESEARCH)
@tool_def()
async def get_portfolio_state() -> dict[str, Any]:
    """
    Return the current portfolio state: positions, cash, equity, and P&L.

    WHEN TO USE: At the start of every trading iteration to ground decisions in
    current exposure, at any point you need cash/equity for sizing, and before
    close_position to confirm a position exists.
    WHEN NOT TO USE: Do not poll repeatedly within the same iteration — the
    snapshot does not change unless a trade was executed.
    WORKFLOW: get_system_status → THIS → get_signal_brief / execute_trade
    RELATED: get_risk_metrics, get_system_status, execute_trade

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
        logger.error(f"[quantstack_mcp] get_portfolio_state failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOL 3: get_regime
# =============================================================================


@domain(Domain.SIGNALS, Domain.DATA, Domain.RESEARCH, Domain.ML, Domain.FINRL, Domain.INTEL, Domain.RISK)
@tool_def()
async def get_regime(symbol: str) -> dict[str, Any]:
    """
    Detect the current market regime for a symbol using ADX/ATR classification.

    WHEN TO USE: Cross-cutting — needed by almost every workflow. Call before
    get_signal_brief to pass a pre-computed regime (saves one redundant
    detection), before strategy selection to filter the regime-strategy matrix,
    and in research loops to label backtest periods.
    WHEN NOT TO USE: Do not call if you already have a regime dict from a
    prior get_signal_brief in the same iteration (it embeds regime_used).
    SIGNAL TIER: tier_4_regime_macro
    WORKFLOW: THIS → get_signal_brief / strategy selection / backtest labeling
    RELATED: get_signal_brief, run_multi_signal_brief

    Args:
        symbol: Ticker symbol (e.g., "SPY").

    Returns:
        Dict with keys: success, symbol, trend_regime, volatility_regime,
        confidence, adx, atr, atr_percentile, error.
    """
    try:
        detector = RegimeDetectorAgent(symbols=[symbol])
        result = await asyncio.get_event_loop().run_in_executor(
            None, detector.detect_regime, symbol
        )
        return result
    except Exception as e:
        logger.error(f"[quantstack_mcp] get_regime({symbol}) failed: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


# =============================================================================
# TOOL 4: get_recent_decisions
# =============================================================================


@domain(Domain.PORTFOLIO)
@tool_def()
async def get_recent_decisions(
    symbol: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Query recent audit trail entries for decision review.

    WHEN TO USE: During after-market review to inspect today's decisions, when
    debugging why a trade was taken or rejected, or to check for duplicate
    actions before entering a new trade.
    WHEN NOT TO USE: Do not use for detailed fill data — use get_fills instead.
    Do not use for risk metrics — use get_risk_metrics.
    WORKFLOW: execute_trade / close_position → THIS (post-hoc review)
    RELATED: get_audit_trail, get_fills, get_portfolio_state

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
        logger.error(f"[quantstack_mcp] get_recent_decisions failed: {e}")
        return {"success": False, "error": str(e), "decisions": [], "total": 0}


# =============================================================================
# TOOL 5: get_system_status
# =============================================================================


@domain(Domain.EXECUTION, Domain.SIGNALS, Domain.PORTFOLIO)
@tool_def()
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
        logger.error(f"[quantstack_mcp] get_system_status failed: {e}")
        return {"success": False, "error": str(e)}


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
