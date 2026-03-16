# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1 MCP tools — read-only analysis and system introspection.

Tools:
  - run_analysis       — run TradingCrew, return DailyBrief
  - get_portfolio_state — positions, cash, equity, P&L
  - get_regime         — ADX/ATR regime classification
  - get_recent_decisions — audit trail query
  - get_system_status  — kill switch, risk halt, broker mode
"""

import asyncio
import time
from typing import Any

from loguru import logger

from quant_pod.mcp.server import mcp
from quant_pod.mcp._state import (
    require_ctx,
    require_live_db,
    live_db_or_error,
    _serialize,
    _read_memory_file,
    populate_ic_cache_from_result,
    is_degraded,
    get_degraded_reason,
)
from quant_pod.execution.broker_factory import get_broker_mode


# =============================================================================
# TOOL 1: run_analysis
# =============================================================================


@mcp.tool()
async def run_analysis(
    symbol: str,
    regime: dict[str, Any] | None = None,
    include_historical_context: bool = True,
) -> dict[str, Any]:
    """
    Run TradingCrew analysis for a symbol and return a DailyBrief.

    The crew runs all ICs (data, technicals, quant, risk, market monitor),
    Pod Managers compile their findings, and the Trading Assistant synthesizes
    a structured DailyBrief.  The SuperTrader is NOT invoked — Claude Code
    acts as the decision maker.

    Args:
        symbol: Ticker symbol (e.g., "SPY", "AAPL").
        regime: Pre-computed regime dict.  If None, regime is detected
                automatically using ADX/ATR indicators.
        include_historical_context: Whether to load blackboard history
                                    as context for the crew.

    Returns:
        Dict with keys: success, daily_brief, regime_used, elapsed_seconds, error.
    """
    ctx = require_ctx()
    start = time.monotonic()

    try:
        # 1. Detect regime if not provided
        if regime is None:
            from quant_pod.agents.regime_detector import RegimeDetectorAgent

            detector = RegimeDetectorAgent(symbols=[symbol])
            regime_result = await asyncio.get_event_loop().run_in_executor(
                None, detector.detect_regime, symbol
            )
            if regime_result.get("success"):
                regime = {
                    "trend": regime_result.get("trend_regime", "unknown"),
                    "volatility": regime_result.get("volatility_regime", "normal"),
                    "confidence": regime_result.get("confidence", 0.5),
                }
            else:
                regime = {"trend": "unknown", "volatility": "normal", "confidence": 0.5}

        # 2. Build portfolio context
        portfolio = _serialize(ctx.portfolio.get_snapshot())

        # 3. Load historical context from blackboard
        historical_context = ""
        if include_historical_context:
            historical_context = ctx.blackboard.read_as_context(symbol=symbol, limit=10)

        # 3b. Inject cross-session context from .claude/memory files.
        #     strategy_context tells the crew which strategies are active and
        #     what regimes they target — so the assistant can frame its synthesis
        #     in terms of the strategies Claude Code is actually considering.
        #     session_notes carries recent handoff findings (IC biases, alerts).
        strategy_context = _read_memory_file("strategy_registry.md", max_chars=2000)
        session_notes = _read_memory_file("session_handoffs.md", max_chars=1000)

        # 4. Run crew in stop-at-assistant mode (sync call in thread pool)
        from quant_pod.crews.trading_crew import run_analysis_only

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_analysis_only(
                symbol=symbol,
                regime=regime,
                portfolio=portfolio,
                historical_context=historical_context,
                strategy_context=strategy_context,
                session_notes=session_notes,
            ),
        )

        # 5. Extract DailyBrief from crew result
        brief = None
        if hasattr(result, "pydantic") and result.pydantic is not None:
            brief = _serialize(result.pydantic)
        elif hasattr(result, "json_dict") and result.json_dict is not None:
            brief = result.json_dict
        elif isinstance(result, dict):
            brief = result
        else:
            brief = {"raw_output": str(result)}

        # 6. Populate per-IC output cache (non-blocking best-effort)
        populate_ic_cache_from_result(symbol, result)

        elapsed = time.monotonic() - start
        return {
            "success": True,
            "daily_brief": brief,
            "regime_used": regime,
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"[quantpod_mcp] run_analysis({symbol}) failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "regime_used": regime or {},
            "elapsed_seconds": round(elapsed, 2),
        }


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
