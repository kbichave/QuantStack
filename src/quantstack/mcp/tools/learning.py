"""Phase 6 — Learning Loop tools for the QuantStack MCP server.

Strategy lifecycle management (promote/retire), live performance tracking,
strategy re-validation, and regime matrix updates from trade performance.

RL model tools have moved to quantstack.mcp.tools.finrl_tools.

Tools:
  - promote_strategy                      — promote forward_testing to live
  - retire_strategy                       — retire strategy + remove from matrix
  - get_strategy_performance              — live performance metrics vs backtest
  - validate_strategy                     — re-run backtest and compare to registered summary
  - update_regime_matrix_from_performance — propose matrix updates from trade data
"""

from datetime import datetime as _dt
from datetime import timedelta as _td
from typing import Any

import numpy as np
from loguru import logger

from quantstack.db import pg_conn
from quantstack.mcp._state import (
    _serialize,
    live_db_or_error,
    require_ctx,
    require_live_db,
)
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._impl import get_strategy_impl as _get_strategy_impl
from quantstack.mcp.tools._impl import run_backtest_impl as run_backtest
from quantstack.mcp.tools._registry import domain
from quantstack.mcp.tools._tool_def import tool_def



@domain(Domain.RESEARCH)
@tool_def()
async def promote_strategy(
    strategy_id: str,
    evidence: str,
) -> dict[str, Any]:
    """
    Promote a strategy to "live" status after validation.

    Validates:
      - Backtest exists with positive Sharpe
      - Walk-forward OOS passes (if available)
      - Current status is "forward_testing"
    The evidence field documents why promotion is justified — required.

    Args:
        strategy_id: Strategy to promote.
        evidence: REQUIRED. Justification for promotion.

    Returns:
        Success with updated record, or rejection with failed criteria.
    """

    _, err = live_db_or_error()
    if err:
        return err
    try:
        strat_result = await _get_strategy_impl(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": "Strategy not found"}

        strat = strat_result["strategy"]
        failures = []

        # Check current status
        if strat.get("status") != "forward_testing":
            failures.append(
                f"Status is '{strat.get('status')}', expected 'forward_testing'"
            )

        # Check backtest exists with positive Sharpe
        bt = strat.get("backtest_summary") or {}
        if not bt:
            failures.append("No backtest_summary — run backtest first")
        elif bt.get("sharpe_ratio", 0) <= 0:
            failures.append(f"Backtest Sharpe {bt.get('sharpe_ratio', 0):.2f} <= 0")

        # Check walk-forward if available
        wf = strat.get("walkforward_summary") or {}
        if wf:
            oos_sharpe = wf.get("oos_sharpe_mean", 0)
            if oos_sharpe <= 0:
                failures.append(f"OOS Sharpe {oos_sharpe:.2f} <= 0")

        if failures:
            return {
                "success": False,
                "error": "Promotion criteria not met",
                "failures": failures,
                "strategy_id": strategy_id,
            }

        # Promote
        with pg_conn() as conn:
            conn.execute(
                "UPDATE strategies SET status = 'live', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
                [strategy_id],
            )
        logger.info(
            f"[quantstack_mcp] Promoted strategy {strategy_id} to LIVE: {evidence}"
        )

        return {
            "success": True,
            "strategy_id": strategy_id,
            "new_status": "live",
            "evidence": evidence,
        }
    except Exception as e:
        logger.error(f"[quantstack_mcp] promote_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def retire_strategy(
    strategy_id: str,
    reason: str,
) -> dict[str, Any]:
    """
    Retire a strategy and remove it from the regime-strategy matrix.

    The reason field is required — retirement reasons are learning data
    for /reflect sessions.

    Args:
        strategy_id: Strategy to retire.
        reason: REQUIRED. Why this strategy is being retired.

    Returns:
        Confirmation with the retirement details.
    """
    _, err = live_db_or_error()
    if err:
        return err
    try:
        with pg_conn() as conn:
            # Update status
            conn.execute(
                "UPDATE strategies SET status = 'retired', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
                [strategy_id],
            )

            # Remove from regime matrix
            conn.execute(
                "DELETE FROM regime_strategy_matrix WHERE strategy_id = ?",
                [strategy_id],
            )

        logger.info(f"[quantstack_mcp] Retired strategy {strategy_id}: {reason}")
        return {
            "success": True,
            "strategy_id": strategy_id,
            "new_status": "retired",
            "reason": reason,
            "removed_from_matrix": True,
        }
    except Exception as e:
        logger.error(f"[quantstack_mcp] retire_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def get_strategy_performance(
    strategy_id: str,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """
    Compute live performance metrics for a strategy over a lookback period.

    Queries closed_trades linked to this strategy (by session correlation),
    computes win rate, average win/loss, Sharpe approximation, and compares
    against the registered backtest_summary.

    Args:
        strategy_id: Strategy to evaluate.
        lookback_days: Number of days to look back.

    Returns:
        Dict with live metrics, backtest comparison, and degradation flag.
    """

    _, err = live_db_or_error()
    if err:
        return err
    try:
        # Get strategy record for backtest comparison
        strat_result = await _get_strategy_impl(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": "Strategy not found"}

        strat = strat_result["strategy"]
        bt = strat.get("backtest_summary") or {}

        # Query closed trades in lookback period, filtered by strategy
        cutoff = _dt.now() - _td(days=lookback_days)
        with pg_conn() as conn:
            rows = conn.execute(
                """
                SELECT realized_pnl, closed_at, holding_days
                FROM closed_trades
                WHERE closed_at >= ?
                  AND strategy_id = ?
                ORDER BY closed_at
                """,
                [cutoff, strategy_id],
            ).fetchall()

        if not rows:
            return {
                "success": True,
                "strategy_id": strategy_id,
                "lookback_days": lookback_days,
                "total_trades": 0,
                "note": "No closed trades in lookback period",
            }

        pnls = [float(r[0]) for r in rows]
        total_trades = len(pnls)
        winners = sum(1 for p in pnls if p > 0)
        win_rate = winners / total_trades * 100
        avg_win = sum(p for p in pnls if p > 0) / max(1, winners)
        avg_loss = sum(p for p in pnls if p < 0) / max(1, total_trades - winners)
        total_pnl = sum(pnls)

        # Simple Sharpe approximation
        pnl_arr = np.array(pnls)
        live_sharpe = (
            float(np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-10) * np.sqrt(252))
            if len(pnl_arr) > 1
            else 0.0
        )

        # Compare to backtest
        bt_sharpe = bt.get("sharpe_ratio", 0)
        degradation_pct = 0.0
        if bt_sharpe > 0:
            degradation_pct = (bt_sharpe - live_sharpe) / bt_sharpe * 100

        return {
            "success": True,
            "strategy_id": strategy_id,
            "lookback_days": lookback_days,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "total_pnl": round(total_pnl, 2),
            "live_sharpe": round(live_sharpe, 4),
            "backtest_sharpe": round(bt_sharpe, 4),
            "degradation_pct": round(degradation_pct, 2),
            "degraded": degradation_pct > 30,
        }
    except Exception as e:
        logger.error(f"[quantstack_mcp] get_strategy_performance failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def validate_strategy(strategy_id: str) -> dict[str, Any]:
    """
    Re-validate a strategy by comparing current backtest to registered summary.

    Runs a fresh backtest (if price data is available) and compares metrics
    to the stored backtest_summary. Flags significant degradation.

    Args:
        strategy_id: Strategy to validate.

    Returns:
        Dict with still_valid flag, current vs historical metrics, degradation.
    """
    try:
        strat_result = await _get_strategy_impl(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": "Strategy not found"}

        strat = strat_result["strategy"]
        bt = strat.get("backtest_summary") or {}

        if not bt:
            return {
                "success": True,
                "strategy_id": strategy_id,
                "still_valid": False,
                "reason": "No backtest_summary to compare against",
            }

        symbol = bt.get("symbol", "SPY")
        bt_sharpe = bt.get("sharpe_ratio", 0)
        bt_dd = bt.get("max_drawdown", 0)

        # Re-run backtest on recent data
        fresh = await run_backtest(
            strategy_id=strategy_id,
            symbol=symbol,
        )

        if not fresh.get("success"):
            return {
                "success": True,
                "strategy_id": strategy_id,
                "still_valid": None,
                "reason": f"Could not re-run backtest: {fresh.get('error', 'unknown')}",
            }

        fresh_sharpe = fresh.get("sharpe_ratio", 0)
        fresh_dd = fresh.get("max_drawdown", 0)

        sharpe_degradation = 0.0
        if bt_sharpe > 0:
            sharpe_degradation = (bt_sharpe - fresh_sharpe) / bt_sharpe * 100

        still_valid = sharpe_degradation < 30 and fresh_sharpe > 0

        return {
            "success": True,
            "strategy_id": strategy_id,
            "still_valid": still_valid,
            "original_sharpe": round(bt_sharpe, 4),
            "fresh_sharpe": round(fresh_sharpe, 4),
            "sharpe_degradation_pct": round(sharpe_degradation, 2),
            "original_max_dd": round(bt_dd, 2),
            "fresh_max_dd": round(fresh_dd, 2),
        }
    except Exception as e:
        logger.error(f"[quantstack_mcp] validate_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def update_regime_matrix_from_performance(
    lookback_days: int = 60,
) -> dict[str, Any]:
    """
    Propose updated regime-strategy allocations based on actual trade performance.

    Analyzes closed trades, groups by regime context, and proposes
    updated allocation weights. Does NOT auto-apply — returns proposals
    for /reflect or /meta to review and apply via set_regime_allocation.

    Args:
        lookback_days: Days of trade history to analyze.

    Returns:
        Dict with proposed changes per regime + reasoning.
    """
    _, err = live_db_or_error()
    if err:
        return err
    try:
        # Get all closed trades and current matrix in a single pooled connection
        cutoff = _dt.now() - _td(days=lookback_days)
        with pg_conn() as conn:
            rows = conn.execute(
                """
                SELECT symbol, side, realized_pnl, closed_at
                FROM closed_trades
                WHERE closed_at >= ?
                """,
                [cutoff],
            ).fetchall()

            if not rows:
                return {
                    "success": True,
                    "proposals": [],
                    "note": f"No closed trades in last {lookback_days} days. Cannot propose changes.",
                }

            # Get current matrix
            matrix_rows = conn.execute(
                "SELECT regime, strategy_id, allocation_pct FROM regime_strategy_matrix"
            ).fetchall()

        current_matrix = {}
        for r in matrix_rows:
            current_matrix.setdefault(r[0], {})[r[1]] = r[2]

        # Since we don't have per-trade regime labels in closed_trades,
        # we can report aggregate stats and suggest directional changes.
        total_pnl = sum(float(r[2]) for r in rows)
        total_trades = len(rows)
        win_rate = sum(1 for r in rows if float(r[2]) > 0) / total_trades * 100

        return {
            "success": True,
            "lookback_days": lookback_days,
            "total_trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 2),
            "current_matrix": current_matrix,
            "proposals": [],
            "note": (
                "Per-regime trade attribution requires strategy_id on closed_trades "
                "(Phase 3 execute_trade logs strategy_id in audit trail, not in "
                "closed_trades). Use /reflect to manually review and update allocations "
                "based on trade journal patterns."
            ),
        }
    except Exception as e:
        logger.error(
            f"[quantstack_mcp] update_regime_matrix_from_performance failed: {e}"
        )
        return {"success": False, "error": str(e)}


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
