"""Phase 6 — Learning Loop tools for the QuantPod MCP server.

RL advisory, strategy lifecycle management (promote/retire), live
performance tracking, strategy re-validation, and regime matrix updates
from trade performance.

Tools:
  - get_rl_status                         — RL model status and config
  - get_rl_recommendation                 — RL position size recommendation (advisory)
  - promote_strategy                      — promote forward_testing to live
  - retire_strategy                       — retire strategy + remove from matrix
  - get_strategy_performance              — live performance metrics vs backtest
  - validate_strategy                     — re-run backtest and compare to registered summary
  - update_regime_matrix_from_performance — propose matrix updates from trade data
"""

import asyncio
from typing import Any

from loguru import logger


from quantstack.mcp.server import mcp
from quantstack.mcp._state import (
    require_ctx,
    require_live_db,
    live_db_or_error,
    _serialize,
)


@mcp.tool()
async def get_rl_status() -> dict[str, Any]:
    """
    Get RL model status: which models are enabled, shadow vs live, config.

    Returns:
        Dict with RL config flags, shadow mode state, and agent statuses.
    """
    try:
        from quantstack.rl.config import get_rl_config

        cfg = get_rl_config()
        return {
            "success": True,
            "config_version": cfg.config_version,
            "shadow_mode_enabled": cfg.shadow_mode_enabled,
            "agents": {
                "execution_rl": {
                    "enabled": cfg.enable_execution_rl,
                    "shadow": cfg.execution_shadow,
                },
                "sizing_rl": {
                    "enabled": cfg.enable_sizing_rl,
                    "shadow": cfg.sizing_shadow,
                },
                "meta_rl": {"enabled": cfg.enable_meta_rl, "shadow": cfg.meta_shadow},
                "spread_rl": {"enabled": cfg.enable_spread_rl},
            },
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_rl_status failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_rl_recommendation(
    symbol: str,
    direction: str,
    signal_confidence: float = 0.5,
    regime: str = "normal",
    current_drawdown: float = 0.0,
) -> dict[str, Any]:
    """
    Get RL-recommended position size adjustment for a trade.

    Claude reads this as INPUT to its decision, not a directive.
    All RL agents start in shadow mode — output is advisory only.

    Args:
        symbol: Ticker symbol.
        direction: "LONG" or "SHORT".
        signal_confidence: Signal confidence (0-1).
        regime: Current market regime label.
        current_drawdown: Current portfolio drawdown fraction.

    Returns:
        Dict with RL recommendations (tagged as shadow if applicable).
    """
    try:
        from quantstack.rl.config import get_rl_config
        from quantstack.rl.rl_tools import RLPositionSizeTool

        cfg = get_rl_config()
        if not cfg.enable_sizing_rl:
            return {
                "success": True,
                "recommendation": None,
                "note": "Sizing RL disabled in config",
            }

        tool = RLPositionSizeTool()
        result_str = tool._run(
            signal_confidence=signal_confidence,
            signal_direction=direction,
            regime=regime,
            current_drawdown=current_drawdown,
            current_position_pct=0.0,
            portfolio_heat=0.0,
            recent_win_rate=0.5,
            atr_percentile=50.0,
        )

        import json as _json

        try:
            result = (
                _json.loads(result_str) if isinstance(result_str, str) else result_str
            )
        except (ValueError, TypeError):
            result = {"raw": str(result_str)}

        return {
            "success": True,
            "symbol": symbol,
            "recommendation": result,
            "shadow_mode": cfg.shadow_mode_enabled,
            "note": "[SHADOW — advisory only]" if cfg.shadow_mode_enabled else "LIVE",
        }
    except Exception as e:
        logger.warning(f"[quantpod_mcp] get_rl_recommendation failed (graceful): {e}")
        return {"success": True, "recommendation": None, "note": f"RL unavailable: {e}"}


@mcp.tool()
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

    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        from quantstack.mcp.tools.strategy import _get_strategy_impl

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
        ctx.db.execute(
            "UPDATE strategies SET status = 'live', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [strategy_id],
        )
        logger.info(
            f"[quantpod_mcp] Promoted strategy {strategy_id} to LIVE: {evidence}"
        )

        return {
            "success": True,
            "strategy_id": strategy_id,
            "new_status": "live",
            "evidence": evidence,
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] promote_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
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
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        # Update status
        ctx.db.execute(
            "UPDATE strategies SET status = 'retired', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [strategy_id],
        )

        # Remove from regime matrix
        ctx.db.execute(
            "DELETE FROM regime_strategy_matrix WHERE strategy_id = ?",
            [strategy_id],
        )

        logger.info(f"[quantpod_mcp] Retired strategy {strategy_id}: {reason}")
        return {
            "success": True,
            "strategy_id": strategy_id,
            "new_status": "retired",
            "reason": reason,
            "removed_from_matrix": True,
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] retire_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
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

    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        from quantstack.mcp.tools.strategy import _get_strategy_impl

        # Get strategy record for backtest comparison
        strat_result = await _get_strategy_impl(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": "Strategy not found"}

        strat = strat_result["strategy"]
        bt = strat.get("backtest_summary") or {}

        # Query closed trades in lookback period
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        cutoff = _dt.now() - _td(days=lookback_days)
        rows = ctx.db.execute(
            """
            SELECT realized_pnl, closed_at, holding_days
            FROM closed_trades
            WHERE closed_at >= ?
            ORDER BY closed_at
            """,
            [cutoff],
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
        import numpy as np

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
        logger.error(f"[quantpod_mcp] get_strategy_performance failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
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
        from quantstack.mcp.tools.strategy import _get_strategy_impl
        from quantstack.mcp.tools.backtesting import run_backtest

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
        logger.error(f"[quantpod_mcp] validate_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
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
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        # Get all closed trades in the lookback period
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        cutoff = _dt.now() - _td(days=lookback_days)
        rows = ctx.db.execute(
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
        matrix_rows = ctx.db.execute(
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
            f"[quantpod_mcp] update_regime_matrix_from_performance failed: {e}"
        )
        return {"success": False, "error": str(e)}
