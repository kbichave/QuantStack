"""Phase 5 — Meta Orchestration tools for the QuantPod MCP server.

Portfolio-level tools for regime-strategy allocation, multi-symbol analysis,
and signal conflict resolution.

Tools:
  - get_regime_strategies        — get strategy allocations for a regime
  - set_regime_allocation        — set/update regime-strategy allocation matrix
  - run_multi_analysis           — run analysis for multiple symbols
  - resolve_portfolio_conflicts  — resolve signal conflicts across strategies
"""

import asyncio
from typing import Any

from loguru import logger

from quant_pod.mcp.server import mcp
from quant_pod.mcp._state import require_ctx, require_live_db, live_db_or_error, _serialize


@mcp.tool()
async def get_regime_strategies(regime: str) -> dict[str, Any]:
    """
    Get strategy allocations for a given regime from the matrix.

    Args:
        regime: Regime label (e.g., "trending_up", "ranging").

    Returns:
        Dict with list of (strategy_id, allocation_pct, confidence).
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        rows = ctx.db.execute(
            "SELECT strategy_id, allocation_pct, confidence, last_updated "
            "FROM regime_strategy_matrix WHERE regime = ? ORDER BY allocation_pct DESC",
            [regime],
        ).fetchall()

        allocations = [
            {
                "strategy_id": r[0],
                "allocation_pct": r[1],
                "confidence": r[2],
                "last_updated": str(r[3]) if r[3] else None,
            }
            for r in rows
        ]
        return {
            "success": True,
            "regime": regime,
            "allocations": allocations,
            "total": len(allocations),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_regime_strategies failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def set_regime_allocation(
    regime: str,
    allocations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Set or update strategy allocations for a regime.

    Upserts into the regime_strategy_matrix. This is how /reflect updates
    the matrix based on accumulated performance data.

    Args:
        regime: Regime label.
        allocations: List of dicts with strategy_id, allocation_pct, confidence (optional).

    Returns:
        Confirmation with the updated allocations.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        # Validate total allocation <= 1.0
        total = sum(a.get("allocation_pct", 0) for a in allocations)
        if total > 1.0:
            return {
                "success": False,
                "error": f"Total allocation {total:.0%} exceeds 100%. Reduce allocations.",
            }

        for alloc in allocations:
            strategy_id = alloc.get("strategy_id")
            allocation_pct = alloc.get("allocation_pct", 0)
            confidence = alloc.get("confidence", 0.5)

            if not strategy_id:
                continue

            # Upsert: try update, then insert
            ctx.db.execute(
                "UPDATE regime_strategy_matrix "
                "SET allocation_pct = ?, confidence = ?, last_updated = CURRENT_TIMESTAMP "
                "WHERE regime = ? AND strategy_id = ?",
                [allocation_pct, confidence, regime, strategy_id],
            ).fetchone()

            # Check if row existed
            exists = ctx.db.execute(
                "SELECT 1 FROM regime_strategy_matrix WHERE regime = ? AND strategy_id = ?",
                [regime, strategy_id],
            ).fetchone()

            if not exists:
                ctx.db.execute(
                    "INSERT INTO regime_strategy_matrix (regime, strategy_id, allocation_pct, confidence) "
                    "VALUES (?, ?, ?, ?)",
                    [regime, strategy_id, allocation_pct, confidence],
                )

        logger.info(
            f"[quantpod_mcp] Updated regime matrix for '{regime}': {len(allocations)} strategies"
        )
        return await get_regime_strategies.fn(regime)
    except Exception as e:
        logger.error(f"[quantpod_mcp] set_regime_allocation failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def run_multi_analysis(
    symbols: list[str],
) -> dict[str, Any]:
    """
    Run TradingCrew analysis for multiple symbols.

    Runs run_analysis sequentially for each symbol and collects all DailyBriefs.

    Args:
        symbols: List of ticker symbols to analyze.

    Returns:
        Dict with list of per-symbol results.
    """
    from quant_pod.mcp.tools.analysis import run_analysis

    results = []
    for symbol in symbols:
        result = await run_analysis.fn(symbol=symbol)
        results.append({"symbol": symbol, **result})

    successes = sum(1 for r in results if r.get("success"))
    return {
        "success": successes > 0,
        "results": results,
        "symbols_analyzed": len(symbols),
        "symbols_succeeded": successes,
        "symbols_failed": len(symbols) - successes,
    }


@mcp.tool()
async def resolve_portfolio_conflicts(
    proposed_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Resolve signal conflicts across multiple strategies for the same symbols.

    Rules:
      - Same symbol, different directions: high confidence wins, or SKIP if both high
      - Same symbol, same direction: merge with conservative sizing

    Args:
        proposed_trades: List of trade dicts, each with:
            symbol, action, confidence, strategy_id, capital_pct.

    Returns:
        Dict with resolved_trades, resolutions, conflicts_count.
    """
    try:
        from quant_pod.mcp.allocation import resolve_conflicts

        result = resolve_conflicts(proposed_trades)
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"[quantpod_mcp] resolve_portfolio_conflicts failed: {e}")
        return {"success": False, "error": str(e)}
