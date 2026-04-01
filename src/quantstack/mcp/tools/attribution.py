# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""P&L Attribution MCP tools — equity curve and strategy-level P&L.

Exposes the EquityTracker and BenchmarkTracker data to LLM agents so the
research loop can answer: "Which strategy is losing money?", "Are we
beating SPY?", "Which decision step causes the most losses?"

Tools:
  - get_daily_equity   — daily equity curve + headline performance stats
  - get_strategy_pnl   — per-strategy P&L with optional step credit breakdown
"""

from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantstack.db import pg_conn
from quantstack.mcp._state import live_db_or_error
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain



@domain(Domain.PORTFOLIO)
@tool_def()
async def get_daily_equity(
    start_date: str = "",
    end_date: str = "",
    include_summary: bool = True,
) -> dict[str, Any]:
    """
    Query the daily equity curve and headline performance stats.

    Returns daily NAV, return, drawdown from the daily_equity table.
    Optionally includes summary stats (total return, Sharpe, Sortino,
    max drawdown, benchmark comparison).

    Args:
        start_date: ISO date string (YYYY-MM-DD). Default: last 30 days.
        end_date: ISO date string (YYYY-MM-DD). Default: today.
        include_summary: Include headline stats (Sharpe, Sortino, etc).

    Returns:
        Dict with equity_curve list and optional summary dict.
    """
    _, err = live_db_or_error()
    if err:
        return err
    try:
        from quantstack.performance.benchmark import BenchmarkTracker
        from quantstack.performance.equity_tracker import EquityTracker

        sd = date.fromisoformat(start_date) if start_date else date.today() - timedelta(days=30)
        ed = date.fromisoformat(end_date) if end_date else date.today()

        with pg_conn() as conn:
            tracker = EquityTracker(conn)
            curve = tracker.get_equity_curve(start_date=sd, end_date=ed)

            # Serialize dates for JSON
            for row in curve:
                for k, v in row.items():
                    if isinstance(v, date):
                        row[k] = str(v)

            result: dict[str, Any] = {
                "success": True,
                "equity_curve": curve,
                "count": len(curve),
            }

            if include_summary:
                result["summary"] = tracker.get_summary()

                # Include latest benchmark comparison if available
                try:
                    bench = BenchmarkTracker(conn)
                    comparisons = bench.get_comparison("SPY", start_date=sd, end_date=ed)
                    if comparisons:
                        # Return the most recent comparison per window
                        latest = {}
                        for c in comparisons:
                            for k, v in c.items():
                                if isinstance(v, date):
                                    c[k] = str(v)
                            latest[c["window_days"]] = c
                        result["benchmark_comparison"] = list(latest.values())
                except Exception as exc:
                    logger.debug(f"[attribution] Benchmark comparison unavailable: {exc}")

        return result
    except Exception as e:
        logger.error(f"[quantstack_mcp] get_daily_equity failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.PORTFOLIO)
@tool_def()
async def get_strategy_pnl(
    strategy_id: str = "",
    start_date: str = "",
    end_date: str = "",
    include_credits: bool = False,
) -> dict[str, Any]:
    """
    Query per-strategy P&L attribution with optional step credit breakdown.

    Shows realized/unrealized P&L, trade counts, and win/loss per strategy.
    When include_credits=True, also aggregates step_credits to show which
    decision steps (signal, regime, strategy_selection, sizing, debate)
    contribute most to losses for the given strategy.

    Args:
        strategy_id: Filter to a single strategy. Empty = all strategies.
        start_date: ISO date string. Default: last 30 days.
        end_date: ISO date string. Default: today.
        include_credits: Include step-level credit breakdown.

    Returns:
        Dict with strategy_pnl list and optional credit_breakdown.
    """
    _, err = live_db_or_error()
    if err:
        return err
    try:
        from quantstack.performance.equity_tracker import EquityTracker

        sd = date.fromisoformat(start_date) if start_date else date.today() - timedelta(days=30)
        ed = date.fromisoformat(end_date) if end_date else date.today()

        with pg_conn() as conn:
            tracker = EquityTracker(conn)
            pnl_rows = tracker.get_strategy_pnl(
                strategy_id=strategy_id or None,
                start_date=sd,
                end_date=ed,
            )

            # Serialize dates
            for row in pnl_rows:
                for k, v in row.items():
                    if isinstance(v, date):
                        row[k] = str(v)

            # Compute aggregates per strategy
            aggregates: dict[str, dict] = {}
            for row in pnl_rows:
                sid = row.get("strategy_id", "unknown")
                if sid not in aggregates:
                    aggregates[sid] = {
                        "strategy_id": sid,
                        "total_realized_pnl": 0.0,
                        "total_trades": 0,
                        "total_wins": 0,
                        "total_losses": 0,
                        "days_active": 0,
                    }
                agg = aggregates[sid]
                agg["total_realized_pnl"] += row.get("realized_pnl", 0) or 0
                agg["total_trades"] += row.get("num_trades", 0) or 0
                agg["total_wins"] += row.get("win_count", 0) or 0
                agg["total_losses"] += row.get("loss_count", 0) or 0
                agg["days_active"] += 1

            for agg in aggregates.values():
                agg["total_realized_pnl"] = round(agg["total_realized_pnl"], 2)
                total = agg["total_trades"]
                agg["win_rate"] = round(agg["total_wins"] / total * 100, 1) if total > 0 else 0.0

            result: dict[str, Any] = {
                "success": True,
                "strategy_pnl_daily": pnl_rows,
                "strategy_aggregates": sorted(
                    aggregates.values(),
                    key=lambda x: x["total_realized_pnl"],
                ),
                "count": len(pnl_rows),
            }

            if include_credits:
                try:
                    # Aggregate step credits for losing trades
                    credit_query = """
                        SELECT step_type,
                               ROUND(AVG(credit_score), 3) as avg_credit,
                               ROUND(MIN(credit_score), 3) as worst_credit,
                               COUNT(*) as observations
                        FROM step_credits
                        WHERE credit_score < 0
                    """
                    params: list[Any] = []

                    if strategy_id:
                        # Join to closed_trades to filter by strategy
                        credit_query = """
                            SELECT sc.step_type,
                                   ROUND(AVG(sc.credit_score), 3) as avg_credit,
                                   ROUND(MIN(sc.credit_score), 3) as worst_credit,
                                   COUNT(*) as observations
                            FROM step_credits sc
                            JOIN closed_trades ct ON sc.trade_id = ct.id
                            WHERE sc.credit_score < 0
                              AND ct.strategy_id = ?
                        """
                        params = [strategy_id]

                    credit_query += " GROUP BY step_type ORDER BY avg_credit ASC"
                    credit_rows = conn.execute(credit_query, params).fetchall()
                    result["credit_breakdown"] = [
                        {
                            "step_type": r[0],
                            "avg_credit": float(r[1]),
                            "worst_credit": float(r[2]),
                            "observations": int(r[3]),
                        }
                        for r in credit_rows
                    ]
                except Exception as exc:
                    logger.debug(f"[attribution] Credit breakdown unavailable: {exc}")
                    result["credit_breakdown"] = []

        return result
    except Exception as e:
        logger.error(f"[quantstack_mcp] get_strategy_pnl failed: {e}")
        return {"success": False, "error": str(e)}


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
