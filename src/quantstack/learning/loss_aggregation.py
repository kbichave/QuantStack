"""Loss aggregation — groups classified losses, ranks by P&L impact, queues research tasks."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from loguru import logger

from quantstack.db import db_conn


async def run_loss_aggregation() -> dict[str, Any]:
    """Aggregate losses from trailing 30 days, grouped by failure_mode/strategy/symbol.

    Ranks by absolute P&L impact, stores snapshot in loss_aggregation table,
    and auto-generates research tasks for top 3 patterns.
    """
    try:
        return _run_loss_aggregation_sync()
    except Exception as exc:
        logger.error(f"[loss_aggregation] Failed: {exc}")
        return {"groups_found": 0, "tasks_created": 0, "top_patterns": [], "error": str(exc)}


def _run_loss_aggregation_sync() -> dict[str, Any]:
    today = date.today()

    with db_conn() as conn:
        rows = conn.fetchall(
            "SELECT COALESCE(failure_mode, 'unclassified') AS failure_mode, "
            "strategy_id, symbol, pnl, pnl_pct "
            "FROM strategy_outcomes "
            "WHERE pnl < 0 AND closed_at >= NOW() - INTERVAL '30 days'"
        )

    if not rows:
        return {"groups_found": 0, "tasks_created": 0, "top_patterns": []}

    # Group and aggregate
    groups: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        failure_mode = row.get("failure_mode") or "unclassified"
        strategy_id = row["strategy_id"]
        symbol = row.get("symbol", "")
        key = (failure_mode, strategy_id, symbol)
        if key not in groups:
            groups[key] = {"trade_count": 0, "cumulative_pnl": 0.0, "pnl_pcts": []}
        groups[key]["trade_count"] += 1
        groups[key]["cumulative_pnl"] += row["pnl"]
        pnl_pct = row.get("pnl_pct", 0.0)
        groups[key]["pnl_pcts"].append(pnl_pct if pnl_pct else 0.0)

    # Rank by absolute cumulative P&L
    ranked = sorted(groups.items(), key=lambda x: abs(x[1]["cumulative_pnl"]), reverse=True)

    # Store snapshot in loss_aggregation table
    with db_conn() as conn:
        for rank, ((failure_mode, strategy_id, symbol), agg) in enumerate(ranked, 1):
            avg_loss = sum(agg["pnl_pcts"]) / len(agg["pnl_pcts"]) if agg["pnl_pcts"] else 0.0
            conn.execute(
                "INSERT INTO loss_aggregation (date, failure_mode, strategy_id, symbol, "
                "trade_count, cumulative_pnl, avg_loss_pct, rank) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (date, failure_mode, strategy_id, symbol) "
                "DO UPDATE SET trade_count=EXCLUDED.trade_count, cumulative_pnl=EXCLUDED.cumulative_pnl, "
                "avg_loss_pct=EXCLUDED.avg_loss_pct, rank=EXCLUDED.rank",
                [today, failure_mode, strategy_id, symbol, agg["trade_count"], agg["cumulative_pnl"], round(avg_loss, 4), rank],
            )

    # Auto-generate research tasks for top 3 patterns
    top_patterns = []
    tasks_created = 0
    with db_conn() as conn:
        for (failure_mode, strategy_id, symbol), agg in ranked[:3]:
            priority = min(9, int(abs(agg["cumulative_pnl"]) / 100))
            context = json.dumps({
                "failure_mode": failure_mode,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "trade_count": agg["trade_count"],
                "cumulative_pnl": agg["cumulative_pnl"],
            })
            conn.execute(
                "INSERT INTO research_queue (task_type, priority, context_json, source) "
                "VALUES (%s, %s, %s, 'loss_aggregation')",
                [failure_mode, priority, context],
            )
            tasks_created += 1
            top_patterns.append({
                "failure_mode": failure_mode,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "trade_count": agg["trade_count"],
                "cumulative_pnl": agg["cumulative_pnl"],
                "priority": priority,
            })

    return {
        "groups_found": len(groups),
        "tasks_created": tasks_created,
        "top_patterns": top_patterns,
    }
