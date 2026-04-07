"""
Daily loss analysis pipeline — 5-stage error-driven research.

Stages:
  1. collect_daily_losers  — query closed_trades for negative P&L on a given date
  2. classify_losses       — apply failure_taxonomy rules to each loser
  3. aggregate_failure_modes — upsert rolling 30-day stats into failure_mode_stats
  4. prioritize_failure_modes — rank failure modes by cumulative P&L impact
  5. generate_research_tasks — insert top failure modes into research_queue

Orchestrator: run_daily_loss_analysis() runs all stages in sequence.
Called from scheduled_refresh.run_eod_refresh() after market data sync.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantstack.db import pg_conn
from quantstack.learning.failure_taxonomy import FailureMode, classify_failure


def collect_daily_losers(
    conn: Any,
    trade_date: date | None = None,
) -> list[dict]:
    """Query closed_trades + fills for negative realized P&L on *trade_date*.

    Returns a list of dicts with keys:
        strategy_id, symbol, entry_regime, exit_regime, holding_period_days,
        signal_strength_at_entry, realized_pnl, realized_pnl_pct, slippage_pct,
        entry_price, exit_price.
    """
    if trade_date is None:
        trade_date = date.today()

    rows = conn.execute(
        """
        SELECT
            ct.strategy_id,
            ct.symbol,
            ct.regime_at_entry   AS entry_regime,
            ct.regime_at_exit    AS exit_regime,
            ct.holding_days      AS holding_period_days,
            ct.realized_pnl,
            ct.entry_price,
            ct.exit_price,
            COALESCE(f.slippage_bps, 0.0) AS slippage_bps
        FROM closed_trades ct
        LEFT JOIN fills f
            ON f.symbol = ct.symbol
           AND f.filled_at::date = ct.closed_at::date
        WHERE ct.realized_pnl < 0
          AND ct.closed_at::date = %s
        """,
        [trade_date],
    ).fetchall()

    losers: list[dict] = []
    for row in rows:
        # Handle both dict-style (PgConnection) and tuple-style (raw cursor) rows
        if isinstance(row, dict):
            strategy_id = row.get("strategy_id", "") or ""
            symbol = row.get("symbol", "") or ""
            entry_regime = row.get("entry_regime", "unknown") or "unknown"
            exit_regime = row.get("exit_regime", "unknown") or "unknown"
            holding_period_days = row.get("holding_period_days", 0) or 0
            realized_pnl = row.get("realized_pnl", 0.0) or 0.0
            entry_price = row.get("entry_price", 0.0) or 0.0
            exit_price = row.get("exit_price", 0.0) or 0.0
            slippage_bps = row.get("slippage_bps", 0.0) or 0.0
        else:
            strategy_id = row[0] or ""
            symbol = row[1] or ""
            entry_regime = row[2] or "unknown"
            exit_regime = row[3] or "unknown"
            holding_period_days = row[4] or 0
            realized_pnl = row[5] or 0.0
            entry_price = row[6] or 0.0
            exit_price = row[7] or 0.0
            slippage_bps = row[8] or 0.0

        pnl_pct = (realized_pnl / (entry_price * 1.0)) if entry_price else 0.0
        slippage_pct = slippage_bps / 10_000.0

        losers.append({
            "strategy_id": strategy_id,
            "symbol": symbol,
            "entry_regime": entry_regime,
            "exit_regime": exit_regime,
            "holding_period_days": holding_period_days,
            "signal_strength_at_entry": 0.0,  # not stored in closed_trades yet
            "realized_pnl": realized_pnl,
            "realized_pnl_pct": round(pnl_pct, 6),
            "slippage_pct": round(slippage_pct, 6),
            "entry_price": entry_price,
            "exit_price": exit_price,
        })

    logger.info("[loss_analyzer] Collected %d losers for %s", len(losers), trade_date)
    return losers


def classify_losses(losers: list[dict]) -> list[dict]:
    """Apply failure_taxonomy.classify_failure() to each loser.

    Adds a 'failure_mode' key (FailureMode enum value string) to each dict.
    Falls back to LLM-based classification for UNCLASSIFIED if available.
    """
    for loser in losers:
        mode = classify_failure(
            realized_pnl_pct=loser["realized_pnl_pct"],
            regime_at_entry=loser["entry_regime"],
            regime_at_exit=loser["exit_regime"],
            strategy_id=loser["strategy_id"],
            symbol=loser["symbol"],
            entry_price=loser["entry_price"],
            exit_price=loser["exit_price"],
        )

        # Apply new extended rules before accepting UNCLASSIFIED
        if mode == FailureMode.UNCLASSIFIED:
            mode = _apply_extended_rules(loser, mode)

        loser["failure_mode"] = mode.value

    # Optional LLM fallback for remaining UNCLASSIFIED
    unclassified = [l for l in losers if l["failure_mode"] == FailureMode.UNCLASSIFIED.value]
    if unclassified:
        _try_llm_classify(unclassified)

    classified_count = sum(1 for l in losers if l["failure_mode"] != FailureMode.UNCLASSIFIED.value)
    logger.info(
        "[loss_analyzer] Classified %d/%d losses (%.0f%% classification rate)",
        classified_count, len(losers),
        (classified_count / len(losers) * 100) if losers else 0,
    )
    return losers


def _apply_extended_rules(loser: dict, current_mode: FailureMode) -> FailureMode:
    """Apply Phase-10 extended failure mode rules."""
    # LIQUIDITY_TRAP: slippage > 2%
    if loser.get("slippage_pct", 0.0) > 0.02:
        return FailureMode.LIQUIDITY_TRAP

    # ADVERSE_SELECTION: loss within 30 min of entry (holding_period_days == 0
    # is closest proxy we have — sub-day trades)
    if loser.get("holding_period_days", 1) == 0 and loser.get("realized_pnl_pct", 0.0) < -0.005:
        return FailureMode.ADVERSE_SELECTION

    return current_mode


def _try_llm_classify(unclassified: list[dict]) -> None:
    """Attempt LLM-based classification for unclassified losses. Best-effort."""
    try:
        from quantstack.llm.provider import get_chat_model

        model = get_chat_model("light")
        for loser in unclassified:
            prompt = (
                f"Classify this losing trade into exactly one failure mode.\n"
                f"Modes: regime_mismatch, factor_crowding, data_stale, timing_error, "
                f"thesis_wrong, black_swan, liquidity_trap, model_degradation, "
                f"signal_decay, adverse_selection, correlation_breakdown.\n\n"
                f"Trade: {loser['symbol']} strategy={loser['strategy_id']} "
                f"pnl={loser['realized_pnl_pct']:.4f} "
                f"entry_regime={loser['entry_regime']} exit_regime={loser['exit_regime']} "
                f"holding_days={loser['holding_period_days']} "
                f"slippage={loser['slippage_pct']:.4f}\n\n"
                f"Reply with ONLY the failure mode name, nothing else."
            )
            try:
                resp = model.invoke(prompt)
                content = resp.content.strip().lower().replace(" ", "_")
                # Validate against known modes
                try:
                    validated = FailureMode(content)
                    loser["failure_mode"] = validated.value
                except ValueError:
                    pass  # Keep UNCLASSIFIED
            except Exception:
                pass  # Keep UNCLASSIFIED — LLM is best-effort
    except (ImportError, Exception) as exc:
        logger.debug("[loss_analyzer] LLM classify unavailable: %s", exc)


def aggregate_failure_modes(
    conn: Any,
    classified_losses: list[dict],
) -> None:
    """Write classified losses to failure_mode_stats table.

    Rolling 30-day window: delete rows older than 30 days, then upsert
    today's aggregates grouped by failure_mode.
    """
    if not classified_losses:
        return

    today = date.today()
    window_start = today - timedelta(days=30)

    # Prune stale rows outside rolling window
    conn.execute(
        "DELETE FROM failure_mode_stats WHERE window_end::date < %s",
        [str(window_start)],
    )

    # Group by failure_mode
    groups: dict[str, dict] = {}
    for loss in classified_losses:
        fm = loss["failure_mode"]
        if fm not in groups:
            groups[fm] = {
                "frequency": 0,
                "total_pnl": 0.0,
                "strategies": set(),
            }
        groups[fm]["frequency"] += 1
        groups[fm]["total_pnl"] += loss["realized_pnl"]
        groups[fm]["strategies"].add(loss["strategy_id"])

    # Upsert each group
    for failure_mode, agg in groups.items():
        row_id = f"{failure_mode}_{today.isoformat()}"
        avg_loss = agg["total_pnl"] / agg["frequency"] if agg["frequency"] else 0.0
        strategies_json = json.dumps(sorted(agg["strategies"]))

        conn.execute(
            """
            INSERT INTO failure_mode_stats
                (id, failure_mode, window_start, window_end, frequency,
                 cumulative_pnl_impact, avg_loss_size, affected_strategies, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
            ON CONFLICT (id) DO UPDATE SET
                frequency = EXCLUDED.frequency,
                cumulative_pnl_impact = EXCLUDED.cumulative_pnl_impact,
                avg_loss_size = EXCLUDED.avg_loss_size,
                affected_strategies = EXCLUDED.affected_strategies,
                updated_at = NOW()
            """,
            [
                row_id,
                failure_mode,
                str(window_start),
                str(today),
                agg["frequency"],
                round(agg["total_pnl"], 2),
                round(avg_loss, 2),
                strategies_json,
            ],
        )

    logger.info(
        "[loss_analyzer] Aggregated %d failure modes into failure_mode_stats",
        len(groups),
    )


def prioritize_failure_modes(conn: Any) -> list[dict]:
    """Query failure_mode_stats for trailing 30 days, rank by cumulative P&L impact.

    Returns top 3 failure modes sorted by absolute cumulative_pnl_impact descending.
    """
    today = date.today()
    window_start = today - timedelta(days=30)

    rows = conn.execute(
        """
        SELECT failure_mode,
               SUM(frequency) AS total_frequency,
               SUM(cumulative_pnl_impact) AS total_pnl_impact,
               AVG(avg_loss_size) AS avg_loss
        FROM failure_mode_stats
        WHERE window_end::date >= %s
        GROUP BY failure_mode
        ORDER BY ABS(SUM(cumulative_pnl_impact)) DESC
        LIMIT 3
        """,
        [str(window_start)],
    ).fetchall()

    top_modes: list[dict] = []
    for row in rows:
        if isinstance(row, dict):
            top_modes.append({
                "failure_mode": row["failure_mode"],
                "total_frequency": row["total_frequency"],
                "total_pnl_impact": row["total_pnl_impact"],
                "avg_loss": row["avg_loss"],
            })
        else:
            top_modes.append({
                "failure_mode": row[0],
                "total_frequency": row[1],
                "total_pnl_impact": row[2],
                "avg_loss": row[3],
            })

    logger.info(
        "[loss_analyzer] Top %d failure modes by P&L impact: %s",
        len(top_modes),
        [m["failure_mode"] for m in top_modes],
    )
    return top_modes


def generate_research_tasks(
    conn: Any,
    top_modes: list[dict],
) -> list[str]:
    """Insert research tasks for the top failure modes into research_queue.

    Uses task_type='strategy_hypothesis' (closest match in the CHECK constraint).
    Returns list of generated task_ids.
    """
    if not top_modes:
        return []

    task_ids: list[str] = []
    for mode in top_modes:
        task_id = str(uuid.uuid4())
        priority = min(9, max(1, int(abs(mode.get("total_pnl_impact", 0)) / 50)))
        context = json.dumps({
            "failure_mode": mode["failure_mode"],
            "total_frequency": mode.get("total_frequency", 0),
            "total_pnl_impact": mode.get("total_pnl_impact", 0.0),
            "avg_loss": mode.get("avg_loss", 0.0),
            "source_pipeline": "daily_loss_analysis",
        })

        conn.execute(
            """
            INSERT INTO research_queue
                (task_id, task_type, priority, topic, context_json, source)
            VALUES (%s, 'strategy_hypothesis', %s, %s, %s::jsonb, 'loss_analyzer')
            """,
            [task_id, priority, f"loss_pattern:{mode['failure_mode']}", context],
        )
        task_ids.append(task_id)

    logger.info(
        "[loss_analyzer] Generated %d research tasks: %s",
        len(task_ids), task_ids,
    )
    return task_ids


def run_daily_loss_analysis(
    conn: Any | None = None,
    trade_date: date | None = None,
) -> dict:
    """Orchestrator — run all 5 stages of the daily loss analysis pipeline.

    If *conn* is None, opens a fresh pg_conn() for the entire pipeline.
    Returns a summary dict.
    """
    def _run(c: Any) -> dict:
        # Stage 1: collect
        losers = collect_daily_losers(c, trade_date)
        if not losers:
            logger.info("[loss_analyzer] No losers found — skipping analysis")
            return {
                "losers_found": 0,
                "classified": 0,
                "failure_modes_aggregated": 0,
                "top_modes": [],
                "research_tasks": [],
            }

        # Stage 2: classify
        classified = classify_losses(losers)

        # Stage 3: aggregate
        aggregate_failure_modes(c, classified)

        # Stage 4: prioritize
        top_modes = prioritize_failure_modes(c)

        # Stage 5: generate research tasks
        task_ids = generate_research_tasks(c, top_modes)

        summary = {
            "losers_found": len(losers),
            "classified": sum(
                1 for l in classified
                if l["failure_mode"] != FailureMode.UNCLASSIFIED.value
            ),
            "failure_modes_aggregated": len(
                {l["failure_mode"] for l in classified}
            ),
            "top_modes": top_modes,
            "research_tasks": task_ids,
        }
        logger.info("[loss_analyzer] Pipeline complete: %s", summary)
        return summary

    if conn is not None:
        return _run(conn)

    with pg_conn() as c:
        return _run(c)
