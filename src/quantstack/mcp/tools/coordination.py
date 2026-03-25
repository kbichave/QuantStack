# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
MCP tool wrappers for the coordination layer.

These tools expose the event bus, heartbeats, auto-promotion, and daily
digest to Claude Code sessions and Ralph loop prompts.

All writes go through the MCP server's PostgreSQL connection.
The coordination modules are instantiated lazily on first use.
"""

import json
from datetime import date, datetime, timezone
from typing import Any

from loguru import logger

from quantstack.coordination.auto_promoter import AutoPromoter
from quantstack.coordination.conversation_logger import ConversationLogger
from quantstack.coordination.daily_digest import DailyDigest
from quantstack.coordination.event_bus import Event, EventBus, EventType
from quantstack.coordination.preflight import PreflightCheck
from quantstack.coordination.slack_client import SlackClient
from quantstack.coordination.strategy_lock import StrategyStatusLock
from quantstack.coordination.supervisor import LoopSupervisor
from quantstack.db import open_db
from quantstack.mcp.domains import Domain
from quantstack.mcp.server import mcp
from quantstack.mcp.tools._registry import domain


def _get_conn():
    """Get a PostgreSQL connection for operational tables."""
    return open_db()


def _get_event_bus():
    """Lazy-init EventBus singleton."""
    return EventBus(_get_conn())


def _get_strategy_lock():
    """Lazy-init StrategyStatusLock singleton."""
    return StrategyStatusLock(_get_conn(), _get_event_bus())


# ── Event Bus Tools ──────────────────────────────────────────────────────────


@domain(Domain.EXECUTION)
@mcp.tool()
def publish_event(
    event_type: str,
    source: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish an event to the inter-loop event bus.

    WHEN TO USE: After significant state changes — strategy promoted, model trained,
    loop error, trade executed. Enables cross-loop communication.
    WHEN NOT TO USE: For routine data queries — use direct DB reads instead.
    WORKFLOW: [state change occurs] → publish_event → poll_events (other loop)
    RELATED: poll_events, record_heartbeat

    Args:
        event_type: Event type (e.g., "strategy_promoted", "model_trained",
                    "loop_heartbeat", "loop_error").
        source: Source loop name (e.g., "strategy_factory", "live_trader").
        payload: Optional JSON-serializable payload.

    Returns:
        {"success": True, "event_id": "..."}
    """
    try:
        try:
            etype = EventType(event_type)
        except ValueError:
            etype = event_type  # type: ignore[assignment]

        event = Event(
            event_type=etype,
            source_loop=source,
            payload=payload or {},
        )
        bus = _get_event_bus()
        eid = bus.publish(event)
        return {"success": True, "event_id": eid}
    except Exception as exc:
        logger.error(f"[MCP:publish_event] {exc}")
        return {"success": False, "error": str(exc)}


@domain(Domain.EXECUTION)
@mcp.tool()
def poll_events(
    consumer_id: str,
    event_types: list[str] | None = None,
    since_minutes: int = 60,
) -> dict[str, Any]:
    """Poll the event bus for new events since the consumer's last cursor.

    WHEN TO USE: At loop iteration start — check what happened since last iteration.
    Research loop polls for model_trained, strategy_promoted events. Trading loop
    polls for strategy changes, risk alerts.
    WORKFLOW: [loop start] → poll_events → [process events] → [continue iteration]
    RELATED: publish_event, get_loop_health

    Args:
        consumer_id: Unique consumer name (e.g., "factory_loop", "trader_loop").
        event_types: Optional filter — only return these event types.
        since_minutes: Ignored (cursor-based), kept for API compatibility.

    Returns:
        {"success": True, "events": [...], "count": N}
    """
    try:
        types = None
        if event_types:
            types = []
            for et in event_types:
                try:
                    types.append(EventType(et))
                except ValueError:
                    types.append(et)  # type: ignore[arg-type]

        bus = _get_event_bus()
        events = bus.poll(consumer_id, event_types=types)

        return {
            "success": True,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": (
                        e.event_type.value
                        if hasattr(e.event_type, "value")
                        else e.event_type
                    ),
                    "source_loop": e.source_loop,
                    "payload": e.payload,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ],
            "count": len(events),
        }
    except Exception as exc:
        logger.error(f"[MCP:poll_events] {exc}")
        return {"success": False, "error": str(exc), "events": [], "count": 0}


# ── Heartbeat Tools ──────────────────────────────────────────────────────────


@domain(Domain.EXECUTION)
@mcp.tool()
def record_heartbeat(
    loop_name: str,
    iteration: int,
    symbols_processed: int = 0,
    errors: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    """Record a loop heartbeat for health monitoring.

    WHEN TO USE: At the start (status="running") and end (status="completed")
    of EVERY loop iteration. This is how the supervisor detects stale/crashed loops.
    WORKFLOW: [iteration start] → record_heartbeat(running) → [work] → record_heartbeat(completed)
    RELATED: get_loop_health, publish_event

    Args:
        loop_name: Loop identifier ("research_loop", "trading_loop").
        iteration: Monotonically increasing iteration counter.
        symbols_processed: Number of symbols processed in this iteration.
        errors: Number of errors encountered.
        status: "running" or "completed" or "failed".

    Returns:
        {"success": True}
    """
    try:
        conn = _get_conn()
        now = datetime.now(timezone.utc)

        if status == "running":
            conn.execute(
                """
                INSERT INTO loop_heartbeats (loop_name, iteration, started_at, status)
                VALUES (?, ?, ?, 'running')
                ON CONFLICT (loop_name, iteration)
                DO UPDATE SET started_at = EXCLUDED.started_at, status = 'running'
                """,
                [loop_name, iteration, now],
            )
        else:
            conn.execute(
                """
                INSERT INTO loop_heartbeats
                    (loop_name, iteration, started_at, finished_at,
                     symbols_processed, errors, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (loop_name, iteration)
                DO UPDATE SET finished_at = EXCLUDED.finished_at,
                              symbols_processed = EXCLUDED.symbols_processed,
                              errors = EXCLUDED.errors,
                              status = EXCLUDED.status
                """,
                [loop_name, iteration, now, now, symbols_processed, errors, status],
            )

        # Also publish heartbeat event for the supervisor
        bus = _get_event_bus()
        bus.publish(
            Event(
                event_type=EventType.LOOP_HEARTBEAT,
                source_loop=loop_name,
                payload={
                    "iteration": iteration,
                    "symbols_processed": symbols_processed,
                    "errors": errors,
                    "status": status,
                },
            )
        )

        return {"success": True}
    except Exception as exc:
        logger.error(f"[MCP:record_heartbeat] {exc}")
        return {"success": False, "error": str(exc)}


# ── Health Tools ─────────────────────────────────────────────────────────────


@domain(Domain.EXECUTION, Domain.PORTFOLIO)
@mcp.tool()
def get_loop_health() -> dict[str, Any]:
    """Get health status of all monitored loops (research, trading, ML).

    WHEN TO USE: At iteration start to verify all loops are running. If a loop
    is stale (>10 min since last heartbeat), investigate before relying on its output.
    WORKFLOW: get_system_status → get_loop_health → [if unhealthy] → investigate
    RELATED: record_heartbeat, get_system_status

    Returns:
        {"success": True, "loops": [...], "all_healthy": bool}
    """
    try:
        supervisor = LoopSupervisor(_get_conn())
        results = supervisor.check_health()

        loops = [
            {
                "loop_name": h.loop_name,
                "status": h.status,
                "last_heartbeat": (
                    h.last_heartbeat.isoformat() if h.last_heartbeat else None
                ),
                "staleness_seconds": round(h.staleness_seconds, 1),
                "last_iteration": h.last_iteration,
                "consecutive_errors": h.consecutive_errors,
            }
            for h in results
        ]

        all_healthy = all(h.status in ("healthy", "unknown") for h in results)

        return {"success": True, "loops": loops, "all_healthy": all_healthy}
    except Exception as exc:
        logger.error(f"[MCP:get_loop_health] {exc}")
        return {"success": False, "error": str(exc)}


# ── Auto-Promotion Tools ────────────────────────────────────────────────────


@domain(Domain.EXECUTION, Domain.RESEARCH)
@mcp.tool()
def auto_promote_eligible() -> dict[str, Any]:
    """Evaluate all forward_testing strategies for automatic promotion to live.

    WHEN TO USE: End of research cycle or daily check. Strategies must have 30+ days
    of forward testing data and meet performance thresholds before promotion.
    WHEN NOT TO USE: Requires AUTO_PROMOTE_ENABLED=true env var. Disabled by default.
    WORKFLOW: run_walkforward → [30+ days forward testing] → auto_promote_eligible → [if promoted] → publish_event
    RELATED: publish_event, get_loop_health

    Returns:
        {"success": True, "decisions": [...], "promoted_count": N}
    """
    try:
        promoter = AutoPromoter(
            conn=_get_conn(),
            event_bus=_get_event_bus(),
            strategy_lock=_get_strategy_lock(),
        )
        decisions = promoter.evaluate_all()

        return {
            "success": True,
            "decisions": [
                {
                    "strategy_id": d.strategy_id,
                    "name": d.name,
                    "decision": d.decision,
                    "reason": d.reason,
                    "evidence": d.evidence,
                }
                for d in decisions
            ],
            "promoted_count": sum(1 for d in decisions if d.decision == "promote"),
            "enabled": promoter.is_enabled(),
        }
    except Exception as exc:
        logger.error(f"[MCP:auto_promote_eligible] {exc}")
        return {"success": False, "error": str(exc)}


# ── Daily Digest Tools ───────────────────────────────────────────────────────


@domain(Domain.EXECUTION, Domain.PORTFOLIO)
@mcp.tool()
def generate_daily_digest(target_date: str | None = None) -> dict[str, Any]:
    """Generate a daily digest report summarizing positions, trades, loops, and P&L.

    WHEN TO USE: End of trading day or morning review. Produces markdown summary
    of portfolio state, trade activity, loop health, and sends to Discord.
    WORKFLOW: [end of day] → generate_daily_digest → [review next morning]
    RELATED: get_portfolio_state, get_loop_health

    Args:
        target_date: ISO date string (YYYY-MM-DD). Defaults to today.

    Returns:
        {"success": True, "report": {...}, "markdown": "..."}
    """
    try:
        digest = DailyDigest(_get_conn())
        td = date.fromisoformat(target_date) if target_date else None
        report = digest.generate(td)
        md = digest.format_markdown(report)

        # Try to send to Discord
        digest.send_discord(report)

        return {
            "success": True,
            "report": {
                "report_date": report.report_date.isoformat(),
                "open_positions": report.open_positions,
                "trades_today": report.trades_today,
                "total_realized_pnl": report.total_realized_pnl,
                "total_live": report.total_live,
                "total_forward_testing": report.total_forward_testing,
                "factory_iterations": report.factory_iterations,
                "trader_iterations": report.trader_iterations,
                "ml_iterations": report.ml_iterations,
                "total_loop_errors": report.total_loop_errors,
                "universe_size": report.universe_size,
                "watchlist_size": report.watchlist_size,
            },
            "markdown": md,
        }
    except Exception as exc:
        logger.error(f"[MCP:generate_daily_digest] {exc}")
        return {"success": False, "error": str(exc)}


# ── Preflight Check Tools ────────────────────────────────────────────────────


@domain(Domain.EXECUTION)
@mcp.tool()
def run_preflight_check(
    target_symbols: list[str] | None = None,
    target_wallet: float = 1000.0,
) -> dict[str, Any]:
    """Run the production preflight check — the gate between research and trading.

    WHEN TO USE: Before starting the trading loop for the first time, or after any
    infrastructure change. Validates DB tables, kill switch, cash balance, universe,
    strategies, risk limits, data provider, broker, and paper mode.
    WHEN NOT TO USE: Not needed every iteration — only on startup or after changes.
    WORKFLOW: [deploy new strategy] → run_preflight_check → [if ready] → start trading loop
    RELATED: get_system_status, get_loop_health

    Args:
        target_symbols: Symbols to validate (default ["SPY"]).
        target_wallet: Starting equity in dollars (default 1000).

    Returns:
        {"ready": bool, "blockers": [...], "warnings": [...], "summary": "..."}
    """
    try:
        conn = _get_conn()
        check = PreflightCheck(conn, target_symbols, target_wallet)
        report = check.run()

        return {
            "success": True,
            "ready": report.ready,
            "blockers": [{"name": c.name, "detail": c.detail} for c in report.blockers],
            "warnings": [{"name": c.name, "detail": c.detail} for c in report.warnings],
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "severity": c.severity,
                }
                for c in report.checks
            ],
            "summary": report.summary(),
        }
    except Exception as exc:
        logger.error(f"[MCP:run_preflight_check] {exc}")
        return {"success": False, "error": str(exc)}


# ── Conversation Logging Tools ───────────────────────────────────────────────


def _get_conversation_logger():
    """Lazy-init ConversationLogger."""
    return ConversationLogger(conn=_get_conn())


@domain(Domain.EXECUTION)
@mcp.tool()
def log_agent_conversation(
    agent_name: str,
    content: str,
    summary: str = "",
    symbol: str | None = None,
    strategy_id: str | None = None,
    iteration: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log a desk agent's report for audit trail and Slack notification.

    WHEN TO USE: After every desk agent interaction (trade-debater, position-monitor,
    risk, fund-manager). Creates an audit trail and posts summary to Slack.
    WORKFLOW: [spawn agent] → [get result] → log_agent_conversation → [continue]
    RELATED: log_signal_snapshot, post_slack_message

    Args:
        agent_name: Agent identifier (market_intel, alpha_research, risk,
                    execution, strategy_rd, data_scientist, watchlist, pm).
        content: Full report text from the agent.
        summary: 1-line summary for Slack. Auto-generated if empty.
        symbol: Ticker symbol the report is about.
        strategy_id: Strategy this report relates to.
        iteration: Loop iteration number.
        metadata: Extra context (model used, tokens, duration_ms).

    Returns:
        {"success": True, "conversation_id": "..."}
    """
    try:
        clogger = _get_conversation_logger()
        cid = clogger.log_agent_report(
            agent_name=agent_name,
            symbol=symbol,
            content=content,
            summary=summary,
            strategy_id=strategy_id,
            iteration=iteration,
            metadata=metadata,
        )
        return {"success": True, "conversation_id": cid}
    except Exception as exc:
        logger.error(f"[MCP:log_agent_conversation] {exc}")
        return {"success": False, "error": str(exc)}


@domain(Domain.EXECUTION)
@mcp.tool()
def log_signal_snapshot(
    symbol: str,
    collectors: dict[str, Any],
    bias: str = "neutral",
    conviction: float = 0.0,
    failures: list[str] | None = None,
) -> dict[str, Any]:
    """Log raw SignalEngine collector outputs for audit trail and Slack.

    WHEN TO USE: After every get_signal_brief() call in the trading loop.
    Creates an audit trail of what the signal engine saw at decision time.
    WORKFLOW: get_signal_brief → log_signal_snapshot → [trading decision]
    RELATED: get_signal_brief, log_agent_conversation

    Args:
        symbol: Ticker symbol.
        collectors: Dict of {collector_name: raw_output_dict} from SignalEngine.
        bias: Consensus bias (bullish/bearish/neutral).
        conviction: Consensus conviction (0-1).
        failures: List of collector names that failed.

    Returns:
        {"success": True, "snapshot_id": "..."}
    """
    try:
        clogger = _get_conversation_logger()
        sid = clogger.log_signal_snapshot(
            symbol=symbol,
            collectors=collectors,
            bias=bias,
            conviction=conviction,
            failures=failures,
        )
        return {"success": True, "snapshot_id": sid}
    except Exception as exc:
        logger.error(f"[MCP:log_signal_snapshot] {exc}")
        return {"success": False, "error": str(exc)}


@domain(Domain.EXECUTION)
@mcp.tool()
def post_slack_message(
    channel: str,
    text: str,
) -> dict[str, Any]:
    """Post a message to a Slack channel for notifications.

    WHEN TO USE: For manual notifications — trade alerts, system warnings,
    daily summaries. Most tools post to Slack automatically; use this for
    custom messages only.
    WHEN NOT TO USE: Don't spam — one message per significant event.
    RELATED: log_agent_conversation, generate_daily_digest

    Args:
        channel: Channel key ("agents", "trades", "alerts", "system", etc.)
                 or a channel name ("#my-channel").
        text: Message text (supports Slack mrkdwn formatting).

    Returns:
        {"success": True, "ts": "..."} or {"success": False, "error": "..."}
    """
    try:
        client = SlackClient()
        ts = client.post(channel, text)
        return {"success": ts is not None, "ts": ts}
    except Exception as exc:
        logger.error(f"[MCP:post_slack_message] {exc}")
        return {"success": False, "error": str(exc)}
