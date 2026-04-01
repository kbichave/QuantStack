# Copyright 2024 QuantStack Contributors
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
from quantstack.coordination.strategy_lock import StrategyStatusLock
from quantstack.coordination.supervisor import LoopSupervisor
from quantstack.db import open_db
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.tools._registry import domain


def _get_conn():
    """Get a PostgreSQL connection for operational tables.

    Callers must call conn.commit() after writes — open_db() returns a
    connection with autocommit=False, so INSERTs are invisible to other
    connections until committed.
    """
    return open_db()


def _get_event_bus():
    """Lazy-init EventBus singleton."""
    return EventBus(_get_conn())


def _get_strategy_lock():
    """Lazy-init StrategyStatusLock singleton."""
    return StrategyStatusLock(_get_conn(), _get_event_bus())


# ── Event Bus Tools ──────────────────────────────────────────────────────────


@domain(Domain.EXECUTION)
@tool_def()
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
        conn = _get_conn()
        bus = EventBus(conn)
        eid = bus.publish(event)
        conn.commit()
        return {"success": True, "event_id": eid}
    except Exception as exc:
        logger.error(f"[MCP:publish_event] {exc}")
        return {"success": False, "error": str(exc)}


@domain(Domain.EXECUTION)
@tool_def()
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

        conn = _get_conn()
        bus = EventBus(conn)
        events = bus.poll(consumer_id, event_types=types)
        conn.commit()  # commit cursor updates

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
@tool_def()
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
        bus = EventBus(conn)
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
        conn.commit()

        return {"success": True}
    except Exception as exc:
        logger.error(f"[MCP:record_heartbeat] {exc}")
        return {"success": False, "error": str(exc)}


# ── Loop Context Tools (stateless iteration state) ───────────────────────────


@domain(Domain.EXECUTION, Domain.PORTFOLIO)
@tool_def()
def get_loop_context(loop_name: str, key: str, default: Any = None) -> Any:
    """Read a per-loop context value from PostgreSQL.

    WHEN TO USE: At the START of every loop iteration to restore state that
    was written in the previous iteration. Replaces in-session state[] dict
    so that each `claude` invocation (no --continue) starts fresh but still
    has access to prior iteration context.

    WORKFLOW: [iteration start] → get_loop_context(key) → use value → [work]
              → set_loop_context(key, new_value) → [iteration end]

    Common keys for trading_loop:
        "market_intel"         — market-intel agent output (JSON, TTL 25min)
        "stale_symbols"        — list of symbols with stale OHLCV data
        "closes_since_review"  — int counter for weekly reflector trigger
        "last_weekly_review_at"— ISO timestamp of last weekly review

    Common keys for research_loop:
        "last_domain"          — last domain researched (swing/investment/options)
        "domain_history"       — list of recent domain choices for rotation
        "last_execution_audit_at" — ISO timestamp of last TCA audit
        "cross_domain_transfers"  — list of cross-domain signal findings

    Args:
        loop_name: "trading_loop" or "research_loop"
        key: Context key to retrieve.
        default: Value to return if key not found (default: None).

    Returns:
        The stored value (parsed from JSONB), or default if not found.
    """
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT context_json FROM loop_iteration_context "
            "WHERE loop_name = %s AND context_key = %s",
            [loop_name, key],
        ).fetchone()
        if row is None:
            return default
        value = row[0]
        # JSONB columns come back as dicts/lists; unwrap scalar wrapper if present
        if isinstance(value, dict) and list(value.keys()) == ["v"]:
            return value["v"]
        return value
    except Exception as exc:
        logger.warning(f"[MCP:get_loop_context] {loop_name}/{key}: {exc}")
        return default


@domain(Domain.EXECUTION, Domain.PORTFOLIO)
@tool_def()
def set_loop_context(loop_name: str, key: str, value: Any) -> dict[str, Any]:
    """Write a per-loop context value to PostgreSQL (UPSERT).

    WHEN TO USE: At the END of relevant loop steps to persist state that the
    next iteration will need. Any scalar, list, or dict can be stored.

    WORKFLOW: [compute new value] → set_loop_context(key, value)
              → [next iteration] → get_loop_context(key)

    Args:
        loop_name: "trading_loop" or "research_loop"
        key: Context key to store.
        value: Any JSON-serialisable value (str, int, float, list, dict).

    Returns:
        {"success": True, "loop_name": loop_name, "key": key}
    """
    try:
        conn = _get_conn()
        now = datetime.now(timezone.utc)
        # Wrap scalars so JSONB storage is consistent with retrieval
        if not isinstance(value, (dict, list)):
            stored = {"v": value}
        else:
            stored = value
        conn.execute(
            """
            INSERT INTO loop_iteration_context (loop_name, context_key, context_json, updated_at)
            VALUES (%s, %s, %s::jsonb, %s)
            ON CONFLICT (loop_name, context_key)
            DO UPDATE SET context_json = EXCLUDED.context_json,
                          updated_at   = EXCLUDED.updated_at
            """,
            [loop_name, key, json.dumps(stored), now],
        )
        conn.commit()
        return {"success": True, "loop_name": loop_name, "key": key}
    except Exception as exc:
        logger.error(f"[MCP:set_loop_context] {loop_name}/{key}: {exc}")
        return {"success": False, "error": str(exc)}


# ── Tool Error Tracking ──────────────────────────────────────────────────────

# Number of consecutive errors for the same tool before queuing a bug-fix task.
_BUG_FIX_THRESHOLD = 3

# Files the auto-patcher must never touch, regardless of what ARC suggests.
_PROTECTED_FILES = frozenset([
    "risk_gate.py",
    "kill_switch.py",
    "db.py",
])


@domain(Domain.EXECUTION, Domain.PORTFOLIO)
@tool_def()
def record_tool_error(
    tool_name: str,
    error_message: str,
    stack_trace: str = "",
    loop_name: str = "trading_loop",
    priority: int = 5,
) -> dict[str, Any]:
    """Record a tool call failure in the bugs table and auto-dispatch a fix after 3 hits.

    WHEN TO USE: Whenever a Python tool call raises an exception or returns an
    error dict. Call this instead of silently logging so the supervisor can detect
    systematic failures and dispatch AutoResearchClaw to fix them immediately.

    WORKFLOW: tool raises exception → record_tool_error() → bugs table upsert
              After 3 consecutive hits → research_queue entry inserted (priority=9)
              → supervisor bug-fix watcher fires ARC within 60s → ARC edits source
              → auto_patch validates + commits → loop restarts with fixed code

    Args:
        tool_name:     Function/tool that failed (e.g. "run_multi_signal_brief").
        error_message: Short description of the failure.
        stack_trace:   Full traceback (pass traceback.format_exc()).
        loop_name:     "trading_loop" or "research_loop".
        priority:      1-10. Higher = fixed sooner. Defaults to 5.
                       Use 8-9 for failures that block core paths (signal briefs,
                       execution), 3-4 for non-critical enrichment failures.

    Returns:
        {"success": True, "bug_id": str, "consecutive_errors": int, "bug_fix_queued": bool}
    """
    try:
        conn = _get_conn()
        now = datetime.now(timezone.utc)

        # Stable fingerprint: first 120 chars of the error message.
        # Groups repeated identical failures; different errors get separate rows.
        fingerprint = error_message[:120].strip()
        bug_id = f"bug_{tool_name}_{fingerprint[:40].replace(' ', '_')}"

        # Upsert into bugs table — increment counter on each new occurrence.
        conn.execute(
            """
            INSERT INTO bugs
                (bug_id, tool_name, loop_name, error_message, error_fingerprint,
                 stack_trace, status, priority, consecutive_errors, created_at, last_seen_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'open', %s, 1, %s, %s)
            ON CONFLICT (tool_name, loop_name, error_fingerprint)
            WHERE status IN ('open', 'in_progress')
            DO UPDATE SET
                consecutive_errors = bugs.consecutive_errors + 1,
                last_seen_at       = EXCLUDED.last_seen_at,
                stack_trace        = EXCLUDED.stack_trace,
                priority           = GREATEST(bugs.priority, EXCLUDED.priority)
            """,
            [bug_id, tool_name, loop_name, error_message, fingerprint,
             stack_trace[:4000], priority, now, now],
        )

        # Read back the current consecutive count for this bug.
        row = conn.execute(
            "SELECT consecutive_errors, bug_id FROM bugs "
            "WHERE tool_name = %s AND loop_name = %s AND error_fingerprint = %s "
            "AND status IN ('open', 'in_progress')",
            [tool_name, loop_name, fingerprint],
        ).fetchone()
        consecutive = row[0] if row else 1
        actual_bug_id = row[1] if row else bug_id

        bug_fix_queued = False
        if consecutive >= _BUG_FIX_THRESHOLD:
            # Only queue if no active dispatch already exists for this bug.
            existing_task = conn.execute(
                """
                SELECT 1 FROM research_queue
                WHERE task_type = 'bug_fix'
                  AND status IN ('pending', 'running')
                  AND context_json->>'bug_id' = %s
                """,
                [actual_bug_id],
            ).fetchone()

            if not existing_task:
                task_id = f"bugfix_{actual_bug_id}_{now.strftime('%Y%m%d_%H%M%S')}"
                conn.execute(
                    """
                    INSERT INTO research_queue
                        (task_id, task_type, priority, topic, context_json, source, status, created_at)
                    VALUES (%s, 'bug_fix', 9, %s, %s::jsonb, 'auto_error_tracker', 'pending', %s)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        task_id,
                        f"Auto bug-fix: {tool_name} — {error_message[:80]}",
                        json.dumps({
                            "bug_id": actual_bug_id,
                            "tool_name": tool_name,
                            "loop_name": loop_name,
                            "consecutive_errors": consecutive,
                            "last_error": error_message,
                            "stack_trace": stack_trace[:3000],
                            "triggered_at": now.isoformat(),
                        }),
                        now,
                    ],
                )
                # Link the dispatch task back to the bug row.
                conn.execute(
                    "UPDATE bugs SET arc_task_id = %s WHERE bug_id = %s",
                    [task_id, actual_bug_id],
                )
                bug_fix_queued = True
                logger.warning(
                    f"[MCP:record_tool_error] {tool_name} failed {consecutive}x "
                    f"(bug {actual_bug_id}) — queued {task_id}"
                )

        conn.commit()
        return {
            "success": True,
            "bug_id": actual_bug_id,
            "consecutive_errors": consecutive,
            "bug_fix_queued": bug_fix_queued,
        }

    except Exception as exc:
        logger.error(f"[MCP:record_tool_error] {exc}")
        return {"success": False, "error": str(exc)}


@domain(Domain.EXECUTION, Domain.PORTFOLIO)
@tool_def()
def clear_tool_errors(tool_name: str, loop_name: str = "trading_loop") -> dict[str, Any]:
    """Reset open bug entries for a tool after a successful call.

    WHEN TO USE: After a tool that was previously failing succeeds again.
    Marks all open/in_progress bugs for that tool as 'wont_fix' with a note,
    preventing duplicate dispatch tasks from being queued.
    """
    try:
        conn = _get_conn()
        now = datetime.now(timezone.utc)
        conn.execute(
            """
            UPDATE bugs SET status = 'wont_fix',
                            fix_summary = 'Tool started succeeding — auto-cleared',
                            fixed_at    = %s
            WHERE tool_name = %s AND loop_name = %s
              AND status IN ('open', 'in_progress')
            """,
            [now, tool_name, loop_name],
        )
        conn.commit()
        return {"success": True}
    except Exception as exc:
        logger.error(f"[MCP:clear_tool_errors] {exc}")
        return {"success": False, "error": str(exc)}


@domain(Domain.EXECUTION, Domain.PORTFOLIO)
@tool_def()
def get_open_bugs(limit: int = 20) -> dict[str, Any]:
    """Return open and in-progress bugs ordered by priority then age.

    WHEN TO USE: At iteration start to check for known broken tools before
    attempting to use them. If a tool has an open bug, consider skipping it
    rather than generating another error record.

    Returns:
        {"bugs": [...], "total_open": int}
        Each bug: {bug_id, tool_name, loop_name, error_message, status,
                   priority, consecutive_errors, created_at, last_seen_at}
    """
    try:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT bug_id, tool_name, loop_name, error_message, status,
                   priority, consecutive_errors,
                   created_at, last_seen_at, arc_task_id
            FROM bugs
            WHERE status IN ('open', 'in_progress')
            ORDER BY priority DESC, consecutive_errors DESC, created_at ASC
            LIMIT %s
            """,
            [limit],
        ).fetchall()

        count_row = conn.execute(
            "SELECT COUNT(*) FROM bugs WHERE status IN ('open', 'in_progress')"
        ).fetchone()

        return {
            "success": True,
            "total_open": count_row[0] if count_row else 0,
            "bugs": [
                {
                    "bug_id": r[0],
                    "tool_name": r[1],
                    "loop_name": r[2],
                    "error_message": r[3][:120],
                    "status": r[4],
                    "priority": r[5],
                    "consecutive_errors": r[6],
                    "created_at": r[7].isoformat() if r[7] else None,
                    "last_seen_at": r[8].isoformat() if r[8] else None,
                    "arc_task_id": r[9],
                }
                for r in rows
            ],
        }
    except Exception as exc:
        logger.error(f"[MCP:get_open_bugs] {exc}")
        return {"success": False, "bugs": [], "total_open": 0, "error": str(exc)}


# ── Health Tools ─────────────────────────────────────────────────────────────


@domain(Domain.EXECUTION, Domain.PORTFOLIO)
@tool_def()
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
@tool_def()
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
        conn = _get_conn()
        bus = EventBus(conn)
        lock = StrategyStatusLock(conn, bus)
        promoter = AutoPromoter(
            conn=conn,
            event_bus=bus,
            strategy_lock=lock,
        )
        decisions = promoter.evaluate_all()
        conn.commit()

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
@tool_def()
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
@tool_def()
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
@tool_def()
def log_agent_conversation(
    agent_name: str,
    content: str,
    summary: str = "",
    symbol: str | None = None,
    strategy_id: str | None = None,
    iteration: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log a desk agent's report for audit trail.

    WHEN TO USE: After every desk agent interaction (trade-debater, position-monitor,
    risk, fund-manager). Creates an audit trail.
    WORKFLOW: [spawn agent] → [get result] → log_agent_conversation → [continue]
    RELATED: log_signal_snapshot

    Args:
        agent_name: Agent identifier (market_intel, alpha_research, risk,
                    execution, strategy_rd, data_scientist, watchlist, pm).
        content: Full report text from the agent.
        summary: 1-line summary. Auto-generated if empty.
        symbol: Ticker symbol the report is about.
        strategy_id: Strategy this report relates to.
        iteration: Loop iteration number.
        metadata: Extra context (model used, tokens, duration_ms).

    Returns:
        {"success": True, "conversation_id": "..."}
    """
    try:
        conn = _get_conn()
        clogger = ConversationLogger(conn=conn)
        cid = clogger.log_agent_report(
            agent_name=agent_name,
            symbol=symbol,
            content=content,
            summary=summary,
            strategy_id=strategy_id,
            iteration=iteration,
            metadata=metadata,
        )
        conn.commit()
        return {"success": True, "conversation_id": cid}
    except Exception as exc:
        logger.error(f"[MCP:log_agent_conversation] {exc}")
        return {"success": False, "error": str(exc)}


@domain(Domain.EXECUTION)
@tool_def()
def log_signal_snapshot(
    symbol: str,
    collectors: dict[str, Any],
    bias: str = "neutral",
    conviction: float = 0.0,
    failures: list[str] | None = None,
) -> dict[str, Any]:
    """Log raw SignalEngine collector outputs for audit trail.

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
        conn = _get_conn()
        clogger = ConversationLogger(conn=conn)
        sid = clogger.log_signal_snapshot(
            symbol=symbol,
            collectors=collectors,
            bias=bias,
            conviction=conviction,
            failures=failures,
        )
        conn.commit()
        return {"success": True, "snapshot_id": sid}
    except Exception as exc:
        logger.error(f"[MCP:log_signal_snapshot] {exc}")
        return {"success": False, "error": str(exc)}


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
