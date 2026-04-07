# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
PostgreSQL-based inter-loop event bus.

The three Ralph loops (Strategy Factory, Live Trader, ML Research) run as
independent Claude Opus sessions in tmux.  They need a lightweight coordination
mechanism so that, for example, the Trader loop picks up newly promoted
strategies without waiting for its next full iteration to discover them via
a DB scan.

Design choices:
  - **Poll-based, not push.**  Each loop polls at the start of its iteration
    for events since its last cursor position.  Latency = one iteration cycle
    (60s for factory, 300s for trader, 120s for ML).  Acceptable for
    swing-trading timeframes.
  - **Append-only event log.**  Events are immutable once written.
  - **Per-consumer cursors.**  Each loop tracks its own high-water mark in
    ``loop_cursors``.  No shared offset, no consumer groups.
  - **7-day TTL.**  Events older than 7 days are pruned on each publish() to
    prevent unbounded growth.

All writes go through the PostgreSQL connection pool.
Multiple consumers can poll concurrently via PostgreSQL's MVCC — no lock
contention between readers and writers.

Usage:
    from quantstack.coordination.event_bus import EventBus, Event, EventType

    bus = EventBus(conn)
    bus.publish(Event(event_type=EventType.STRATEGY_PROMOTED, source_loop="factory", payload={"strategy_id": "abc"}))

    events = bus.poll("trader_loop", event_types=[EventType.STRATEGY_PROMOTED])
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from loguru import logger

from quantstack.db import PgConnection


class EventType(str, Enum):
    """Known event types for inter-loop coordination."""

    STRATEGY_PROMOTED = "strategy_promoted"
    STRATEGY_RETIRED = "strategy_retired"
    STRATEGY_DEMOTED = "strategy_demoted"
    MODEL_TRAINED = "model_trained"
    DEGRADATION_DETECTED = "degradation_detected"
    SCREENER_COMPLETED = "screener_completed"
    UNIVERSE_REFRESHED = "universe_refreshed"
    LOOP_HEARTBEAT = "loop_heartbeat"
    LOOP_ERROR = "loop_error"
    # Agent activation event types (Section 01)
    MARKET_MOVE = "market_move"
    IDEAS_DISCOVERED = "ideas_discovered"
    # Risk monitoring event types (Section 06)
    RISK_WARNING = "risk_warning"
    RISK_SIZING_OVERRIDE = "risk_sizing_override"
    RISK_ENTRY_HALT = "risk_entry_halt"
    RISK_LIQUIDATION = "risk_liquidation"
    RISK_EMERGENCY = "risk_emergency"
    MODEL_DEGRADATION = "model_degradation"
    IC_DECAY = "ic_decay"
    REGIME_CHANGE = "regime_change"
    RISK_ALERT = "risk_alert"
    # Kill switch coordination (Section 07)
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    # Feedback loop event types (Phase 7)
    SIGNAL_DEGRADATION = "signal_degradation"
    SIGNAL_CONFLICT = "signal_conflict"
    AGENT_DEGRADATION = "agent_degradation"
    # Phase 10: Advanced Research coordination
    TOOL_ADDED = "tool_added"
    TOOL_DISABLED = "tool_disabled"
    EXPERIMENT_COMPLETED = "experiment_completed"
    FEATURE_DECAYED = "feature_decayed"
    FEATURE_REPLACED = "feature_replaced"
    MANDATE_ISSUED = "mandate_issued"
    META_OPTIMIZATION_APPLIED = "meta_optimization_applied"
    CONSENSUS_REQUIRED = "consensus_required"
    CONSENSUS_REACHED = "consensus_reached"


@dataclass(frozen=True)
class Event:
    """Immutable event record."""

    event_type: EventType
    source_loop: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    requires_ack: bool = False


# TTL for event pruning
_EVENT_TTL_DAYS = 7

# Events that require consumer acknowledgement
ACK_REQUIRED_EVENTS: set[EventType] = {
    EventType.RISK_WARNING,
    EventType.RISK_ENTRY_HALT,
    EventType.RISK_LIQUIDATION,
    EventType.RISK_EMERGENCY,
    EventType.IC_DECAY,
    EventType.REGIME_CHANGE,
    EventType.MODEL_DEGRADATION,
}

# Time allowed for consumers to ACK before escalation (2x supervisor cycle)
ACK_TIMEOUT_SECONDS: int = 600


class EventBus:
    """
    Append-only PostgreSQL event log with per-consumer cursors.

    PostgreSQL MVCC ensures concurrent readers and writers do not block each
    other. The caller injects the shared connection pool connection.
    """

    def __init__(self, conn: PgConnection) -> None:
        self._conn = conn

    def publish(self, event: Event) -> str:
        """
        Write an event to the bus.  Also prunes events older than 7 days.

        Returns the event_id.
        """
        payload_json = json.dumps(event.payload) if event.payload else "{}"
        etype_val = (
            event.event_type.value
            if isinstance(event.event_type, EventType)
            else event.event_type
        )
        needs_ack = (
            event.event_type in ACK_REQUIRED_EVENTS
            if isinstance(event.event_type, EventType)
            else False
        )
        ack_deadline = (
            event.created_at + timedelta(seconds=ACK_TIMEOUT_SECONDS)
            if needs_ack else None
        )
        self._conn.execute(
            """
            INSERT INTO loop_events
                (event_id, event_type, source_loop, payload, created_at,
                 requires_ack, expected_ack_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event.event_id,
                etype_val,
                event.source_loop,
                payload_json,
                event.created_at,
                needs_ack,
                ack_deadline,
            ],
        )

        # Prune old events (cheap — index on created_at)
        cutoff = datetime.now(timezone.utc) - timedelta(days=_EVENT_TTL_DAYS)
        self._conn.execute(
            "DELETE FROM loop_events WHERE created_at < ?",
            [cutoff],
        )

        logger.debug(
            f"[EventBus] Published {etype_val} from {event.source_loop} "
            f"(id={event.event_id}, ack={needs_ack})"
        )
        return event.event_id

    def poll(
        self,
        consumer_id: str,
        event_types: list[EventType] | None = None,
    ) -> list[Event]:
        """
        Read events since this consumer's last cursor position.

        Updates the cursor after reading.  Returns events in chronological order.

        Args:
            consumer_id: Unique consumer name (e.g. "factory_loop", "trader_loop").
            event_types: Optional filter — only return these event types.
        """
        # Get cursor
        row = self._conn.execute(
            "SELECT last_event_id FROM loop_cursors WHERE consumer_id = ?",
            [consumer_id],
        ).fetchone()

        last_event_id = row[0] if row else None

        # Build query
        if last_event_id:
            # Get the created_at of the cursor event for efficient filtering
            cursor_row = self._conn.execute(
                "SELECT created_at FROM loop_events WHERE event_id = ?",
                [last_event_id],
            ).fetchone()

            if cursor_row:
                cursor_ts = cursor_row[0]
                base_query = (
                    "SELECT event_id, event_type, source_loop, payload, created_at, requires_ack "
                    "FROM loop_events WHERE created_at >= ? AND event_id != ?"
                )
                params: list[Any] = [cursor_ts, last_event_id]
            else:
                # Cursor event was pruned — read all remaining events
                base_query = (
                    "SELECT event_id, event_type, source_loop, payload, created_at, requires_ack "
                    "FROM loop_events WHERE 1=1"
                )
                params = []
        else:
            # First poll — read all events
            base_query = (
                "SELECT event_id, event_type, source_loop, payload, created_at "
                "FROM loop_events WHERE 1=1"
            )
            params = []

        if event_types:
            placeholders = ", ".join("?" for _ in event_types)
            base_query += f" AND event_type IN ({placeholders})"
            params.extend(
                et.value if isinstance(et, EventType) else et for et in event_types
            )

        base_query += " ORDER BY created_at ASC"

        rows = self._conn.execute(base_query, params).fetchall()

        events: list[Event] = []
        for eid, etype, source, payload_raw, created, *extra in rows:
            requires_ack_val = extra[0] if extra else False
            payload = json.loads(payload_raw) if payload_raw else {}
            try:
                event_type = EventType(etype)
            except ValueError:
                event_type = etype  # type: ignore[assignment]
            events.append(
                Event(
                    event_id=eid,
                    event_type=event_type,
                    source_loop=source,
                    payload=payload,
                    created_at=created,
                    requires_ack=bool(requires_ack_val),
                )
            )

        # Update cursor atomically via upsert (no crash window between DELETE+INSERT)
        now = datetime.now(timezone.utc)
        new_cursor = events[-1].event_id if events else None
        self._conn.execute(
            "INSERT INTO loop_cursors (consumer_id, last_event_id, last_polled_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (consumer_id) DO UPDATE SET "
            "last_event_id = EXCLUDED.last_event_id, "
            "last_polled_at = EXCLUDED.last_polled_at",
            [consumer_id, new_cursor, now],
        )

        return events

    def get_latest(self, event_type: EventType) -> Event | None:
        """Get the most recent event of a given type.  Read-only."""
        row = self._conn.execute(
            """
            SELECT event_id, event_type, source_loop, payload, created_at
            FROM loop_events
            WHERE event_type = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [event_type.value],
        ).fetchone()

        if not row:
            return None

        eid, etype, source, payload_raw, created = row
        payload = json.loads(payload_raw) if payload_raw else {}
        return Event(
            event_id=eid,
            event_type=EventType(etype),
            source_loop=source,
            payload=payload,
            created_at=created,
        )

    def count_events(
        self,
        event_type: EventType | None = None,
        since: datetime | None = None,
    ) -> int:
        """Count events, optionally filtered by type and/or time window."""
        query = "SELECT COUNT(*) FROM loop_events WHERE 1=1"
        params: list[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        if since:
            query += " AND created_at >= ?"
            params.append(since)

        row = self._conn.execute(query, params).fetchone()
        return row[0] if row else 0

    def ack(self, event_id: str, consumer_id: str) -> None:
        """Acknowledge receipt and processing of an event.

        Sets acked_at and acked_by on the event row.
        Idempotent: if already acked, this is a no-op (acked_at not overwritten).
        """
        now = datetime.now(timezone.utc)
        self._conn.execute(
            "UPDATE loop_events SET acked_at = ?, acked_by = ? "
            "WHERE event_id = ? AND acked_at IS NULL",
            [now, consumer_id, event_id],
        )
        logger.debug("[EventBus] ACKed %s by %s", event_id, consumer_id)


async def check_missed_acks(conn: PgConnection) -> list[dict]:
    """Detect events that missed their ACK deadline and escalate.

    Escalation tiers based on how long overdue:
    - <=600s past deadline: re-publish (retry)
    - 600-1500s: WARNING system alert
    - >1500s: dead letter + CRITICAL alert

    Returns list of system alerts created.
    """
    now = datetime.now(timezone.utc)
    # Grace buffer: ignore events whose deadline just passed (within 60s)
    grace_cutoff = now - timedelta(seconds=60)

    rows = conn.execute(
        """
        SELECT event_id, event_type, source_loop, payload, created_at,
               expected_ack_by
        FROM loop_events
        WHERE requires_ack = TRUE
          AND acked_at IS NULL
          AND expected_ack_by < ?
        ORDER BY expected_ack_by ASC
        """,
        [grace_cutoff],
    ).fetchall()

    if not rows:
        return []

    alerts: list[dict] = []
    bus = EventBus(conn)

    for row in rows:
        eid = row["event_id"]
        etype = row["event_type"]
        source = row["source_loop"]
        payload_raw = row["payload"]
        created = row["created_at"]
        expected_by = row["expected_ack_by"]

        overdue_seconds = (now - expected_by).total_seconds()
        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})

        if overdue_seconds <= 600:
            # Tier 1: retry — re-publish the event
            try:
                new_event = Event(
                    event_type=EventType(etype),
                    source_loop=source,
                    payload={**payload, "_retry_of": eid},
                )
                bus.publish(new_event)
                logger.info("[ACK] Re-published %s (retry of %s)", new_event.event_id, eid)
                # Mark original so we don't re-process it
                conn.execute(
                    "UPDATE loop_events SET acked_by = 'retried' WHERE event_id = ?",
                    [eid],
                )
            except Exception as e:
                logger.warning("[ACK] Retry failed for %s: %s", eid, e)

        elif overdue_seconds <= 1500:
            # Tier 2: WARNING alert
            try:
                from quantstack.tools.functions.system_alerts import emit_system_alert

                alert_id = await emit_system_alert(
                    category="ack_timeout",
                    severity="warning",
                    title=f"Missed ACK: {etype} (event {eid})",
                    detail=(
                        f"Event {eid} ({etype}) from {source} has been unacknowledged "
                        f"for {int(overdue_seconds)}s (deadline was {expected_by})."
                    ),
                    source="ack_monitor",
                    metadata={"event_id": eid, "event_type": etype, "overdue_seconds": overdue_seconds},
                )
                alerts.append({"alert_id": alert_id, "severity": "warning", "event_id": eid})
            except Exception as e:
                logger.warning("[ACK] Warning alert failed for %s: %s", eid, e)

        else:
            # Tier 3: dead letter + CRITICAL alert
            try:
                # Count retries for this event
                retry_row = conn.execute(
                    "SELECT COUNT(*) FROM loop_events "
                    "WHERE payload::text LIKE ? AND event_id != ?",
                    [f'%"_retry_of": "{eid}"%', eid],
                ).fetchone()
                retry_count = retry_row[0] if retry_row else 0

                conn.execute(
                    """
                    INSERT INTO dead_letter_events
                        (original_event_id, event_type, source_loop, payload,
                         published_at, expected_ack_by, retry_count, dead_lettered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [eid, etype, source, payload_raw, created, expected_by, retry_count, now],
                )

                # Mark original as dead-lettered
                conn.execute(
                    "UPDATE loop_events SET acked_by = 'dead_lettered' WHERE event_id = ?",
                    [eid],
                )

                from quantstack.tools.functions.system_alerts import emit_system_alert

                alert_id = await emit_system_alert(
                    category="ack_timeout",
                    severity="critical",
                    title=f"Dead-lettered: {etype} (event {eid})",
                    detail=(
                        f"Event {eid} ({etype}) from {source} dead-lettered after "
                        f"{int(overdue_seconds)}s unacknowledged. Retry count: {retry_count}."
                    ),
                    source="ack_monitor",
                    metadata={
                        "event_id": eid, "event_type": etype,
                        "overdue_seconds": overdue_seconds, "retry_count": retry_count,
                    },
                )
                alerts.append({"alert_id": alert_id, "severity": "critical", "event_id": eid})
                logger.warning("[ACK] Dead-lettered event %s (%s)", eid, etype)
            except Exception as e:
                logger.error("[ACK] Dead-letter failed for %s: %s", eid, e)

    return alerts
