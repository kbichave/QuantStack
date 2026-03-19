# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
DuckDB-based inter-loop event bus.

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

All writes go through the MCP server's DuckDB connection (single writer).
Read-only consumers can poll via open_db_readonly() but cannot acknowledge
(ack is a write operation).

Usage:
    from quant_pod.coordination.event_bus import EventBus, Event, EventType

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

import duckdb
from loguru import logger


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


@dataclass(frozen=True)
class Event:
    """Immutable event record."""

    event_type: EventType
    source_loop: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# TTL for event pruning
_EVENT_TTL_DAYS = 7


class EventBus:
    """
    Append-only DuckDB event log with per-consumer cursors.

    Thread-safe only via DuckDB's single-writer serialization — the caller
    (MCP server) is responsible for ensuring writes go through one connection.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def publish(self, event: Event) -> str:
        """
        Write an event to the bus.  Also prunes events older than 7 days.

        Returns the event_id.
        """
        payload_json = json.dumps(event.payload) if event.payload else "{}"
        self._conn.execute(
            """
            INSERT INTO loop_events (event_id, event_type, source_loop, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                event.event_id,
                event.event_type.value if isinstance(event.event_type, EventType) else event.event_type,
                event.source_loop,
                payload_json,
                event.created_at,
            ],
        )

        # Prune old events (cheap — index on created_at)
        cutoff = datetime.now(timezone.utc) - timedelta(days=_EVENT_TTL_DAYS)
        self._conn.execute(
            "DELETE FROM loop_events WHERE created_at < ?",
            [cutoff],
        )

        logger.debug(
            f"[EventBus] Published {event.event_type.value} from {event.source_loop} "
            f"(id={event.event_id})"
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
                    "SELECT event_id, event_type, source_loop, payload, created_at "
                    "FROM loop_events WHERE created_at >= ? AND event_id != ?"
                )
                params: list[Any] = [cursor_ts, last_event_id]
            else:
                # Cursor event was pruned — read all remaining events
                base_query = (
                    "SELECT event_id, event_type, source_loop, payload, created_at "
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
        for eid, etype, source, payload_raw, created in rows:
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
                )
            )

        # Update cursor to the latest event we returned
        now = datetime.now(timezone.utc)
        if events:
            new_cursor = events[-1].event_id
            # Upsert cursor: delete + insert (DuckDB ON CONFLICT has limitations)
            self._conn.execute(
                "DELETE FROM loop_cursors WHERE consumer_id = ?",
                [consumer_id],
            )
            self._conn.execute(
                "INSERT INTO loop_cursors (consumer_id, last_event_id, last_polled_at) "
                "VALUES (?, ?, ?)",
                [consumer_id, new_cursor, now],
            )
        else:
            # Update polled_at even if no events (so supervisor knows consumer is alive)
            self._conn.execute(
                "DELETE FROM loop_cursors WHERE consumer_id = ?",
                [consumer_id],
            )
            self._conn.execute(
                "INSERT INTO loop_cursors (consumer_id, last_event_id, last_polled_at) "
                "VALUES (?, NULL, ?)",
                [consumer_id, now],
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
