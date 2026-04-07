"""Unit tests for EventBus ACK pattern."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from quantstack.coordination.event_bus import (
    ACK_REQUIRED_EVENTS,
    ACK_TIMEOUT_SECONDS,
    Event,
    EventBus,
    EventType,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class MockCursor:
    """Minimal mock cursor that tracks execute calls."""

    def __init__(self):
        self.calls: list[tuple[str, list]] = []
        self._fetchone_val = None
        self._fetchall_val = []
        self.rowcount = 1

    def fetchone(self):
        return self._fetchone_val

    def fetchall(self):
        return self._fetchall_val


class MockConn:
    """Mock PgConnection that records all execute calls."""

    def __init__(self):
        self.calls: list[tuple[str, list]] = []
        self._cursor = MockCursor()

    def execute(self, sql, params=None):
        self.calls.append((sql, params or []))
        return self._cursor


# ---------------------------------------------------------------------------
# ACK constants
# ---------------------------------------------------------------------------


class TestACKConstants:
    def test_risk_events_require_ack(self):
        assert EventType.RISK_WARNING in ACK_REQUIRED_EVENTS
        assert EventType.RISK_LIQUIDATION in ACK_REQUIRED_EVENTS
        assert EventType.RISK_EMERGENCY in ACK_REQUIRED_EVENTS
        assert EventType.IC_DECAY in ACK_REQUIRED_EVENTS
        assert EventType.REGIME_CHANGE in ACK_REQUIRED_EVENTS

    def test_non_risk_events_do_not_require_ack(self):
        assert EventType.STRATEGY_PROMOTED not in ACK_REQUIRED_EVENTS
        assert EventType.LOOP_HEARTBEAT not in ACK_REQUIRED_EVENTS

    def test_timeout_is_600s(self):
        assert ACK_TIMEOUT_SECONDS == 600


# ---------------------------------------------------------------------------
# publish() ACK behavior
# ---------------------------------------------------------------------------


class TestPublishACK:
    def test_sets_requires_ack_true_for_risk_event(self):
        conn = MockConn()
        bus = EventBus(conn)
        event = Event(event_type=EventType.RISK_WARNING, source_loop="test")
        bus.publish(event)

        # First call is the INSERT
        sql, params = conn.calls[0]
        assert "requires_ack" in sql
        assert "expected_ack_by" in sql
        # params: [event_id, etype, source, payload, created_at, requires_ack, expected_ack_by]
        assert params[5] is True  # requires_ack
        assert params[6] is not None  # expected_ack_by set

    def test_sets_requires_ack_false_for_non_risk_event(self):
        conn = MockConn()
        bus = EventBus(conn)
        event = Event(event_type=EventType.STRATEGY_PROMOTED, source_loop="test")
        bus.publish(event)

        sql, params = conn.calls[0]
        assert params[5] is False
        assert params[6] is None

    def test_expected_ack_by_is_600s_from_event_time(self):
        conn = MockConn()
        bus = EventBus(conn)
        now = datetime.now(timezone.utc)
        event = Event(event_type=EventType.RISK_ENTRY_HALT, source_loop="test", created_at=now)
        bus.publish(event)

        sql, params = conn.calls[0]
        deadline = params[6]
        expected = now + timedelta(seconds=600)
        assert abs((deadline - expected).total_seconds()) < 1


# ---------------------------------------------------------------------------
# ack() method
# ---------------------------------------------------------------------------


class TestACKMethod:
    def test_sets_acked_fields(self):
        conn = MockConn()
        bus = EventBus(conn)
        bus.ack("evt123", "trading_graph")

        sql, params = conn.calls[0]
        assert "acked_at" in sql
        assert "acked_by" in sql
        assert "acked_at IS NULL" in sql  # idempotency guard
        assert params[1] == "trading_graph"
        assert params[2] == "evt123"

    def test_idempotent_no_error(self):
        """Calling ack twice doesn't raise."""
        conn = MockConn()
        bus = EventBus(conn)
        bus.ack("evt123", "trading_graph")
        bus.ack("evt123", "trading_graph")  # no exception
        assert len(conn.calls) == 2


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


class TestEventDataclass:
    def test_requires_ack_defaults_false(self):
        event = Event(event_type=EventType.STRATEGY_PROMOTED, source_loop="test")
        assert event.requires_ack is False

    def test_requires_ack_can_be_set(self):
        event = Event(
            event_type=EventType.RISK_WARNING,
            source_loop="test",
            requires_ack=True,
        )
        assert event.requires_ack is True


# ---------------------------------------------------------------------------
# check_missed_acks (basic structure tests with mocks)
# ---------------------------------------------------------------------------


class TestCheckMissedAcks:
    def test_returns_empty_when_all_acked(self):
        """No unacknowledged events -> empty result."""
        from quantstack.coordination.event_bus import check_missed_acks

        conn = MockConn()
        conn._cursor._fetchall_val = []  # no missed ACKs
        result = _run(check_missed_acks(conn))
        assert result == []

    def test_query_filters_on_requires_ack_true(self):
        """The query must filter on requires_ack = TRUE."""
        from quantstack.coordination.event_bus import check_missed_acks

        conn = MockConn()
        conn._cursor._fetchall_val = []
        _run(check_missed_acks(conn))

        sql = conn.calls[0][0]
        assert "requires_ack = TRUE" in sql
        assert "acked_at IS NULL" in sql
