"""Tests for event bus cursor atomicity (section-09).

Verifies the DELETE+INSERT anti-pattern is replaced with a single upsert.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event_bus(mock_conn):
    """Create an EventBus with a mocked connection."""
    from quantstack.coordination.event_bus import EventBus

    bus = EventBus.__new__(EventBus)
    bus._conn = mock_conn
    return bus


def _make_mock_conn_returning_events(rows):
    """Mock conn where execute().fetchall() returns given rows for the poll query."""
    conn = MagicMock()

    def _execute_side_effect(sql, params=None):
        result = MagicMock()
        if "SELECT" in sql and "loop_events" in sql:
            result.fetchall = MagicMock(return_value=rows)
        elif "SELECT" in sql and "loop_cursors" in sql:
            result.fetchone = MagicMock(return_value=None)
        else:
            result.fetchall = MagicMock(return_value=[])
            result.fetchone = MagicMock(return_value=None)
        return result

    conn.execute = MagicMock(side_effect=_execute_side_effect)
    return conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_upsert_replaces_delete_insert():
    """Cursor update uses ON CONFLICT upsert, not DELETE+INSERT."""
    rows = [
        (1, "signal_update", "trading", '{"foo": 1}', datetime.now(timezone.utc)),
    ]
    conn = _make_mock_conn_returning_events(rows)
    bus = _make_event_bus(conn)

    bus.poll("test_consumer")

    all_sqls = [c.args[0] for c in conn.execute.call_args_list]
    cursor_sqls = [s for s in all_sqls if "loop_cursors" in s and "loop_events" not in s]

    # Must have ON CONFLICT in cursor update
    on_conflict_sqls = [s for s in cursor_sqls if "ON CONFLICT" in s]
    assert len(on_conflict_sqls) >= 1, f"Expected ON CONFLICT upsert, got: {cursor_sqls}"

    # Must NOT have DELETE FROM loop_cursors
    delete_sqls = [s for s in cursor_sqls if "DELETE" in s]
    assert len(delete_sqls) == 0, f"DELETE+INSERT pattern still present: {cursor_sqls}"


def test_new_consumer_creates_cursor():
    """poll() with a new consumer_id creates a cursor row via upsert INSERT path."""
    rows = [
        (42, "signal_update", "trading", '{}', datetime.now(timezone.utc)),
    ]
    conn = _make_mock_conn_returning_events(rows)
    bus = _make_event_bus(conn)

    bus.poll("brand_new_consumer")

    all_sqls = [c.args[0] for c in conn.execute.call_args_list]
    upsert_sqls = [s for s in all_sqls if "ON CONFLICT" in s and "loop_cursors" in s]
    assert len(upsert_sqls) == 1

    # Verify consumer_id was passed
    upsert_call = [c for c in conn.execute.call_args_list if "ON CONFLICT" in c.args[0]][0]
    assert "brand_new_consumer" in upsert_call.args[1]


def test_existing_consumer_updates_cursor():
    """poll() with events updates the cursor to the last event's ID."""
    rows = [
        (10, "signal_update", "trading", '{}', datetime.now(timezone.utc)),
        (20, "regime_change", "supervisor", '{}', datetime.now(timezone.utc)),
    ]
    conn = _make_mock_conn_returning_events(rows)
    bus = _make_event_bus(conn)

    bus.poll("existing_consumer")

    upsert_calls = [
        c for c in conn.execute.call_args_list if "ON CONFLICT" in c.args[0]
    ]
    assert len(upsert_calls) == 1
    params = upsert_calls[0].args[1]
    # params should be [consumer_id, last_event_id, last_polled_at]
    assert params[0] == "existing_consumer"
    assert params[1] == 20  # last event's ID


def test_poll_with_no_events_updates_polled_at():
    """poll() with no events sets last_event_id=NULL and updates last_polled_at."""
    conn = _make_mock_conn_returning_events([])  # no events
    bus = _make_event_bus(conn)

    before = datetime.now(timezone.utc)
    bus.poll("heartbeat_consumer")

    upsert_calls = [
        c for c in conn.execute.call_args_list if "ON CONFLICT" in c.args[0]
    ]
    assert len(upsert_calls) == 1
    params = upsert_calls[0].args[1]
    assert params[0] == "heartbeat_consumer"
    assert params[1] is None  # no events → NULL
    assert params[2] >= before  # last_polled_at is recent


def test_concurrent_cursor_updates_no_lost_cursors():
    """Two consumers polling concurrently both get their cursor rows."""
    rows_a = [(1, "signal_update", "trading", '{}', datetime.now(timezone.utc))]
    rows_b = [(2, "regime_change", "supervisor", '{}', datetime.now(timezone.utc))]

    conn_a = _make_mock_conn_returning_events(rows_a)
    conn_b = _make_mock_conn_returning_events(rows_b)
    bus_a = _make_event_bus(conn_a)
    bus_b = _make_event_bus(conn_b)

    bus_a.poll("consumer_a")
    bus_b.poll("consumer_b")

    # Both should have issued upserts (not interfering with each other)
    upserts_a = [c for c in conn_a.execute.call_args_list if "ON CONFLICT" in c.args[0]]
    upserts_b = [c for c in conn_b.execute.call_args_list if "ON CONFLICT" in c.args[0]]
    assert len(upserts_a) == 1
    assert len(upserts_b) == 1
    assert upserts_a[0].args[1][0] == "consumer_a"
    assert upserts_b[0].args[1][0] == "consumer_b"


def test_pgconnection_supports_on_conflict():
    """The upsert SQL string is well-formed and contains required clauses."""
    conn = _make_mock_conn_returning_events([])
    bus = _make_event_bus(conn)

    bus.poll("syntax_check")

    upsert_calls = [c for c in conn.execute.call_args_list if "ON CONFLICT" in c.args[0]]
    assert len(upsert_calls) == 1
    sql = upsert_calls[0].args[0]
    assert "INSERT INTO loop_cursors" in sql
    assert "ON CONFLICT (consumer_id)" in sql
    assert "DO UPDATE SET" in sql
    assert "EXCLUDED.last_event_id" in sql
    assert "EXCLUDED.last_polled_at" in sql
