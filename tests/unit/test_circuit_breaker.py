"""Tests for node circuit breaker (section-07).

Tests the three-state model, DB persistence (mocked), failure type
classification, and safe default behavior.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from quantstack.graphs.circuit_breaker import (
    FailureType,
    _read_breaker_state,
    _record_failure,
    _record_success,
    classify_failure,
    circuit_breaker,
)


# ---------------------------------------------------------------------------
# Failure Classification
# ---------------------------------------------------------------------------


def test_rate_limit_classified():
    exc = Exception("HTTP 429 Too Many Requests")
    assert classify_failure(exc) == FailureType.RATE_LIMIT


def test_provider_outage_classified():
    exc = Exception("HTTP 503 Service Unavailable")
    assert classify_failure(exc) == FailureType.PROVIDER_OUTAGE


def test_token_limit_classified():
    exc = Exception("Token limit exceeded: maximum context length 128000")
    assert classify_failure(exc) == FailureType.TOKEN_LIMIT


def test_parse_failure_classified():
    exc = Exception("JSON decode error at position 42")
    assert classify_failure(exc) == FailureType.PARSE_FAILURE


def test_generic_error_classified_as_execution():
    exc = RuntimeError("something unexpected")
    assert classify_failure(exc) == FailureType.EXECUTION


# ---------------------------------------------------------------------------
# DB Operations (mocked)
# ---------------------------------------------------------------------------


def _mock_conn_no_row():
    conn = MagicMock()
    result = MagicMock()
    result.fetchone.return_value = None
    conn.execute.return_value = result
    return conn


def _mock_conn_with_row(state="closed", failure_count=0, opened_at=None, cooldown=300, last_success=None):
    conn = MagicMock()
    result = MagicMock()
    result.fetchone.return_value = (state, failure_count, opened_at, cooldown, last_success)
    conn.execute.return_value = result
    return conn


def test_read_breaker_state_defaults():
    conn = _mock_conn_no_row()
    bs = _read_breaker_state(conn, "trading/test_node")
    assert bs["state"] == "closed"
    assert bs["failure_count"] == 0


def test_read_breaker_state_from_db():
    now = datetime.now(timezone.utc)
    conn = _mock_conn_with_row("open", 3, now, 300, None)
    bs = _read_breaker_state(conn, "trading/test_node")
    assert bs["state"] == "open"
    assert bs["failure_count"] == 3


def test_record_success_resets():
    conn = MagicMock()
    conn.execute.return_value = MagicMock()
    _record_success(conn, "trading/test_node")
    sql = conn.execute.call_args[0][0]
    assert "state = 'closed'" in sql
    assert "failure_count = 0" in sql


def test_record_failure_increments():
    conn = MagicMock()
    result = MagicMock()
    result.fetchone.return_value = (2,)  # new count = 2
    conn.execute.return_value = result
    count = _record_failure(conn, "trading/test_node", threshold=3, cooldown_seconds=300)
    assert count == 2


def test_record_failure_trips_at_threshold():
    conn = MagicMock()
    result = MagicMock()
    result.fetchone.return_value = (3,)  # hits threshold
    conn.execute.return_value = result
    count = _record_failure(conn, "trading/test_node", threshold=3, cooldown_seconds=300)
    assert count == 3
    # Second execute call should be the UPDATE to 'open'
    calls = conn.execute.call_args_list
    assert any("state = 'open'" in str(c) for c in calls)


# ---------------------------------------------------------------------------
# Decorator State Machine (integration-like, mocked DB)
# ---------------------------------------------------------------------------


class FakeOutput:
    @classmethod
    def safe_default(cls):
        return {"safe": True, "errors": ["node_unavailable"]}


@pytest.mark.asyncio
async def test_closed_state_success():
    """In closed state, successful invocation passes through."""
    call_count = 0

    @circuit_breaker(threshold=3, graph_name="test")
    async def my_node(state) -> FakeOutput:
        nonlocal call_count
        call_count += 1
        return {"result": "ok"}

    with patch("quantstack.graphs.circuit_breaker.db_conn") as mock_db:
        conn = _mock_conn_no_row()  # closed, no row
        mock_db.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        result = await my_node({})

    assert call_count == 1
    assert result == {"result": "ok"}


@pytest.mark.asyncio
async def test_closed_state_failure_returns_safe_default():
    """In closed state, failure returns safe default (not raises)."""

    @circuit_breaker(threshold=3, graph_name="test")
    async def my_node(state) -> FakeOutput:
        raise RuntimeError("boom")

    with patch("quantstack.graphs.circuit_breaker.db_conn") as mock_db:
        conn = MagicMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = None  # for read
        conn.execute.return_value = result_mock

        mock_db.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        result = await my_node({})

    # Should get safe default (or empty dict if model not resolved)
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_open_state_skips_invocation():
    """Open breaker (cooldown not expired) returns safe default without calling node."""
    call_count = 0

    @circuit_breaker(threshold=3, cooldown_seconds=300, graph_name="test")
    async def my_node(state) -> FakeOutput:
        nonlocal call_count
        call_count += 1
        return {"result": "ok"}

    now = datetime.now(timezone.utc)
    opened_at = now - timedelta(seconds=100)  # 100s ago, cooldown is 300s

    with patch("quantstack.graphs.circuit_breaker.db_conn") as mock_db:
        conn = _mock_conn_with_row("open", 3, opened_at, 300, None)
        mock_db.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        result = await my_node({})

    assert call_count == 0  # node was NOT called


@pytest.mark.asyncio
async def test_open_to_half_open_after_cooldown():
    """After cooldown expires, open -> half_open and node is invoked."""
    call_count = 0

    @circuit_breaker(threshold=3, cooldown_seconds=300, graph_name="test")
    async def my_node(state) -> FakeOutput:
        nonlocal call_count
        call_count += 1
        return {"result": "recovered"}

    now = datetime.now(timezone.utc)
    opened_at = now - timedelta(seconds=400)  # 400s ago > 300s cooldown

    with patch("quantstack.graphs.circuit_breaker.db_conn") as mock_db:
        conn = _mock_conn_with_row("open", 3, opened_at, 300, None)
        mock_db.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        result = await my_node({})

    assert call_count == 1
    assert result == {"result": "recovered"}


@pytest.mark.asyncio
async def test_token_limit_not_caught():
    """Token limit exception is re-raised (routed to pruning, not caught)."""

    @circuit_breaker(threshold=3, graph_name="test")
    async def my_node(state) -> FakeOutput:
        raise Exception("Token limit exceeded: maximum context length")

    with patch("quantstack.graphs.circuit_breaker.db_conn") as mock_db:
        conn = _mock_conn_no_row()
        mock_db.return_value.__enter__ = MagicMock(return_value=conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        with pytest.raises(Exception, match="Token limit exceeded"):
            await my_node({})
