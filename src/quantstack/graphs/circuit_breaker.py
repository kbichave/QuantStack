"""Node circuit breaker: three-state model backed by PostgreSQL.

Decorates graph node functions to track consecutive failures and skip
invocation when a node is unhealthy. Uses the ``circuit_breaker_state``
table created by the Phase 4 DB migration (section-01).

States:
  closed  — normal operation, failures counted
  open    — node skipped, safe default returned
  half_open — probe: one request allowed, success resets, failure re-opens
"""
from __future__ import annotations

import enum
import functools
import logging
from datetime import datetime, timezone

from quantstack.db import db_conn

logger = logging.getLogger(__name__)


class FailureType(enum.Enum):
    """Classifies node failures for circuit breaker decision-making."""
    EXECUTION = "execution"       # generic runtime error
    RATE_LIMIT = "rate_limit"     # HTTP 429 — trips immediately
    PROVIDER_OUTAGE = "provider_outage"  # 5xx / connection — trips immediately
    TOKEN_LIMIT = "token_limit"   # does NOT trip breaker, route to pruning
    PARSE_FAILURE = "parse_failure"  # counted separately


def classify_failure(exc: BaseException) -> FailureType:
    """Inspect exception to determine failure type."""
    msg = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    if "429" in msg or "rate_limit" in msg or "ratelimit" in exc_type:
        return FailureType.RATE_LIMIT
    if "500" in msg or "502" in msg or "503" in msg or "connection" in msg:
        return FailureType.PROVIDER_OUTAGE
    if "token" in msg and ("limit" in msg or "exceed" in msg or "maximum" in msg):
        return FailureType.TOKEN_LIMIT
    if "json" in msg or "parse" in msg or "validation" in msg or "decode" in msg:
        return FailureType.PARSE_FAILURE
    return FailureType.EXECUTION


def _read_breaker_state(conn, breaker_key: str) -> dict:
    """Read current circuit breaker state from DB. Returns defaults if no row."""
    row = conn.execute(
        "SELECT state, failure_count, opened_at, cooldown_seconds, last_success_at "
        "FROM circuit_breaker_state WHERE breaker_key = ?",
        [breaker_key],
    ).fetchone()
    if not row:
        return {
            "state": "closed",
            "failure_count": 0,
            "opened_at": None,
            "cooldown_seconds": 300,
            "last_success_at": None,
        }
    return {
        "state": row[0],
        "failure_count": row[1],
        "opened_at": row[2],
        "cooldown_seconds": row[3],
        "last_success_at": row[4],
    }


def _record_success(conn, breaker_key: str) -> None:
    """Reset breaker to closed with zero failures."""
    now = datetime.now(timezone.utc)
    conn.execute(
        "INSERT INTO circuit_breaker_state (breaker_key, state, failure_count, last_success_at) "
        "VALUES (?, 'closed', 0, ?) "
        "ON CONFLICT (breaker_key) DO UPDATE SET "
        "state = 'closed', failure_count = 0, last_success_at = EXCLUDED.last_success_at",
        [breaker_key, now],
    )


def _record_failure(conn, breaker_key: str, threshold: int, cooldown_seconds: int) -> int:
    """Increment failure count atomically. Returns new count. Trips breaker if threshold hit."""
    now = datetime.now(timezone.utc)
    row = conn.execute(
        "INSERT INTO circuit_breaker_state "
        "(breaker_key, state, failure_count, last_failure_at, cooldown_seconds) "
        "VALUES (?, 'closed', 1, ?, ?) "
        "ON CONFLICT (breaker_key) DO UPDATE SET "
        "failure_count = circuit_breaker_state.failure_count + 1, "
        "last_failure_at = EXCLUDED.last_failure_at "
        "RETURNING failure_count",
        [breaker_key, now, cooldown_seconds],
    ).fetchone()
    new_count = row[0] if row else 1

    if new_count >= threshold:
        conn.execute(
            "UPDATE circuit_breaker_state SET state = 'open', opened_at = ? "
            "WHERE breaker_key = ?",
            [now, breaker_key],
        )
    return new_count


def _trip_immediately(conn, breaker_key: str, cooldown_seconds: int) -> None:
    """Trip breaker immediately (rate limit / provider outage)."""
    now = datetime.now(timezone.utc)
    conn.execute(
        "INSERT INTO circuit_breaker_state "
        "(breaker_key, state, failure_count, last_failure_at, opened_at, cooldown_seconds) "
        "VALUES (?, 'open', 1, ?, ?, ?) "
        "ON CONFLICT (breaker_key) DO UPDATE SET "
        "state = 'open', failure_count = circuit_breaker_state.failure_count + 1, "
        "last_failure_at = EXCLUDED.last_failure_at, opened_at = EXCLUDED.opened_at",
        [breaker_key, now, now, cooldown_seconds],
    )


def _transition_to_half_open(conn, breaker_key: str) -> None:
    """Transition from open to half_open (cooldown expired)."""
    conn.execute(
        "UPDATE circuit_breaker_state SET state = 'half_open' WHERE breaker_key = ?",
        [breaker_key],
    )


def circuit_breaker(
    threshold: int = 3,
    alert_threshold: int = 5,
    cooldown_seconds: int = 300,
    graph_name: str = "trading",
):
    """Decorator for graph node functions with circuit breaker protection.

    The decorated function must accept ``state`` as first arg and its class
    must have a ``safe_default()`` classmethod on its return type.
    """
    def decorator(func):
        # Try to find the output model class from the function's return annotation
        _output_model = None
        annotations = getattr(func, "__annotations__", {})
        ret = annotations.get("return")
        if ret and hasattr(ret, "safe_default"):
            _output_model = ret

        @functools.wraps(func)
        async def wrapper(state, *args, **kwargs):
            node_name = func.__name__
            breaker_key = f"{graph_name}/{node_name}"

            try:
                with db_conn() as conn:
                    bs = _read_breaker_state(conn, breaker_key)
            except Exception:
                # DB unavailable — fail open (let the node run)
                logger.warning("Circuit breaker DB read failed for %s — failing open", breaker_key)
                return await func(state, *args, **kwargs)

            now = datetime.now(timezone.utc)

            if bs["state"] == "open":
                # Check cooldown
                opened_at = bs["opened_at"]
                if opened_at and (now - opened_at).total_seconds() >= bs["cooldown_seconds"]:
                    # Cooldown expired → half-open probe
                    try:
                        with db_conn() as conn:
                            _transition_to_half_open(conn, breaker_key)
                    except Exception:
                        pass
                    logger.info("Circuit breaker %s: half-open probe", breaker_key)
                else:
                    # Still in cooldown → skip
                    logger.warning("Circuit breaker %s: OPEN — returning safe default", breaker_key)
                    if _output_model:
                        return _output_model.safe_default()
                    return {}

            # Closed or half-open: invoke the node
            try:
                result = await func(state, *args, **kwargs)
                # Success
                try:
                    with db_conn() as conn:
                        _record_success(conn, breaker_key)
                except Exception:
                    pass
                return result

            except Exception as exc:
                failure_type = classify_failure(exc)

                if failure_type == FailureType.TOKEN_LIMIT:
                    # Don't trip — route to message pruning
                    logger.warning("Circuit breaker %s: token limit — not tripping", breaker_key)
                    raise

                try:
                    with db_conn() as conn:
                        if failure_type in (FailureType.RATE_LIMIT, FailureType.PROVIDER_OUTAGE):
                            _trip_immediately(conn, breaker_key, cooldown_seconds)
                            new_count = threshold  # already tripped
                        else:
                            new_count = _record_failure(conn, breaker_key, threshold, cooldown_seconds)

                        if new_count >= alert_threshold:
                            logger.critical(
                                "ALERT: Circuit breaker %s hit alert threshold (%d failures)",
                                breaker_key, new_count,
                            )
                except Exception:
                    logger.warning("Circuit breaker DB write failed for %s", breaker_key)

                # Return safe default instead of raising
                logger.error("Circuit breaker %s: node failed (%s) — safe default", breaker_key, exc)
                if _output_model:
                    return _output_model.safe_default()
                return {}

        return wrapper
    return decorator
