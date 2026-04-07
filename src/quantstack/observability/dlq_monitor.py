"""Dead Letter Queue monitoring: rate computation and alert thresholds.

Queries the ``agent_dlq`` table for per-agent failure rates over a rolling
window and emits Langfuse events when thresholds are breached.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from quantstack.db import db_conn

logger = logging.getLogger(__name__)

DLQ_WARN_RATE_PCT = 5.0
DLQ_CRITICAL_RATE_PCT = 10.0


def count_dlq_entries(agent_name: str, window_hours: int = 24) -> int:
    """Count DLQ entries for an agent within the rolling window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    with db_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM agent_dlq "
            "WHERE agent_name = ? AND created_at >= ?",
            [agent_name, cutoff],
        ).fetchone()
    return row[0] if row else 0


def compute_dlq_rate(agent_name: str, total_attempts: int, window_hours: int = 24) -> float:
    """Compute DLQ failure rate as a percentage.

    ``total_attempts`` is the denominator (total parse calls for this agent
    in the window). Caller provides this from Langfuse traces or a counter.
    Returns 0.0 if total_attempts is 0.
    """
    if total_attempts <= 0:
        return 0.0
    failures = count_dlq_entries(agent_name, window_hours)
    return (failures / total_attempts) * 100.0


def check_dlq_alerts(agent_name: str, total_attempts: int) -> str | None:
    """Check DLQ rate and return alert level if threshold breached.

    Returns "critical", "warn", or None.
    """
    rate = compute_dlq_rate(agent_name, total_attempts)
    if rate >= DLQ_CRITICAL_RATE_PCT:
        logger.critical(
            "DLQ CRITICAL: agent=%s rate=%.1f%% (threshold=%.1f%%)",
            agent_name, rate, DLQ_CRITICAL_RATE_PCT,
        )
        return "critical"
    if rate >= DLQ_WARN_RATE_PCT:
        logger.warning(
            "DLQ WARNING: agent=%s rate=%.1f%% (threshold=%.1f%%)",
            agent_name, rate, DLQ_WARN_RATE_PCT,
        )
        return "warn"
    return None
