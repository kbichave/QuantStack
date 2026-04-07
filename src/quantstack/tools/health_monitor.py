"""Daily tool health check — auto-disables tools with low success rates.

Queries the tool_health table for 7-day trailing metrics and moves tools
below the success threshold to DEGRADED status, publishing TOOL_DISABLED
events via the event bus.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loguru import logger

from quantstack.coordination.event_bus import Event, EventBus, EventType
from quantstack.db import PgConnection

# Tools with a 7-day success rate below this threshold are auto-disabled.
SUCCESS_RATE_THRESHOLD = 0.50

# Minimum invocations before a tool can be auto-disabled (avoid disabling
# tools that failed once on their first invocation).
MIN_INVOCATIONS = 5


def run_daily_health_check(
    conn: PgConnection,
    event_bus: EventBus | None = None,
) -> list[str]:
    """Check tool_health for low success rate tools and disable them.

    Args:
        conn: Active PgConnection for querying tool_health.
        event_bus: Optional EventBus to publish TOOL_DISABLED events.

    Returns:
        List of tool names that were moved to DEGRADED status.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    rows = conn.execute(
        """
        SELECT tool_name, invocation_count, success_count, failure_count,
               avg_latency_ms, last_error
        FROM tool_health
        WHERE last_invoked >= %s
          AND invocation_count >= %s
        """,
        [cutoff, MIN_INVOCATIONS],
    ).fetchall()

    if not rows:
        logger.info("[HealthCheck] No tools with sufficient invocations in last 7 days")
        return []

    # Lazy import to avoid circular dependency at module level
    from quantstack.tools.registry import ACTIVE_TOOLS, move_tool

    degraded: list[str] = []

    for row in rows:
        tool_name = row[0]
        invocation_count = row[1]
        success_count = row[2]
        failure_count = row[3]
        avg_latency_ms = row[4]
        last_error = row[5]

        if invocation_count == 0:
            continue

        success_rate = success_count / invocation_count

        if success_rate < SUCCESS_RATE_THRESHOLD and tool_name in ACTIVE_TOOLS:
            logger.warning(
                "[HealthCheck] Disabling %s: success_rate=%.2f (%d/%d), last_error=%s",
                tool_name, success_rate, success_count, invocation_count, last_error,
            )

            try:
                move_tool(tool_name, "active", "degraded")
            except KeyError:
                # Already moved or not in active — skip
                continue

            # Update status in DB
            conn.execute(
                "UPDATE tool_health SET status = 'degraded' WHERE tool_name = %s",
                [tool_name],
            )

            # Publish event
            if event_bus is not None:
                event_bus.publish(Event(
                    event_type=EventType.TOOL_DISABLED,
                    source_loop="health_monitor",
                    payload={
                        "tool_name": tool_name,
                        "success_rate": round(success_rate, 4),
                        "invocation_count": invocation_count,
                        "failure_count": failure_count,
                        "avg_latency_ms": round(avg_latency_ms, 2) if avg_latency_ms else None,
                        "last_error": last_error,
                    },
                ))

            degraded.append(tool_name)

    logger.info(
        "[HealthCheck] Complete: %d tools checked, %d degraded",
        len(rows), len(degraded),
    )
    return degraded
