"""System health and coordination functions for graph nodes."""

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import db_conn
from quantstack.execution.broker_factory import get_broker_mode
from quantstack.tools._state import require_ctx


async def check_system_status() -> dict[str, Any]:
    """Check overall system health status.

    Called by supervisor graph's safety_check node.
    Returns dict with service health, kill switch state, and data freshness.
    """
    try:
        ctx = require_ctx()
        ks_status = ctx.kill_switch.status()
        return {
            "success": True,
            "kill_switch_active": ks_status.active,
            "kill_switch_reason": ks_status.reason,
            "risk_halted": ctx.risk_gate.is_halted(),
            "broker_mode": get_broker_mode(),
            "session_id": ctx.session_id,
        }
    except Exception as e:
        logger.error(f"check_system_status failed: {e}")
        return {"success": False, "error": str(e)}


async def check_heartbeat(service: str) -> dict[str, Any]:
    """Check heartbeat freshness for a specific service.

    Args:
        service: Service name ("trading-graph", "research-graph").

    Returns dict with last_heartbeat timestamp and staleness.
    """
    try:
        with db_conn() as conn:
            row = conn.execute(
                """
                SELECT iteration, started_at, finished_at, status
                FROM loop_heartbeats
                WHERE loop_name = ?
                ORDER BY iteration DESC
                LIMIT 1
                """,
                [service],
            ).fetchone()

        if row is None:
            return {
                "success": True,
                "service": service,
                "status": "unknown",
                "last_heartbeat": None,
                "staleness_seconds": None,
            }

        iteration, started_at, finished_at, status = row
        last_ts = finished_at or started_at
        staleness = (datetime.now(timezone.utc) - last_ts).total_seconds() if last_ts else None

        return {
            "success": True,
            "service": service,
            "status": status,
            "last_iteration": iteration,
            "last_heartbeat": last_ts.isoformat() if last_ts else None,
            "staleness_seconds": round(staleness, 1) if staleness is not None else None,
        }
    except Exception as e:
        logger.error(f"check_heartbeat({service}) failed: {e}")
        return {"success": False, "service": service, "error": str(e)}


async def record_heartbeat(
    service: str,
    iteration: int = 0,
    symbols_processed: int = 0,
    errors: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    """Record a heartbeat for the current service.

    Called at the start and end of each graph cycle.

    Args:
        service: Loop identifier ("trading_loop", "research_loop").
        iteration: Monotonically increasing iteration counter.
        symbols_processed: Number of symbols processed.
        errors: Number of errors encountered.
        status: "running", "completed", or "failed".
    """
    try:
        now = datetime.now(timezone.utc)
        with db_conn() as conn:
            if status == "running":
                conn.execute(
                    """
                    INSERT INTO loop_heartbeats (loop_name, iteration, started_at, status)
                    VALUES (?, ?, ?, 'running')
                    ON CONFLICT (loop_name, iteration)
                    DO UPDATE SET started_at = EXCLUDED.started_at, status = 'running'
                    """,
                    [service, iteration, now],
                )
            else:
                conn.execute(
                    """
                    INSERT INTO loop_heartbeats
                        (loop_name, iteration, started_at, finished_at,
                         symbols_processed, errors, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (loop_name, iteration)
                    DO UPDATE SET
                        finished_at = EXCLUDED.finished_at,
                        symbols_processed = EXCLUDED.symbols_processed,
                        errors = EXCLUDED.errors,
                        status = EXCLUDED.status
                    """,
                    [service, iteration, now, now, symbols_processed, errors, status],
                )
        return {"success": True}
    except Exception as e:
        logger.error(f"record_heartbeat({service}) failed: {e}")
        return {"success": False, "error": str(e)}
