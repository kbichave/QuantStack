"""System-level alert lifecycle tools for LangGraph agents.

Five LLM-facing tools for the supervisor graph to manage operational alerts:
create, acknowledge, escalate, resolve, and query.

All tools share the ``system_alerts`` table with the internal
``emit_system_alert()`` helper in ``tools/functions/system_alerts.py``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Annotated

from langchain_core.tools import tool
from loguru import logger
from pydantic import Field

from quantstack.db import db_conn
from quantstack.tools.functions.system_alerts import (
    ALLOWED_CATEGORIES,
    ALLOWED_SEVERITIES,
    SEVERITY_ORDER,
)


def _relative_age(created_at: datetime) -> str:
    """Format a timestamp as a human-readable relative age string."""
    now = datetime.now(timezone.utc)
    delta = now - created_at
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


@tool
async def create_system_alert(
    category: Annotated[str, Field(description="One of: risk_breach, service_failure, kill_switch, data_quality, performance_degradation, factor_drift, ack_timeout, thesis_review")],
    severity: Annotated[str, Field(description="One of: info, warning, critical, emergency")],
    title: Annotated[str, Field(description="One-line summary of the alert")],
    detail: Annotated[str, Field(description="Full context -- what happened, what state the system was in")],
    metadata: Annotated[dict | None, Field(description="Optional structured context (positions affected, thresholds, etc.)")] = None,
) -> str:
    """Create a new system-level operational alert. Returns alert ID. Use when detecting risk breaches, service failures, data quality issues, or other infrastructure events that need tracking and lifecycle management."""
    if category not in ALLOWED_CATEGORIES:
        return json.dumps({
            "error": f"Invalid category {category!r}. Allowed: {sorted(ALLOWED_CATEGORIES)}"
        })
    if severity not in ALLOWED_SEVERITIES:
        return json.dumps({
            "error": f"Invalid severity {severity!r}. Allowed: {sorted(ALLOWED_SEVERITIES)}"
        })

    meta_json = json.dumps(metadata) if metadata else None
    try:
        with db_conn() as conn:
            row = conn.execute(
                "INSERT INTO system_alerts "
                "(category, severity, source, title, detail, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                [category, severity, "supervisor", title, detail, meta_json],
            ).fetchone()
            alert_id = row["id"]
        logger.info("[ALERT] created id=%d category=%s severity=%s", alert_id, category, severity)
        return json.dumps({"alert_id": alert_id, "status": "open"})
    except Exception as e:
        logger.error("create_system_alert failed: %s", e)
        return json.dumps({"error": str(e)})


@tool
async def acknowledge_alert(
    alert_id: Annotated[int, Field(description="Numeric ID of the alert to acknowledge")],
    agent_name: Annotated[str, Field(description="Name of the agent acknowledging the alert")],
) -> str:
    """Mark a system alert as being investigated. Sets status to 'acknowledged'. Idempotent -- re-acknowledging an already-acknowledged alert is a no-op. Use when an agent starts working on an issue."""
    try:
        with db_conn() as conn:
            row = conn.execute(
                "SELECT status FROM system_alerts WHERE id = %s",
                [alert_id],
            ).fetchone()
            if row is None:
                return json.dumps({"error": f"Alert {alert_id} not found"})
            if row["status"] in ("acknowledged", "resolved"):
                return json.dumps({
                    "alert_id": alert_id,
                    "status": row["status"],
                    "message": f"Already {row['status']}, no change",
                })
            now = datetime.now(timezone.utc)
            conn.execute(
                "UPDATE system_alerts SET status = 'acknowledged', "
                "acknowledged_by = %s, acknowledged_at = %s WHERE id = %s",
                [agent_name, now, alert_id],
            )
        return json.dumps({"alert_id": alert_id, "status": "acknowledged"})
    except Exception as e:
        logger.error("acknowledge_alert(%d) failed: %s", alert_id, e)
        return json.dumps({"error": str(e)})


@tool
async def escalate_alert(
    alert_id: Annotated[int, Field(description="Numeric ID of the alert to escalate")],
    reason: Annotated[str, Field(description="Reason for escalation, appended to detail for audit trail")],
) -> str:
    """Bump alert severity one level and set status to 'escalated'. Severity ladder: info -> warning -> critical -> emergency (ceiling). Use when an issue is worsening or not being resolved in time."""
    try:
        with db_conn() as conn:
            row = conn.execute(
                "SELECT severity, detail FROM system_alerts WHERE id = %s",
                [alert_id],
            ).fetchone()
            if row is None:
                return json.dumps({"error": f"Alert {alert_id} not found"})

            current_severity = row["severity"]
            current_idx = SEVERITY_ORDER.index(current_severity)
            new_idx = min(current_idx + 1, len(SEVERITY_ORDER) - 1)
            new_severity = SEVERITY_ORDER[new_idx]

            existing_detail = row["detail"] or ""
            updated_detail = (
                f"{existing_detail}\n\n[ESCALATION] {reason}" if existing_detail
                else f"[ESCALATION] {reason}"
            )

            now = datetime.now(timezone.utc)
            conn.execute(
                "UPDATE system_alerts SET severity = %s, status = 'escalated', "
                "detail = %s, escalated_at = %s WHERE id = %s",
                [new_severity, updated_detail, now, alert_id],
            )
        return json.dumps({
            "alert_id": alert_id,
            "status": "escalated",
            "old_severity": current_severity,
            "new_severity": new_severity,
        })
    except Exception as e:
        logger.error("escalate_alert(%d) failed: %s", alert_id, e)
        return json.dumps({"error": str(e)})


@tool
async def resolve_alert(
    alert_id: Annotated[int, Field(description="Numeric ID of the alert to resolve")],
    resolution: Annotated[str, Field(description="Resolution notes explaining what was done to fix the issue")],
) -> str:
    """Close a system alert with resolution notes. Sets status to 'resolved'. Idempotent -- re-resolving an already-resolved alert is a no-op. Use when an issue has been fixed or is no longer applicable."""
    try:
        with db_conn() as conn:
            row = conn.execute(
                "SELECT status FROM system_alerts WHERE id = %s",
                [alert_id],
            ).fetchone()
            if row is None:
                return json.dumps({"error": f"Alert {alert_id} not found"})
            if row["status"] == "resolved":
                return json.dumps({
                    "alert_id": alert_id,
                    "status": "resolved",
                    "message": "Already resolved, no change",
                })
            now = datetime.now(timezone.utc)
            conn.execute(
                "UPDATE system_alerts SET status = 'resolved', "
                "resolution = %s, resolved_at = %s WHERE id = %s",
                [resolution, now, alert_id],
            )
        return json.dumps({"alert_id": alert_id, "status": "resolved"})
    except Exception as e:
        logger.error("resolve_alert(%d) failed: %s", alert_id, e)
        return json.dumps({"error": str(e)})


@tool
async def query_system_alerts(
    severity: Annotated[str | None, Field(description="Filter by severity: info, warning, critical, emergency. Omit to include all.")] = None,
    status: Annotated[str | None, Field(description="Filter by status: open, acknowledged, escalated, resolved. Omit to include all.")] = None,
    category: Annotated[str | None, Field(description="Filter by category. Omit to include all.")] = None,
    since_hours: Annotated[int, Field(description="Only return alerts created within this many hours (default 24)")] = 24,
) -> str:
    """Query system alerts with optional filters. Returns formatted alert list ordered by severity DESC, created_at DESC. Use for reviewing active alerts, checking alert history, or finding unresolved issues."""
    try:
        clauses = ["created_at >= %s"]
        params: list = [datetime.now(timezone.utc) - timedelta(hours=since_hours)]

        if severity:
            clauses.append("severity = %s")
            params.append(severity)
        if status:
            clauses.append("status = %s")
            params.append(status)
        if category:
            clauses.append("category = %s")
            params.append(category)

        where = " AND ".join(clauses)
        # Order by severity (emergency first) then newest first
        query = (
            f"SELECT id, category, severity, status, source, title, detail, "
            f"created_at, acknowledged_by, acknowledged_at, resolved_at "
            f"FROM system_alerts WHERE {where} "
            f"ORDER BY CASE severity "
            f"WHEN 'emergency' THEN 0 WHEN 'critical' THEN 1 "
            f"WHEN 'warning' THEN 2 WHEN 'info' THEN 3 END, "
            f"created_at DESC LIMIT 50"
        )

        with db_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return "No system alerts found matching the given filters."

        lines = [f"Found {len(rows)} alert(s):\n"]
        for r in rows:
            age = _relative_age(r["created_at"]) if r["created_at"] else "unknown"
            line = (
                f"[#{r['id']}] {r['severity'].upper()} | {r['category']} | "
                f"{r['status']} | {r['title']} ({age})"
            )
            lines.append(line)

        return "\n".join(lines)
    except Exception as e:
        logger.error("query_system_alerts failed: %s", e)
        return json.dumps({"error": str(e)})
