"""System-level alert helpers for deterministic code paths.

Non-LLM code (risk gate, kill switch, corporate actions, factor exposure,
EventBus ACK monitor) calls ``emit_system_alert()`` directly to create
system alerts without routing through an LLM agent.

Shared constants (ALLOWED_CATEGORIES, ALLOWED_SEVERITIES, SEVERITY_ORDER)
are defined here and imported by the LangChain tool module to avoid
duplication.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from loguru import logger

from quantstack.dashboard.events import publish_event
from quantstack.db import db_conn

# ------------------------------------------------------------------
# Shared constants (imported by langchain/system_alert_tools.py)
# ------------------------------------------------------------------

ALLOWED_CATEGORIES = frozenset({
    "risk_breach",
    "service_failure",
    "kill_switch",
    "data_quality",
    "performance_degradation",
    "factor_drift",
    "ack_timeout",
    "thesis_review",
})

ALLOWED_SEVERITIES = frozenset({"info", "warning", "critical", "emergency"})

SEVERITY_ORDER = ["info", "warning", "critical", "emergency"]


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------


async def emit_system_alert(
    category: str,
    severity: str,
    title: str,
    detail: str,
    source: str = "system",
    metadata: dict | None = None,
) -> int:
    """Direct DB insert for system alerts from deterministic code paths.

    Used by: risk_gate, kill_switch, corporate_actions, factor_exposure,
    event_bus_monitor, and any other non-LLM code that needs to raise alerts.

    NOT a LangChain tool -- called directly by Python code.

    Args:
        category: Alert category (validated against ALLOWED_CATEGORIES).
        severity: Alert severity (validated against ALLOWED_SEVERITIES).
        title: One-line summary.
        detail: Full context.
        source: The module/graph that created the alert (for attribution).
        metadata: Optional JSONB payload.

    Returns:
        The auto-generated alert ID (BIGSERIAL).

    Raises:
        ValueError: If category or severity is not in the allowed set.
    """
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(
            f"Invalid alert category {category!r}. "
            f"Allowed: {sorted(ALLOWED_CATEGORIES)}"
        )
    if severity not in ALLOWED_SEVERITIES:
        raise ValueError(
            f"Invalid alert severity {severity!r}. "
            f"Allowed: {sorted(ALLOWED_SEVERITIES)}"
        )

    meta_json = json.dumps(metadata) if metadata else None

    with db_conn() as conn:
        row = conn.execute(
            "INSERT INTO system_alerts "
            "(category, severity, source, title, detail, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            [category, severity, source, title, detail, meta_json],
        ).fetchone()
        alert_id: int = row["id"]

    logger.info(
        "[ALERT] created id=%d category=%s severity=%s title=%s source=%s",
        alert_id, category, severity, title, source,
    )

    # Push to dashboard SSE stream (best-effort, never raises)
    try:
        publish_event(
            graph_name="system",
            node_name="alert_engine",
            event_type="system_alert",
            content=f"[{severity.upper()}] {title}: {detail[:200]}",
            metadata={
                "alert_id": alert_id,
                "category": category,
                "severity": severity,
                **(metadata or {}),
            },
        )
    except Exception:
        logger.debug("Failed to publish alert event to dashboard", exc_info=True)

    # TODO(kbichave): Add Discord webhook notification for CRITICAL/EMERGENCY alerts.
    # Trigger: when DISCORD_WEBHOOK_URL env var is set. See Phase 9 spec item 9.5 for
    # webhook patterns (rate limits, batching, embed formatting).
    return alert_id
