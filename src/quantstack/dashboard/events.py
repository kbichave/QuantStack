"""Agent event publisher for the real-time dashboard.

Nodes and the agent executor call ``publish_event()`` to log decisions,
tool calls, and LLM responses to the ``agent_events`` table.  The
dashboard SSE endpoint streams these to the UI.

Writes are best-effort — a failed publish never crashes the graph.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def publish_event(
    graph_name: str,
    node_name: str,
    event_type: str,
    content: str,
    agent_name: str = "",
    cycle_number: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write an agent event to PostgreSQL.  Best-effort, never raises."""
    try:
        from quantstack.db import db_conn

        meta_json = json.dumps(metadata or {}, default=str)
        # Truncate content to 4000 chars to avoid bloat
        content_truncated = content[:4000] if content else ""

        with db_conn() as conn:
            conn.execute(
                """INSERT INTO agent_events
                   (graph_name, node_name, agent_name, event_type, content, metadata, cycle_number)
                   VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)""",
                [graph_name, node_name, agent_name, event_type,
                 content_truncated, meta_json, cycle_number],
            )
    except Exception:
        logger.debug("Failed to publish agent event", exc_info=True)
