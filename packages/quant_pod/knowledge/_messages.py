# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Messages mixin — Agent message CRUD for KnowledgeStore."""

import json
from datetime import datetime, timedelta

import duckdb

from quant_pod.knowledge.models import AgentMessage


class MessagesMixin:
    """Agent message operations."""

    conn: duckdb.DuckDBPyConnection

    # =========================================================================
    # AGENT MESSAGE OPERATIONS
    # =========================================================================

    def send_message(self, message: AgentMessage) -> int:
        """Send an agent message."""
        data = message.model_dump()
        data["data"] = json.dumps(data["data"])

        cols = [k for k in data.keys() if k != "id"]
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)

        result = self.conn.execute(
            f"INSERT INTO agent_messages ({col_names}) VALUES ({placeholders}) RETURNING id",
            [data[k] for k in cols],
        ).fetchone()

        self.conn.commit()
        return result[0]

    def get_messages(
        self,
        to_agent: str | None = None,
        unacknowledged_only: bool = False,
        hours: int = 24,
    ) -> list[AgentMessage]:
        """Get messages for an agent."""
        query = "SELECT * FROM agent_messages WHERE timestamp > ?"
        params = [datetime.now() - timedelta(hours=hours)]

        if to_agent:
            query += " AND (to_agent = ? OR to_agent IS NULL)"
            params.append(to_agent)
        if unacknowledged_only:
            query += " AND acknowledged = FALSE"

        query += " ORDER BY priority ASC, timestamp DESC"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        messages = []
        for row in results:
            data = dict(zip(cols, row, strict=False))
            if data.get("data"):
                data["data"] = json.loads(data["data"])
            messages.append(AgentMessage(**data))

        return messages

    def acknowledge_message(self, message_id: int) -> None:
        """Acknowledge a message."""
        self.conn.execute(
            "UPDATE agent_messages SET acknowledged = TRUE, acknowledged_at = ? WHERE id = ?",
            [datetime.now(), message_id],
        )
        self.conn.commit()
