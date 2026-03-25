from __future__ import annotations

from datetime import datetime

from loguru import logger

from quantstack.db import pg_conn

from .models import DiscordMessage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS discord_messages (
    message_id   VARCHAR NOT NULL,
    channel_id   VARCHAR NOT NULL,
    channel_label VARCHAR NOT NULL,
    author_id    VARCHAR NOT NULL,
    author_name  VARCHAR NOT NULL,
    content      TEXT    NOT NULL,
    timestamp    TIMESTAMPTZ NOT NULL,
    attachments  TEXT,
    embeds       TEXT,
    fetched_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (message_id)
);

CREATE INDEX IF NOT EXISTS idx_discord_label_ts
    ON discord_messages (channel_label, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_discord_channel_ts
    ON discord_messages (channel_id, timestamp DESC);
"""


class DiscordStore:
    """
    PostgreSQL persistence for Discord messages.

    Uses the shared PostgreSQL connection pool. Each method acquires a
    connection from the pool for the duration of the operation.
    """

    def __init__(self) -> None:
        self._init_schema()

    def _init_schema(self) -> None:
        with pg_conn() as conn:
            conn.execute(_SCHEMA)
        logger.debug("Discord schema ready (PostgreSQL)")

    def upsert_messages(self, messages: list[DiscordMessage], label: str) -> int:
        """
        Insert messages, silently skipping duplicates (idempotent).
        Returns the number of rows passed (not just newly inserted).
        """
        if not messages:
            return 0

        rows = [
            (
                m.message_id,
                m.channel_id,
                label,
                m.author_id,
                m.author_name,
                m.content,
                m.timestamp,
                m.attachments_json,
                m.embeds_json,
            )
            for m in messages
        ]

        with pg_conn() as conn:
            conn.executemany(
                """
                INSERT INTO discord_messages
                    (message_id, channel_id, channel_label, author_id, author_name,
                     content, timestamp, attachments, embeds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (message_id) DO NOTHING
                """,
                rows,
            )

        return len(rows)

    def get_latest_message_id(self, channel_id: str) -> str | None:
        """
        Return the snowflake ID of the most recently stored message for a
        given channel. Used as the `after=` cursor for incremental fetches.
        """
        with pg_conn() as conn:
            row = conn.execute(
                """
                SELECT message_id
                FROM   discord_messages
                WHERE  channel_id = ?
                ORDER  BY timestamp DESC
                LIMIT  1
                """,
                [channel_id],
            ).fetchone()
        return row[0] if row else None

    def get_messages(
        self,
        label: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query stored messages for a label, optionally bounded by date."""
        clauses = ["channel_label = ?"]
        params: list = [label]

        if date_from:
            clauses.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("timestamp <= ?")
            params.append(date_to)

        params.append(min(limit, 1000))
        where = " AND ".join(clauses)

        with pg_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT message_id, channel_id, channel_label,
                       author_id, author_name, content, timestamp,
                       attachments, embeds
                FROM   discord_messages
                WHERE  {where}
                ORDER  BY timestamp DESC
                LIMIT  ?
                """,
                params,
            ).fetchall()
            columns = [desc[0] for desc in conn.description]

        records = [dict(zip(columns, row)) for row in rows]
        # Serialize timestamps so callers get plain strings
        for rec in records:
            if "timestamp" in rec and rec["timestamp"] is not None:
                rec["timestamp"] = str(rec["timestamp"])

        return records

    def get_status(self) -> list[dict]:
        """Return per-channel summary stats for the fetch_status tool."""
        with pg_conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    channel_label,
                    channel_id,
                    COUNT(*)          AS message_count,
                    MIN(timestamp)    AS earliest,
                    MAX(timestamp)    AS latest,
                    MAX(fetched_at)   AS last_fetched_at
                FROM   discord_messages
                GROUP  BY channel_label, channel_id
                ORDER  BY channel_label, latest DESC
                """
            ).fetchall()
            columns = [desc[0] for desc in conn.description]

        records = [dict(zip(columns, row)) for row in rows]
        for rec in records:
            for col in ("earliest", "latest", "last_fetched_at"):
                if col in rec and rec[col] is not None:
                    rec[col] = str(rec[col])

        return records
