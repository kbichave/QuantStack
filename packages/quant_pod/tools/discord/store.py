from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
from loguru import logger

from .models import DiscordMessage

# Default to the project's main DuckDB so Discord messages sit alongside
# market data and can be JOINed directly in research queries.
_DEFAULT_DB = str(Path(__file__).parents[4] / "data" / "trader.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS discord_messages (
    message_id   VARCHAR NOT NULL,
    channel_id   VARCHAR NOT NULL,
    channel_label VARCHAR NOT NULL,   -- e.g. "watchlist" or "results"
    author_id    VARCHAR NOT NULL,
    author_name  VARCHAR NOT NULL,
    content      TEXT    NOT NULL,
    timestamp    TIMESTAMPTZ NOT NULL,
    attachments  TEXT,                -- JSON array
    embeds       TEXT,                -- JSON array
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
    DuckDB-backed persistence for Discord messages.

    Opens a new connection per operation — DuckDB handles concurrent readers
    safely and we avoid holding a long-lived connection across async awaits.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        raw = db_path or os.getenv("DISCORD_DB_PATH") or _DEFAULT_DB
        self.db_path = str(Path(raw).expanduser().resolve())
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)
        logger.debug(f"Discord schema ready at {self.db_path}")

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

        with self._connect() as conn:
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

    def get_latest_message_id(self, channel_id: str) -> Optional[str]:
        """
        Return the snowflake ID of the most recently stored message for a
        given channel. Used as the `after=` cursor for incremental fetches.
        """
        with self._connect() as conn:
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
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
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

        with self._connect() as conn:
            df = conn.execute(
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
            ).df()

        # Serialize timestamps so callers get plain strings
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(str)

        return df.to_dict(orient="records")

    def get_status(self) -> list[dict]:
        """Return per-channel summary stats for the fetch_status tool."""
        with self._connect() as conn:
            df = conn.execute(
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
            ).df()

        for col in ("earliest", "latest", "last_fetched_at"):
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df.to_dict(orient="records")
