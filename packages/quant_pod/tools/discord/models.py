from __future__ import annotations

import json
from datetime import datetime, timezone


class DiscordMessage:
    """Lightweight data class for a single Discord message."""

    __slots__ = (
        "message_id",
        "channel_id",
        "author_id",
        "author_name",
        "content",
        "timestamp",
        "attachments",
        "embeds",
    )

    def __init__(
        self,
        message_id: str,
        channel_id: str,
        author_id: str,
        author_name: str,
        content: str,
        timestamp: datetime,
        attachments: list[dict],
        embeds: list[dict],
    ) -> None:
        self.message_id = message_id
        self.channel_id = channel_id
        self.author_id = author_id
        self.author_name = author_name
        self.content = content
        self.timestamp = timestamp
        self.attachments = attachments
        self.embeds = embeds

    @classmethod
    def from_api(cls, data: dict, channel_id: str) -> DiscordMessage:
        """Build from a raw Discord REST API message object."""
        author = data.get("author", {})
        display_name = (
            author.get("global_name")
            or author.get("display_name")
            or author.get("username", "unknown")
        )
        ts_raw = data["timestamp"].replace("Z", "+00:00")
        return cls(
            message_id=data["id"],
            channel_id=channel_id,
            author_id=author.get("id", ""),
            author_name=display_name,
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(ts_raw),
            attachments=data.get("attachments", []),
            embeds=data.get("embeds", []),
        )

    @property
    def attachments_json(self) -> str:
        return json.dumps(self.attachments)

    @property
    def embeds_json(self) -> str:
        return json.dumps(self.embeds)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "channel_id": self.channel_id,
            "author_id": self.author_id,
            "author": self.author_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "attachments": self.attachments,
            "embeds": self.embeds,
        }
