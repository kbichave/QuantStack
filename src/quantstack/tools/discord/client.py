from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import aiohttp
from loguru import logger

from .models import DiscordMessage

# Discord's internal epoch: January 1, 2015 00:00:00 UTC (in milliseconds)
_DISCORD_EPOCH_MS = 1420070400000

DISCORD_API_BASE = "https://discord.com/api/v10"


def snowflake_from_datetime(dt: datetime) -> int:
    """Convert a UTC datetime to a Discord snowflake ID for use as an API cursor."""
    ms = int(dt.timestamp() * 1000) - _DISCORD_EPOCH_MS
    return max(0, ms) << 22


def datetime_from_snowflake(snowflake: int) -> datetime:
    """Extract the UTC timestamp embedded in a Discord snowflake ID."""
    ms = (int(snowflake) >> 22) + _DISCORD_EPOCH_MS
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


class DiscordClient:
    """
    Thin async client for the Discord REST API.

    Supports both bot tokens (prefix "Bot <token>") and user tokens (raw value).
    User tokens let you read channels you're already a member of without needing
    server admin to add a bot.

    Rate limit handling: respects 429 responses with Retry-After and applies a
    conservative 0.5 s inter-request sleep to avoid hitting per-channel limits.
    """

    def __init__(self, token: str, token_type: str = "user") -> None:
        # Authorization header format:
        #   bot  → "Bot <token>"   (bot application added to the server)
        #   user → "<token>"       (your own account token, no prefix)
        #
        # token_type is read from DISCORD_TOKEN_TYPE env var (default "user").
        # You can also pass the token pre-formatted as "Bot <value>" and it
        # will be used as-is regardless of token_type.
        if token.startswith("Bot "):
            self._auth = token  # already formatted
        elif token_type.lower() == "bot":
            self._auth = f"Bot {token}"
        else:
            self._auth = token  # user token — no prefix
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> DiscordClient:
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": self._auth,
                "User-Agent": "DiscordWatchlistMCP/1.0 (trading-research)",
                "Content-Type": "application/json",
            }
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(self, method: str, path: str, **kwargs: object) -> list | dict:
        if self._session is None:
            raise RuntimeError("Use DiscordClient as an async context manager.")

        url = f"{DISCORD_API_BASE}{path}"
        for _attempt in range(4):
            async with self._session.request(method, url, **kwargs) as resp:  # type: ignore[arg-type]
                if resp.status == 429:
                    body = await resp.json()
                    wait: float = body.get("retry_after", 1.0)
                    logger.warning(
                        f"Discord rate-limited on {path}. Waiting {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                    continue

                if resp.status == 401:
                    raise ValueError(
                        "Discord authentication failed. Check that DISCORD_TOKEN is correct."
                    )
                if resp.status == 403:
                    raise PermissionError(
                        f"No permission to access {path}. "
                        "Ensure your token has access to this channel/guild."
                    )

                resp.raise_for_status()
                return await resp.json()  # type: ignore[return-value]

        raise RuntimeError(f"Discord request failed after 4 attempts: {path}")

    # ── Discovery ──────────────────────────────────────────────────────────

    async def get_guild_channels(self, guild_id: str) -> list[dict]:
        return await self._request("GET", f"/guilds/{guild_id}/channels")  # type: ignore[return-value]

    # ── Core message fetch ─────────────────────────────────────────────────

    async def _get_messages_page(
        self,
        channel_id: str,
        *,
        before: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        params: dict[str, str | int] = {"limit": min(limit, 100)}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        result = await self._request(
            "GET", f"/channels/{channel_id}/messages", params=params
        )
        return result  # type: ignore[return-value]

    # ── Historical bulk fetch ──────────────────────────────────────────────

    async def get_all_messages_since(
        self,
        channel_id: str,
        since: datetime,
        *,
        progress_callback: callable | None = None,  # type: ignore[type-arg]
    ) -> list[DiscordMessage]:
        """
        Paginate backwards through channel history, collecting every message
        posted on or after `since`.

        Discord returns pages newest-first when using `before=`. We walk back
        until we hit a message older than the cutoff, then stop.
        """
        cutoff_snowflake = snowflake_from_datetime(since)
        collected: list[DiscordMessage] = []
        before_id: str | None = None

        while True:
            page = await self._get_messages_page(channel_id, before=before_id)
            if not page:
                break

            reached_cutoff = False
            for raw in page:
                if int(raw["id"]) <= cutoff_snowflake:
                    reached_cutoff = True
                    break
                collected.append(DiscordMessage.from_api(raw, channel_id))

            if progress_callback:
                progress_callback(len(collected))

            if reached_cutoff:
                break

            # Use the oldest message in this page as cursor for the next request
            before_id = page[-1]["id"]
            await asyncio.sleep(0.5)  # conservative rate-limit buffer

        return collected

    # ── Incremental fetch (ongoing) ────────────────────────────────────────

    async def get_messages_after(
        self,
        channel_id: str,
        after_id: str,
    ) -> list[DiscordMessage]:
        """
        Fetch all messages posted after `after_id`, paginating forward.

        Discord returns pages oldest-first when using `after=`, so we extend
        `collected` in chronological order and advance the cursor forward.
        """
        collected: list[DiscordMessage] = []
        current_after = after_id

        while True:
            page = await self._get_messages_page(channel_id, after=current_after)
            if not page:
                break

            for raw in page:
                collected.append(DiscordMessage.from_api(raw, channel_id))

            current_after = page[-1]["id"]
            await asyncio.sleep(0.5)

        return collected
