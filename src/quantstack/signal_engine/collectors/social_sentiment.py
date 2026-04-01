# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Social sentiment collector — 16th signal collector.

Sources: Reddit public JSON API + Stocktwits public stream API.
No API keys required. Fails gracefully to neutral defaults.

Design: lightweight HTTP-only, no LLM. Runs in <8s (3s per source + overhead).
Used for per-symbol community buzz scoring during trading hours.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantstack.data.storage import DataStore

logger = logging.getLogger(__name__)

# Keywords for simple title-based sentiment scoring (no LLM needed)
_BULLISH_WORDS = frozenset(
    ["buy", "calls", "moon", "squeeze", "breakout", "bull", "long", "up", "rally",
     "puts", "beat", "upgrade", "strong", "hold", "accumulate", "entry"]
)
# Note: "puts" is bearish in options context but used bullishly sometimes — included conservatively.
_BEARISH_WORDS = frozenset(
    ["puts", "short", "crash", "bear", "sell", "down", "dump", "drop", "miss",
     "downgrade", "weak", "avoid", "exit", "overvalued", "warning", "halt"]
)

# Buzz score denominator — 20 mentions in a day = 1.0 buzz score
_BUZZ_NORMALIZER = 20.0

_SOURCE_TIMEOUT = 3.0  # seconds per source

_DEFAULT_RESULT = {
    "reddit_mention_count": 0,
    "reddit_sentiment": "neutral",
    "stocktwits_bullish_pct": None,
    "social_buzz_score": 0.5,
    "source": "default",
}


async def collect_social_sentiment(symbol: str, _store: "DataStore") -> dict:
    """
    Collect community sentiment for a symbol from Reddit + Stocktwits.

    Returns a dict with keys:
        reddit_mention_count    int   — posts mentioning symbol in last 24h
        reddit_sentiment        str   — "bullish" | "bearish" | "neutral"
        stocktwits_bullish_pct  float | None — fraction of bullish messages
        social_buzz_score       float — 0.0–1.0 (0.5 = neutral/default)
        source                  str   — "reddit+stocktwits" | "reddit_only" |
                                        "stocktwits_only" | "default"
    """
    try:
        reddit_task = asyncio.create_task(_fetch_reddit(symbol))
        stocktwits_task = asyncio.create_task(_fetch_stocktwits(symbol))

        reddit_data, stocktwits_data = await asyncio.gather(
            reddit_task, stocktwits_task, return_exceptions=True
        )

        if isinstance(reddit_data, Exception):
            logger.debug(f"[social_sentiment] Reddit failed for {symbol}: {reddit_data}")
            reddit_data = None
        if isinstance(stocktwits_data, Exception):
            logger.debug(f"[social_sentiment] Stocktwits failed for {symbol}: {stocktwits_data}")
            stocktwits_data = None

        # Score what we have
        reddit_count = 0
        reddit_sentiment = "neutral"
        stocktwits_bullish_pct = None

        if reddit_data:
            reddit_count, reddit_sentiment = _score_reddit(reddit_data)
        if stocktwits_data:
            stocktwits_bullish_pct = _score_stocktwits(stocktwits_data)

        # Combine into a buzz score
        # Base from reddit mentions (0–1), adjusted by stocktwits if available
        buzz = min(1.0, reddit_count / _BUZZ_NORMALIZER)
        if stocktwits_bullish_pct is not None:
            # Skew buzz slightly toward 0.5 baseline when only stocktwits is available
            if reddit_count == 0:
                buzz = 0.5

        # Resolve source
        has_reddit = reddit_data is not None
        has_stocktwits = stocktwits_data is not None
        if has_reddit and has_stocktwits:
            source = "reddit+stocktwits"
        elif has_reddit:
            source = "reddit_only"
        elif has_stocktwits:
            source = "stocktwits_only"
        else:
            return dict(_DEFAULT_RESULT)

        return {
            "reddit_mention_count": reddit_count,
            "reddit_sentiment": reddit_sentiment,
            "stocktwits_bullish_pct": stocktwits_bullish_pct,
            "social_buzz_score": round(buzz, 3),
            "source": source,
        }

    except Exception as exc:
        logger.debug(f"[social_sentiment] Unexpected failure for {symbol}: {exc}")
        return dict(_DEFAULT_RESULT)


async def _fetch_reddit(symbol: str) -> list[str]:
    """
    Fetch recent Reddit posts mentioning symbol.

    Uses the public JSON API — no auth needed.
    Returns a list of post titles from the last 24h.
    """
    import aiohttp

    url = (
        f"https://www.reddit.com/search.json"
        f"?q={symbol}&sort=new&t=day&limit=25"
    )
    headers = {"User-Agent": "QuantStack/1.0 (autonomous trading system)"}

    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=_SOURCE_TIMEOUT)
        ) as resp:
            if resp.status != 200:
                logger.debug(f"[social_sentiment] Reddit HTTP {resp.status} for {symbol}")
                return []
            data = await resp.json(content_type=None)
            posts = data.get("data", {}).get("children", [])
            return [
                p.get("data", {}).get("title", "")
                for p in posts
                if p.get("data", {}).get("title")
            ]


async def _fetch_stocktwits(symbol: str) -> dict:
    """
    Fetch Stocktwits message stream for symbol.

    Public endpoint — no auth needed for read-only stream.
    Returns the raw JSON response dict.
    """
    import aiohttp

    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=_SOURCE_TIMEOUT)
        ) as resp:
            if resp.status != 200:
                logger.debug(f"[social_sentiment] Stocktwits HTTP {resp.status} for {symbol}")
                return {}
            return await resp.json(content_type=None)


def _score_reddit(titles: list[str]) -> tuple[int, str]:
    """
    Score a list of Reddit post titles.

    Returns (mention_count, sentiment_label).
    sentiment_label: "bullish" | "bearish" | "neutral"
    """
    if not titles:
        return 0, "neutral"

    mention_count = len(titles)
    bullish = 0
    bearish = 0

    for title in titles:
        words = title.lower().split()
        word_set = set(words)
        if word_set & _BULLISH_WORDS:
            bullish += 1
        if word_set & _BEARISH_WORDS:
            bearish += 1

    if bullish > bearish * 1.5:
        sentiment = "bullish"
    elif bearish > bullish * 1.5:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return mention_count, sentiment


def _score_stocktwits(data: dict) -> float | None:
    """
    Extract bullish percentage from Stocktwits response.

    Returns fraction of bullish messages (0.0–1.0) or None if unavailable.
    """
    messages = data.get("messages", [])
    if not messages:
        return None

    bullish_count = 0
    bearish_count = 0
    for msg in messages:
        sentiment = (msg.get("entities", {}) or {}).get("sentiment")
        if sentiment is None:
            continue
        label = (sentiment.get("basic") or "").lower()
        if label == "bullish":
            bullish_count += 1
        elif label == "bearish":
            bearish_count += 1

    total = bullish_count + bearish_count
    if total == 0:
        return None
    return round(bullish_count / total, 3)
