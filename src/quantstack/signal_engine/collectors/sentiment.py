# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
SentimentCollector — news-scored linguistic signal via Groq.

Scores recent news headlines for a symbol using Groq's fast inference.
Designed to fail safely: if Groq is unavailable, no headlines exist,
or the call times out, returns safe defaults (sentiment_score=0.5, neutral).

The sentiment_score is orthogonal to price-based signals — it adds a
linguistic alpha dimension that the technical/regime collectors cannot provide.

Design invariants:
- Never raises. All errors are caught and logged at debug level.
- If no headlines: returns safe defaults immediately, no LLM call.
- LLM call capped at 5 headlines × 120 chars to control token cost.
- Uses groq/llama-3.3-70b-versatile via LiteLLM — same model as GroqPM.
"""

import asyncio
import json
from typing import Any

import os

from loguru import logger

from quantstack.data.adapters.financial_datasets_client import (
    FinancialDatasetsClient,
)

import litellm

from quantstack.llm_config import get_llm_for_role


_SENTIMENT_TIMEOUT = 8.0  # seconds — lower than other collectors (network + LLM)
_MAX_HEADLINES = 5
_MAX_HEADLINE_CHARS = 120
_NEWS_LOOKBACK_DAYS = 2


async def collect_sentiment(symbol: str, _store: Any) -> dict[str, Any]:
    """
    Score recent headlines for *symbol* using Groq.

    Returns a dict with keys:
        sentiment_score     : float in [0.0, 1.0] — 0=bearish, 0.5=neutral, 1=bullish
        dominant_sentiment  : "positive" | "negative" | "neutral"
        n_headlines         : int — number of headlines scored
        source              : "groq" | "no_headlines" | "default"
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_sentiment_sync, symbol),
            timeout=_SENTIMENT_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.debug(
            f"[sentiment] {symbol}: {type(exc).__name__} — returning safe defaults"
        )
        return _safe_defaults()


def _collect_sentiment_sync(symbol: str) -> dict[str, Any]:
    """Synchronous sentiment collection — called via asyncio.to_thread."""
    headlines = _fetch_headlines(symbol)
    if not headlines:
        return {**_safe_defaults(), "source": "no_headlines"}

    truncated = [h[:_MAX_HEADLINE_CHARS] for h in headlines[:_MAX_HEADLINES]]
    prompt = _build_prompt(symbol, truncated)

    try:
        _model = get_llm_for_role("bulk")
        response = litellm.completion(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_response(raw, len(truncated))
    except Exception as exc:
        logger.debug(f"[sentiment] {symbol}: LiteLLM call failed: {exc}")
        return _safe_defaults()


def _fetch_headlines(symbol: str) -> list[str]:
    """Fetch recent news headlines via FinancialDatasets.ai.

    Falls back to empty list if the API key is missing or the call fails.
    """
    try:
        api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
        if not api_key:
            logger.debug(
                "[sentiment] FINANCIAL_DATASETS_API_KEY not set, skipping news fetch"
            )
            return []

        with FinancialDatasetsClient(api_key=api_key) as client:
            resp = client.get_company_news(symbol, limit=_MAX_HEADLINES)

        news_items = (resp or {}).get("news", [])
        return [
            item.get("title", item.get("headline", ""))
            for item in news_items
            if item.get("title") or item.get("headline")
        ]
    except Exception as exc:
        logger.debug(f"[sentiment] headline fetch failed for {symbol}: {exc}")
        return []


def _build_prompt(symbol: str, headlines: list[str]) -> str:
    headlines_text = "\n".join(f"- {h}" for h in headlines)
    return (
        f"Rate the market sentiment of these {symbol} news headlines.\n"
        f"Headlines:\n{headlines_text}\n\n"
        "Reply with ONLY a JSON object: "
        '{"sentiment_score": <0.0-1.0>, "dominant_sentiment": "<positive|negative|neutral>"}\n'
        "Where 0.0=very bearish, 0.5=neutral, 1.0=very bullish. No other text."
    )


def _parse_response(raw: str, n_headlines: int) -> dict[str, Any]:
    """Parse Groq JSON response. Returns safe defaults on parse failure."""
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        data = json.loads(cleaned)
        score = float(data.get("sentiment_score", 0.5))
        score = max(0.0, min(1.0, score))  # clamp to valid range
        dominant = data.get("dominant_sentiment", "neutral")
        if dominant not in ("positive", "negative", "neutral"):
            dominant = "neutral"

        return {
            "sentiment_score": round(score, 3),
            "dominant_sentiment": dominant,
            "n_headlines": n_headlines,
            "source": "groq",
        }
    except Exception as exc:
        logger.debug(f"[sentiment] response parse failed: {exc} — raw: {raw[:100]}")
        return _safe_defaults()


def _safe_defaults() -> dict[str, Any]:
    return {
        "sentiment_score": 0.5,
        "dominant_sentiment": "neutral",
        "n_headlines": 0,
        "source": "default",
    }
