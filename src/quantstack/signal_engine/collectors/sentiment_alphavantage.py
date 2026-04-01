# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
AlphaVantageNewsCollector — contextualized news reasoning via Groq.

Fetches actual headlines from Alpha Vantage news_sentiment table,
enriches with technical/fundamental/macro context, and sends to Groq
for nuanced sentiment reasoning.

Design:
- Headlines + context → Groq reasons about implications
- Considers: price action, earnings, sector, macro regime, institutional flow
- Returns: sentiment_score, reasoning, confidence
- Fails gracefully: no context → safe defaults immediately

Replaces simple sentiment scoring with reasoning that connects dots.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

from loguru import logger
import litellm

from quantstack.data.storage import DataStore
from quantstack.llm_config import get_llm_for_role


_SENTIMENT_TIMEOUT = 10.0  # Slightly longer for context gathering
_MAX_HEADLINES = 7
_MAX_HEADLINE_CHARS = 150
_NEWS_LOOKBACK_DAYS = 7


async def collect_sentiment_alphavantage(
    symbol: str, store: DataStore
) -> dict[str, Any]:
    """
    Fetch Alpha Vantage headlines + context, reason with Groq.

    Returns dict with:
        sentiment_score     : float [0.0-1.0]
        dominant_sentiment  : "bullish" | "neutral" | "bearish"
        n_headlines         : int
        reasoning           : str (short explanation)
        confidence          : float [0.0-1.0]
        context_used        : list[str] (which signals informed decision)
        source              : "alphavantage_groq" | "alphavantage_prescore" | "default"
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _collect_sentiment_alphavantage_sync, symbol, store
            ),
            timeout=_SENTIMENT_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.debug(
            f"[sentiment_av] {symbol}: {type(exc).__name__} — returning safe defaults"
        )
        return _safe_defaults()


def _collect_sentiment_alphavantage_sync(
    symbol: str, store: DataStore
) -> dict[str, Any]:
    """Synchronous sentiment collection with context."""

    # Fetch headlines from Alpha Vantage news_sentiment table
    headlines_data = _fetch_alphavantage_headlines(symbol, store)
    if not headlines_data:
        return {**_safe_defaults(), "source": "no_headlines"}

    headlines, raw_scores = headlines_data

    # Gather context from technicals, fundamentals, macro, flow
    context = _gather_context(symbol, store)

    # Build rich reasoning prompt
    prompt = _build_reasoning_prompt(symbol, headlines, context, raw_scores)

    # Send to bulk-tier LLM for reasoning (Groq preferred; configurable via LLM_MODEL_BULK).
    try:
        _model = get_llm_for_role("bulk")
        response = litellm.completion(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,  # Low temp for consistency
        )
        raw = response.choices[0].message.content.strip()
        return _parse_reasoning_response(raw, len(headlines), context)
    except Exception as exc:
        logger.debug(
            f"[sentiment_av] {symbol}: Groq reasoning failed: {exc} — "
            f"falling back to pre-scored sentiment"
        )
        # Fallback: use Alpha Vantage pre-scored sentiment
        return _prescore_fallback(symbol, raw_scores, headlines)


def _fetch_alphavantage_headlines(
    symbol: str, store: DataStore
) -> tuple[list[str], list[float]] | None:
    """
    Fetch recent headlines from Alpha Vantage news_sentiment table.

    Returns: (headlines, raw_sentiment_scores) or None if no data
    """
    try:
        cutoff_date = (
            datetime.now() - timedelta(days=_NEWS_LOOKBACK_DAYS)
        ).isoformat()

        with store._use_conn() as conn:
            rows = conn.execute(
                """
                SELECT title, ticker_sentiment_score
                FROM news_sentiment
                WHERE ticker = ? AND time_published >= ?
                ORDER BY time_published DESC
                LIMIT ?
            """,
                [symbol, cutoff_date, _MAX_HEADLINES],
            ).fetchall()

        if not rows:
            return None

        headlines = [
            row[0][:_MAX_HEADLINE_CHARS] for row in rows if row[0]
        ]
        scores = [float(row[1]) if row[1] is not None else 0.0 for row in rows]

        return (headlines, scores) if headlines else None
    except Exception as exc:
        logger.debug(f"[sentiment_av] headline fetch failed for {symbol}: {exc}")
        return None


def _gather_context(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Gather technical, fundamental, macro context from database.

    Returns dict with relevant signals for reasoning.
    """
    context = {
        "technical": _get_technical_context(symbol, store),
        "fundamental": _get_fundamental_context(symbol, store),
        "macro": _get_macro_context(store),
        "flow": _get_flow_context(symbol, store),
    }
    return context


def _get_technical_context(symbol: str, store: DataStore) -> dict[str, Any]:
    """Extract recent price action, trend, volatility."""
    try:
        with store._use_conn() as conn:
            # Get last 20 daily bars
            bars = conn.execute(
                """
                SELECT timestamp, close, volume
                FROM ohlcv
                WHERE symbol = ? AND timeframe = 'D1'
                ORDER BY timestamp DESC
                LIMIT 20
            """,
                [symbol],
            ).fetchall()

        if not bars:
            return {}

        bars = list(reversed(bars))  # Chronological order
        closes = [b[1] for b in bars]
        recent_price = closes[-1]
        prior_price = closes[0]

        # Simple trend metrics
        price_change_pct = ((recent_price - prior_price) / prior_price) * 100
        highest = max(closes)
        lowest = min(closes)
        volatility = ((highest - lowest) / prior_price) * 100

        return {
            "price_change_20d_pct": round(price_change_pct, 2),
            "volatility_pct": round(volatility, 2),
            "recent_price": round(recent_price, 2),
            "trend": "uptrend" if price_change_pct > 2 else "downtrend" if price_change_pct < -2 else "sideways",
        }
    except Exception as exc:
        logger.debug(f"[sentiment_av] technical context failed: {exc}")
        return {}


def _get_fundamental_context(symbol: str, store: DataStore) -> dict[str, Any]:
    """Extract latest fundamental metrics."""
    try:
        with store._use_conn() as conn:
            overview = conn.execute(
                """
                SELECT market_cap, dividend_yield, beta, sector
                FROM company_overview
                WHERE symbol = ?
            """,
                [symbol],
            ).fetchone()

        if not overview:
            return {}

        market_cap, div_yield, beta, sector = overview
        return {
            "sector": sector,
            "market_cap": market_cap,
            "dividend_yield": div_yield,
            "beta": round(beta, 2) if beta else None,
        }
    except Exception as exc:
        logger.debug(f"[sentiment_av] fundamental context failed: {exc}")
        return {}


def _get_macro_context(store: DataStore) -> dict[str, Any]:
    """Extract macro regime context."""
    try:
        with store._use_conn() as conn:
            # VIX proxy (or use market_indicators if available)
            vix = conn.execute(
                """
                SELECT value FROM macro_indicators
                WHERE indicator LIKE '%VIX%' OR indicator = 'VOLATILITY_INDEX'
                ORDER BY date DESC LIMIT 1
            """
            ).fetchone()

        return {
            "vix_approx": round(vix[0], 1) if vix and vix[0] else 18.0,
            "risk_environment": "high" if vix and vix[0] > 25 else "normal" if vix and vix[0] > 15 else "low",
        }
    except Exception as exc:
        logger.debug(f"[sentiment_av] macro context failed: {exc}")
        return {}


def _get_flow_context(symbol: str, store: DataStore) -> dict[str, Any]:
    """Extract institutional/insider flow context."""
    try:
        with store._use_conn() as conn:
            # Recent insider activity
            insider = conn.execute(
                """
                SELECT transaction_type, COUNT(*) as count
                FROM insider_trades
                WHERE ticker = ? AND (julianday('now') - julianday(transaction_date)) < 30
                GROUP BY transaction_type
            """,
                [symbol],
            ).fetchall()

        buys = sum(c for t, c in insider if "buy" in str(t).lower())
        sells = sum(c for t, c in insider if "sell" in str(t).lower())

        return {
            "insider_buys_30d": buys,
            "insider_sells_30d": sells,
            "insider_signal": "bullish" if buys > sells else "bearish" if sells > buys else "neutral",
        }
    except Exception as exc:
        logger.debug(f"[sentiment_av] flow context failed: {exc}")
        return {}


def _build_reasoning_prompt(
    symbol: str,
    headlines: list[str],
    context: dict[str, Any],
    raw_scores: list[float],
) -> str:
    """
    Build rich reasoning prompt with headlines + context.

    Asks Groq to reason about what the news means given the current state.
    """

    # Format headlines with their pre-scored sentiment
    headlines_text = "\n".join(
        f"  • {h} [AV score: {s:.2f}]"
        for h, s in zip(headlines, raw_scores)
    )

    # Format context
    tech = context.get("technical", {})
    fund = context.get("fundamental", {})
    macro = context.get("macro", {})
    flow = context.get("flow", {})

    context_text = f"""
CURRENT CONTEXT FOR {symbol}:

Technical:
  - Trend: {tech.get('trend', 'unknown')} (20-day: {tech.get('price_change_20d_pct', 'N/A')}%)
  - Volatility: {tech.get('volatility_pct', 'N/A')}%

Fundamental:
  - Sector: {fund.get('sector', 'N/A')}
  - Beta: {fund.get('beta', 'N/A')} (systematic risk)

Macro:
  - Market Risk: {macro.get('risk_environment', 'unknown')} (VIX ~{macro.get('vix_approx', 18)})

Flow:
  - Insider: {flow.get('insider_signal', 'neutral')} ({flow.get('insider_buys_30d', 0)} buys vs {flow.get('insider_sells_30d', 0)} sells in 30d)
"""

    prompt = f"""You are a financial analyst. Reason about the market sentiment for {symbol}.

RECENT NEWS HEADLINES ({len(headlines)} articles, last 7 days):
{headlines_text}

{context_text}

TASK:
1. Read all headlines carefully
2. Consider what each headline implies for the stock
3. Factor in the current technical trend, sector dynamics, macro environment, and insider activity
4. Synthesize: do the headlines align with or contradict current technical/fundamental state?
5. Return a SINGLE JSON object (no markdown, no code blocks):

{{"sentiment_score": <0.0-1.0 float>, "dominant_sentiment": "<bullish|neutral|bearish>", "reasoning": "<1-2 sentence explanation of why>", "confidence": <0.0-1.0 float>}}

Where:
- sentiment_score: 0.0=very bearish, 0.5=neutral, 1.0=very bullish
- confidence: how certain you are (high if many headlines agree, low if mixed)
- reasoning: brief explanation connecting headlines to technical/fundamental context

IMPORTANT: Return ONLY the JSON object. No other text."""

    return prompt


def _parse_reasoning_response(
    raw: str, n_headlines: int, context: dict[str, Any]
) -> dict[str, Any]:
    """Parse Groq reasoning response."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        data = json.loads(cleaned)

        score = float(data.get("sentiment_score", 0.5))
        score = max(0.0, min(1.0, score))

        sentiment = data.get("dominant_sentiment", "neutral")
        if sentiment not in ("bullish", "bearish", "neutral"):
            sentiment = "neutral"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(data.get("reasoning", ""))[:200]

        # Track which signals informed the decision
        context_used = []
        if context.get("technical"):
            context_used.append("technical")
        if context.get("fundamental"):
            context_used.append("fundamental")
        if context.get("macro"):
            context_used.append("macro")
        if context.get("flow"):
            context_used.append("flow")

        return {
            "sentiment_score": round(score, 3),
            "dominant_sentiment": sentiment,
            "n_headlines": n_headlines,
            "reasoning": reasoning,
            "confidence": round(confidence, 3),
            "context_used": context_used,
            "source": "alphavantage_groq",
        }
    except Exception as exc:
        logger.debug(f"[sentiment_av] parse failed: {exc} — raw: {raw[:100]}")
        return _safe_defaults()


def _prescore_fallback(
    symbol: str, raw_scores: list[float], headlines: list[str]
) -> dict[str, Any]:
    """Fallback: use Alpha Vantage pre-scored sentiment (no Groq)."""
    if not raw_scores:
        return _safe_defaults()

    avg_score = sum(raw_scores) / len(raw_scores)
    # Convert AV score [-1, 1] to our scale [0, 1]
    normalized = (avg_score + 1.0) / 2.0
    normalized = max(0.0, min(1.0, normalized))

    dominant = (
        "bullish"
        if avg_score > 0.2
        else "bearish"
        if avg_score < -0.2
        else "neutral"
    )

    return {
        "sentiment_score": round(normalized, 3),
        "dominant_sentiment": dominant,
        "n_headlines": len(headlines),
        "reasoning": f"Avg Alpha Vantage score: {avg_score:.2f}",
        "confidence": 0.6,  # Lower confidence without reasoning
        "context_used": [],
        "source": "alphavantage_prescore",
    }


def _safe_defaults() -> dict[str, Any]:
    return {
        "sentiment_score": 0.5,
        "dominant_sentiment": "neutral",
        "n_headlines": 0,
        "reasoning": "",
        "confidence": 0.0,
        "context_used": [],
        "source": "default",
    }
