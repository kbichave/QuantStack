# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
NLP sentiment analysis tool — multi-dimensional financial text sentiment.

Supports two backends:
  - groq  : Uses Groq LLM (llama-3.3-70b) via LiteLLM for nuanced,
             multi-dimensional sentiment with guidance/tone extraction.
  - finbert: Uses HuggingFace FinBERT (ProsusAI/finbert) for fast local
             inference. Requires `pip install transformers torch`.

Design invariants:
  - Never raises on LLM/model failure — returns error dict with context.
  - Groq method uses structured JSON prompting with temperature=0.
  - FinBERT method maps ProsusAI labels to the same output schema.
  - Text is truncated to 2000 chars to control token cost and latency.
"""

import json
from typing import Any

import litellm
from loguru import logger

from quantstack.llm_config import get_llm_for_role
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain

_MAX_TEXT_CHARS = 2000
_GROQ_MODEL_DEFAULT = "groq/llama-3.3-70b-versatile"


def _resolve_sentiment_model() -> str:
    """Resolve the LLM model string for sentiment analysis.

    Tries get_llm_for_role("ic") first (cheapest tier), falls back to the
    hardcoded Groq model if the LLM config system is unavailable.
    """
    try:
        return get_llm_for_role("ic")
    except Exception as exc:
        logger.debug(f"[nlp] get_llm_for_role('ic') failed, falling back to default: {exc}")
        return _GROQ_MODEL_DEFAULT


def _groq_sentiment(text: str) -> dict[str, Any]:
    """Run multi-dimensional sentiment analysis via Groq/LiteLLM."""
    truncated = text[:_MAX_TEXT_CHARS]
    model = _resolve_sentiment_model()

    system_prompt = (
        "You are a financial sentiment analyst. Analyze the following text and "
        "return ONLY a JSON object with these exact fields:\n"
        "{\n"
        '  "sentiment": "bullish" | "bearish" | "neutral",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "dimensions": {\n'
        '    "revenue_outlook": "positive" | "negative" | "neutral",\n'
        '    "guidance": "raised" | "lowered" | "maintained" | "n/a",\n'
        '    "management_tone": "confident" | "cautious" | "defensive"\n'
        "  },\n"
        '  "key_phrases": [<up to 5 short phrases that drove the sentiment>]\n'
        "}\n"
        "No markdown, no explanation, no text outside the JSON object."
    )

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": truncated},
        ],
        max_tokens=256,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    return _parse_groq_response(raw)


def _parse_groq_response(raw: str) -> dict[str, Any]:
    """Parse Groq JSON response into the canonical sentiment schema."""
    cleaned = raw.strip()
    # Strip markdown code fences if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    data = json.loads(cleaned)

    sentiment = data.get("sentiment", "neutral")
    if sentiment not in ("bullish", "bearish", "neutral"):
        sentiment = "neutral"

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    dimensions = data.get("dimensions", {})
    # Validate dimension values.
    rev = dimensions.get("revenue_outlook", "neutral")
    if rev not in ("positive", "negative", "neutral"):
        rev = "neutral"
    guidance = dimensions.get("guidance", "n/a")
    if guidance not in ("raised", "lowered", "maintained", "n/a"):
        guidance = "n/a"
    tone = dimensions.get("management_tone", "cautious")
    if tone not in ("confident", "cautious", "defensive"):
        tone = "cautious"

    key_phrases = data.get("key_phrases", [])
    if not isinstance(key_phrases, list):
        key_phrases = []
    key_phrases = [str(p) for p in key_phrases[:5]]

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "dimensions": {
            "revenue_outlook": rev,
            "guidance": guidance,
            "management_tone": tone,
        },
        "key_phrases": key_phrases,
        "method": "groq",
    }


def _finbert_sentiment(text: str) -> dict[str, Any]:
    """Run sentiment analysis via HuggingFace FinBERT."""
    truncated = text[:_MAX_TEXT_CHARS]
    from transformers import pipeline as _transformers_pipeline  # noqa: PLC0415
    classifier = _transformers_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        top_k=3,
    )
    results = classifier(truncated)

    # FinBERT returns list of dicts: [{"label": "positive", "score": 0.85}, ...]
    # Map to our schema.
    label_map = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}

    top = results[0] if results else {"label": "neutral", "score": 0.5}
    sentiment = label_map.get(top["label"], "neutral")
    confidence = round(float(top["score"]), 3)

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "dimensions": {
            "revenue_outlook": "neutral",
            "guidance": "n/a",
            "management_tone": "cautious",
        },
        "key_phrases": [],
        "method": "finbert",
    }


@domain(Domain.INTEL)
@tool_def()
async def analyze_text_sentiment(
    text: str,
    method: str = "groq",
) -> dict[str, Any]:
    """
    Analyze financial text sentiment with nuance beyond headline scoring.

    Groq method: Uses Groq LLM for nuanced multi-dimensional sentiment
    including revenue outlook, guidance direction, and management tone.
    FinBERT method: Uses HuggingFace FinBERT for fast local inference
    (requires `transformers` library).

    Args:
        text: Financial text to analyze (earnings transcript, press release,
              analyst note, SEC filing excerpt, etc.). Truncated to 2000 chars.
        method: "groq" (default, richer output) or "finbert" (local, faster)

    Returns:
        Dictionary with sentiment analysis:
            sentiment: "bullish" | "bearish" | "neutral"
            confidence: 0.0-1.0
            dimensions: {revenue_outlook, guidance, management_tone}
            key_phrases: list of up to 5 key phrases
            method: "groq" | "finbert"
    """
    if not text or not text.strip():
        return {
            "error": "Empty text provided",
            "sentiment": "neutral",
            "confidence": 0.0,
            "method": method,
        }

    if method not in ("groq", "finbert"):
        return {
            "error": f"Unknown method '{method}'. Use 'groq' or 'finbert'.",
            "method": method,
        }

    try:
        if method == "finbert":
            return _finbert_sentiment(text)
        return _groq_sentiment(text)
    except json.JSONDecodeError as exc:
        logger.warning(f"[nlp] JSON parse failure in {method} sentiment: {exc}")
        return {
            "error": f"Failed to parse {method} response as JSON",
            "sentiment": "neutral",
            "confidence": 0.0,
            "method": method,
        }
    except Exception as exc:
        logger.warning(f"[nlp] {method} sentiment analysis failed: {exc}")
        return {
            "error": str(exc),
            "sentiment": "neutral",
            "confidence": 0.0,
            "method": method,
        }


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
