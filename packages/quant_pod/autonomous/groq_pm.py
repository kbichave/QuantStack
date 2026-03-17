# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
GroqPM — single Groq call for non-routine PM decisions.

Called by AutonomousRunner when DecisionRouter returns GROQ_SYNTHESIS.
Structured prompt → structured JSON output → validated PMDecision.

Failure modes:
- Timeout (15 sec): returns None → caller falls back to SKIP.
- Malformed output (not valid JSON, missing required fields): returns None → SKIP.
- Groq unavailable: returns None → SKIP.

Never execute on ambiguous Groq output. SKIP is always safer than a bad trade.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from loguru import logger


_TIMEOUT_SECONDS = 15.0
_SYSTEM_PROMPT = """\
You are an autonomous trading PM making a single trade decision.
You must return ONLY valid JSON — no prose, no markdown.

Constraints (hard rules, never override):
- paper_mode is always True unless explicitly told otherwise
- position_size must be "full", "half", or "quarter"
- action must be "buy", "sell", or "skip"
- confidence must be a float in [0.0, 1.0]

Return exactly this JSON structure:
{
  "action": "buy" | "sell" | "skip",
  "confidence": float,
  "position_size": "full" | "half" | "quarter",
  "reasoning": "one sentence explaining the decision",
  "key_risk": "one sentence describing the main risk"
}
"""


@dataclass
class PMDecision:
    action: str          # "buy", "sell", or "skip"
    confidence: float    # [0, 1]
    position_size: str   # "full", "half", "quarter"
    reasoning: str
    key_risk: str


class GroqPM:
    """
    Single-shot Groq call for non-routine trading decisions.

    Uses get_llm_for_role("autonomous_pm") which resolves to:
    - LLM_MODEL_AUTONOMOUS_PM env var if set
    - groq/llama-3.3-70b-versatile if LLM_PROVIDER=groq (or GROQ_API_KEY present)
    - Primary LLM_PROVIDER fallback (Bedrock, Anthropic, etc.)
    """

    async def synthesize(
        self,
        symbol: str,
        brief: Any,                  # SignalBrief
        exception_reason: str,
        portfolio: dict,
        strategies: list[dict],
    ) -> PMDecision | None:
        """
        Ask the Groq PM for a trade decision given an exception condition.

        Returns PMDecision on success, None on timeout/parse error/LLM unavailability.
        """
        try:
            return await asyncio.wait_for(
                self._call(symbol, brief, exception_reason, portfolio, strategies),
                timeout=_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[GroqPM] {symbol}: timeout ({_TIMEOUT_SECONDS}s) — SKIP")
            return None
        except Exception as exc:
            logger.warning(f"[GroqPM] {symbol}: error — {exc} — SKIP")
            return None

    async def _call(
        self,
        symbol: str,
        brief: Any,
        exception_reason: str,
        portfolio: dict,
        strategies: list[dict],
    ) -> PMDecision | None:
        prompt = _build_prompt(symbol, brief, exception_reason, portfolio, strategies)

        try:
            from quant_pod.llm_config import get_llm_for_role
            llm_model = get_llm_for_role("autonomous_pm")
        except Exception as exc:
            logger.warning(f"[GroqPM] LLM resolution failed: {exc}")
            return None

        raw_response = await asyncio.to_thread(
            _invoke_llm, llm_model, _SYSTEM_PROMPT, prompt
        )
        if raw_response is None:
            return None

        return _parse_response(raw_response)


def _build_prompt(
    symbol: str,
    brief: Any,
    exception_reason: str,
    portfolio: dict,
    strategies: list[dict],
) -> str:
    """Build the structured user prompt."""
    strategy_names = [s.get("name", s.get("strategy_id", "unknown")) for s in strategies[:3]]

    # Summarize the brief concisely — don't dump the full dict (token waste)
    brief_summary = {
        "symbol": symbol,
        "market_bias": getattr(brief, "market_bias", "neutral"),
        "conviction": getattr(brief, "market_conviction", 0.5),
        "risk_environment": getattr(brief, "risk_environment", "normal"),
        "regime": getattr(brief, "regime_detail", {}) or {},
        "collector_failures": getattr(brief, "collector_failures", []),
    }
    if hasattr(brief, "symbol_briefs") and brief.symbol_briefs:
        sb = brief.symbol_briefs[0]
        brief_summary["key_observations"] = getattr(sb, "key_observations", [])[:3]
        brief_summary["risk_factors"] = getattr(sb, "risk_factors", [])[:2]

    portfolio_summary = {
        "cash": portfolio.get("cash", 0),
        "equity": portfolio.get("equity", 0),
        "open_position": symbol in portfolio.get("positions", {}),
    }

    return (
        f"Symbol: {symbol}\n"
        f"Exception: {exception_reason}\n"
        f"Active strategies: {strategy_names}\n"
        f"Signal brief: {json.dumps(brief_summary, default=str)}\n"
        f"Portfolio: {json.dumps(portfolio_summary, default=str)}\n"
        f"\nDecide: buy, sell, or skip for {symbol}. Return JSON only."
    )


def _invoke_llm(llm_model: Any, system: str, user: str) -> str | None:
    """Synchronous LLM call via LiteLLM. Returns raw text response or None."""
    try:
        import litellm

        model_str = llm_model if isinstance(llm_model, str) else getattr(llm_model, "model", str(llm_model))
        response = litellm.completion(
            model=model_str,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,   # low temperature for consistent structured output
            max_tokens=256,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.warning(f"[GroqPM] LiteLLM call failed: {exc}")
        return None


def _parse_response(raw: str) -> PMDecision | None:
    """Parse and validate Groq response. Returns None on any parse failure."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.startswith("```") and not line.startswith("json")
        )

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try to extract JSON from mixed text
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.warning("[GroqPM] response is not valid JSON — SKIP")
            return None
        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            logger.warning("[GroqPM] could not extract JSON — SKIP")
            return None

    # Validate required fields and types
    action = str(data.get("action", "")).lower()
    if action not in ("buy", "sell", "skip"):
        logger.warning(f"[GroqPM] invalid action '{action}' — SKIP")
        return None

    try:
        confidence = float(data["confidence"])
        if not (0.0 <= confidence <= 1.0):
            confidence = max(0.0, min(1.0, confidence))
    except (KeyError, TypeError, ValueError):
        confidence = 0.5

    position_size = str(data.get("position_size", "quarter")).lower()
    if position_size not in ("full", "half", "quarter"):
        position_size = "quarter"  # safe default

    return PMDecision(
        action=action,
        confidence=confidence,
        position_size=position_size,
        reasoning=str(data.get("reasoning", "Groq PM decision"))[:500],
        key_risk=str(data.get("key_risk", "unknown"))[:500],
    )
