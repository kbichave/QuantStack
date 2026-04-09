# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
HypothesisAgent — novel strategy rule set generation via Groq.

Generates entry/exit rule combinations from the current market regime and
signal indicators, using a single Groq call with structured JSON output.
Runs during the nightly discovery loop as an additive source of candidates
(template-based discovery runs regardless, regardless of HypothesisAgent).

Design invariants:
- Returns [] on timeout, Groq failure, or JSON parse failure — never raises.
- All returned candidates pass schema validation before being returned.
  Malformed rules are dropped silently, not propagated to CandidateFilter.
- Uses groq/qwen/qwen3-32b via LiteLLM for hypothesis generation.
- Temperature=0 for deterministic structured output.
- max_hypotheses=5 caps token cost per call regardless of prompt response.

Usage:
    agent = HypothesisAgent()
    candidates = await agent.generate(
        symbol="XOM",
        regime={"trend_regime": "trending_up", "volatility_regime": "normal", "confidence": 0.8},
        signal_brief_summary={"rsi_14": 45.0, "adx_14": 28.0, ...},
    )
    # Each candidate: {entry_rules: [...], exit_rules: [...], parameters: {...}}
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import litellm
from loguru import logger

from quantstack.llm.provider import get_model_for_role


_TIMEOUT = 20.0  # seconds — generous for nightly batch
_MAX_HYPOTHESES = 5  # cap regardless of LLM response length

# Valid indicator names for schema validation
_VALID_INDICATORS = frozenset(
    {
        "rsi",
        "macd_hist",
        "adx_14",
        "bb_pct",
        "sma_fast",
        "sma_slow",
        "sma_crossover",
        "breakout",
    }
)
_VALID_CONDITIONS = frozenset(
    {
        "above",
        "below",
        "crosses_above",
        "crosses_below",
    }
)
_VALID_EXIT_TYPES = frozenset({"stop_loss", "take_profit"})


class HypothesisAgent:
    """
    Generate novel strategy rule sets from current market state.

    All methods are async. Falls back to [] on any failure.
    """

    async def generate(
        self,
        symbol: str,
        regime: dict[str, Any],
        signal_brief_summary: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Generate 3-5 novel entry/exit rule sets for this symbol + regime.

        Args:
            symbol: Ticker (e.g., "XOM").
            regime: Regime dict from WeeklyRegimeClassifier or regime collector.
                    Expected keys: trend_regime, volatility_regime, confidence.
            signal_brief_summary: Key indicators from technical collector.
                    Expected keys: rsi_14, adx_14, macd_hist, bb_pct, weekly_trend.

        Returns:
            List of dicts, each {entry_rules: [...], exit_rules: [...], parameters: {}}.
            Empty list if Groq is unavailable, times out, or returns no valid candidates.
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_sync, symbol, regime, signal_brief_summary
                ),
                timeout=_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.debug(
                f"[HypothesisAgent] {symbol}: timed out after {_TIMEOUT}s — no candidates"
            )
            return []
        except Exception as exc:
            logger.debug(f"[HypothesisAgent] {symbol}: unexpected error — {exc}")
            return []

    def _generate_sync(
        self,
        symbol: str,
        regime: dict[str, Any],
        brief: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Synchronous Groq call — runs in a thread."""
        prompt = _build_prompt(symbol, regime, brief)

        try:
            response = litellm.completion(
                model=get_model_for_role("bulk"),
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.debug(f"[HypothesisAgent] LiteLLM call failed: {exc}")
            return []

        return _parse_and_validate(raw)


# =============================================================================
# Prompt construction
# =============================================================================

_SYSTEM_PROMPT = (
    "You are a quantitative strategy researcher. Generate novel entry/exit rules "
    "as a JSON array. Each element must be: "
    '{"entry_rules": [...], "exit_rules": [...], "parameters": {}}. '
    "Rules must use: "
    "indicator (rsi|macd_hist|adx_14|bb_pct|sma_fast|sma_slow|sma_crossover|breakout), "
    "condition (above|below|crosses_above|crosses_below), value (float). "
    'Structural exits: {"type": "stop_loss", "atr_multiple": float} or '
    '{"type": "take_profit", "atr_multiple": float}. '
    f"Return {_MAX_HYPOTHESES} elements maximum. No prose, no markdown. JSON only."
)


def _build_prompt(
    symbol: str,
    regime: dict[str, Any],
    brief: dict[str, Any],
) -> str:
    trend = regime.get("trend_regime", "unknown")
    vol = regime.get("volatility_regime", "normal")
    conf = regime.get("confidence", 0.5)
    rsi = brief.get("rsi_14", "N/A")
    adx = brief.get("adx_14", "N/A")
    macd = brief.get("macd_hist", "N/A")
    bb = brief.get("bb_pct", "N/A")
    weekly = brief.get("weekly_trend", "unknown")

    return (
        f"Symbol: {symbol}\n"
        f"Regime: trend={trend} vol={vol} confidence={conf:.0%}\n"
        f"Signals: RSI={rsi} ADX={adx} MACD_hist={macd} BB_pct={bb} weekly_trend={weekly}\n"
        f"Generate 3-5 novel entry/exit rule combinations suited to this regime. "
        f"Avoid simple RSI threshold and SMA crossover variants — generate structurally distinct combinations."
    )


# =============================================================================
# Response parsing and validation
# =============================================================================


def _parse_and_validate(raw: str) -> list[dict[str, Any]]:
    """
    Extract and validate candidates from the Groq response.

    Strips markdown fences, parses JSON, validates each candidate against
    the rule schema. Returns only valid candidates, max _MAX_HYPOTHESES.
    """
    try:
        cleaned = raw.strip()
        # Strip ```json ... ``` or ``` ... ``` fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            inner = [l for l in lines if not l.startswith("```")]
            cleaned = "\n".join(inner)

        data = json.loads(cleaned)
        if not isinstance(data, list):
            logger.debug("[HypothesisAgent] response is not a JSON array")
            return []

        valid: list[dict[str, Any]] = []
        for item in data:
            if len(valid) >= _MAX_HYPOTHESES:
                break
            candidate = _validate_candidate(item)
            if candidate is not None:
                valid.append(candidate)

        logger.info(
            f"[HypothesisAgent] generated {len(valid)} valid candidates "
            f"from {len(data)} returned"
        )
        return valid

    except (json.JSONDecodeError, Exception) as exc:
        logger.debug(f"[HypothesisAgent] parse failed: {exc} — raw[:100]: {raw[:100]}")
        return []


def _validate_candidate(item: Any) -> dict[str, Any] | None:
    """
    Validate a single candidate dict against the rule schema.

    Returns the normalized candidate or None if invalid.
    Unknown indicator names or conditions are grounds for rejection
    to prevent CandidateFilter from receiving malformed specs.
    """
    if not isinstance(item, dict):
        return None

    entry_rules = item.get("entry_rules")
    exit_rules = item.get("exit_rules")
    parameters = item.get("parameters", {})

    if not isinstance(entry_rules, list) or len(entry_rules) == 0:
        return None
    if not isinstance(exit_rules, list) or len(exit_rules) == 0:
        return None
    if not isinstance(parameters, dict):
        parameters = {}

    validated_entry = [r for r in entry_rules if _valid_rule(r)]
    validated_exit = [r for r in exit_rules if _valid_rule(r, is_exit=True)]

    if not validated_entry or not validated_exit:
        return None

    return {
        "entry_rules": validated_entry,
        "exit_rules": validated_exit,
        "parameters": parameters,
    }


def _valid_rule(rule: Any, is_exit: bool = False) -> bool:
    """Return True if the rule dict matches the expected schema."""
    if not isinstance(rule, dict):
        return False

    # Structural exit rules (stop_loss / take_profit)
    if rule.get("type") in _VALID_EXIT_TYPES:
        atr = rule.get("atr_multiple")
        return isinstance(atr, (int, float)) and float(atr) > 0

    # Indicator-based rules
    indicator = rule.get("indicator", "")
    condition = rule.get("condition", "")
    value = rule.get("value")

    if indicator not in _VALID_INDICATORS:
        return False
    if condition not in _VALID_CONDITIONS:
        return False
    # value can be a float or a column reference string (like "sma_slow")
    if not isinstance(value, (int, float, str)):
        return False

    return True
