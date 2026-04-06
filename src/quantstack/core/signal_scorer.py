"""
Deterministic signal scorer for strategy entry rules.

Evaluates a strategy's entry_rules JSONB against current market data and
produces a signal value in [-1, 1] plus a confidence score in [0, 1].

Called by run_signal_scoring() (section-10) nightly — once per strategy × symbol pair.
No database access, no LLM calls, pure stdlib.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SUPPORTED_OPERATORS = {"<", ">", "<=", ">=", "=="}


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _evaluate_rule(rule: dict, market_data: dict) -> float | None:
    """
    Evaluate a single rule against market_data.

    Returns a score in [0, 1] (unsigned — caller applies direction sign),
    or None if the rule is unevaluable.

    Scoring: linear interpolation from threshold (0.0) to 2× distance (1.0),
    clamped to [0, 1]. Distance reference = |value - threshold| at the
    boundary of full satisfaction (i.e., threshold itself as the zero-score
    point, with the score increasing linearly beyond it).
    """
    field = rule.get("field")
    operator = rule.get("operator")
    threshold = rule.get("threshold")

    # Structural validity
    if not field or not operator or threshold is None:
        logger.debug("Skipping malformed rule (missing field/operator/threshold): %r", rule)
        return None

    if operator not in _SUPPORTED_OPERATORS:
        logger.debug("Skipping rule with unsupported operator %r for field %r", operator, field)
        return None

    # Field presence
    if field not in market_data:
        logger.debug("Skipping rule: field %r absent from market_data", field)
        return None

    value = market_data[field]

    # Numeric check
    if not _is_numeric(value):
        logger.debug(
            "Skipping rule: field %r has non-numeric value %r", field, value
        )
        return None

    if not _is_numeric(threshold):
        logger.debug("Skipping rule: threshold %r is non-numeric", threshold)
        return None

    # Compute signed distance from threshold in the satisfying direction.
    # For "<" / "<=": satisfying direction is below threshold → distance = threshold - value
    # For ">" / ">=": satisfying direction is above threshold → distance = value - threshold
    # For "==": only exact match gets any score (distance = 0 if not equal)
    if operator in ("<", "<="):
        distance = float(threshold) - float(value)
    elif operator in (">", ">="):
        distance = float(value) - float(threshold)
    else:  # "=="
        distance = 0.0 if float(value) == float(threshold) else -1.0

    if distance <= 0:
        # Rule not satisfied (or == not matched) → score 0.0
        return 0.0

    # Linear interpolation: score = distance / reference_distance, clamped to [0, 1].
    #
    # "At 2× distance from threshold (in the satisfying direction): score = 1.0"
    # means the reference distance = abs(threshold) / 3.
    # Concrete example from spec: threshold=30, RSI=20 → distance=10, reference=10 → score=1.0.
    # 10 = 30 / 3, which is consistent with score = distance / (threshold/3).
    #
    # For threshold=0, fall back to 1.0 to avoid division by zero.
    ref = abs(float(threshold)) / 3.0 if float(threshold) != 0 else 1.0
    score = min(distance / ref, 1.0)
    return score


def score_signal(
    entry_rules: list[dict],
    market_data: dict,
) -> tuple[float, float]:
    """
    Evaluate strategy entry_rules against current market data.

    Args:
        entry_rules: list of rule dicts, each with keys:
            - field (str): market data field name
            - operator (str): one of "<", ">", "<=", ">=", "=="
            - threshold (numeric): comparison value
            - direction (str, optional): "bullish" (default) or "bearish"
        market_data: dict mapping field names to their current numeric values

    Returns:
        (signal_value, confidence) where:
            signal_value: float in [-1, 1]. Positive = bullish, negative = bearish.
            confidence:   float in [0, 1]. Fraction of rules that were evaluable.

    If entry_rules is empty or fewer than 50% of rules are evaluable,
    returns (0.0, 0.0).
    """
    if not entry_rules:
        return (0.0, 0.0)

    signed_scores: list[float] = []
    evaluable_count = 0

    for rule in entry_rules:
        score = _evaluate_rule(rule, market_data)
        if score is None:
            # Unevaluable — already logged inside _evaluate_rule
            continue

        evaluable_count += 1
        direction = rule.get("direction", "bullish")
        # Bearish rules contribute negative signal
        signed_scores.append(-score if direction == "bearish" else score)

    total = len(entry_rules)
    confidence = evaluable_count / total

    if confidence < 0.5:
        return (0.0, 0.0)

    signal_value = sum(signed_scores) / len(signed_scores) if signed_scores else 0.0

    # Clamp to valid ranges
    signal_value = max(-1.0, min(1.0, signal_value))
    confidence = max(0.0, min(1.0, confidence))

    return (signal_value, confidence)
