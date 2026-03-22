# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Prompt Tuner — outcome-driven prompt improvement recommendations.

Analyzes trade outcomes to identify patterns where desk agent recommendations
were systematically wrong. Generates specific, actionable prompt modifications.

This does NOT auto-modify prompts. It produces a report that the PM reviews
during /reflect sessions and applies manually.

Design invariants:
  - All persistence is best-effort JSON. A write failure never blocks a trade.
  - Recommendations require 5+ outcomes showing the same pattern before
    surfacing — avoids reacting to noise.
  - Thread-safe via Lock on all state mutations.

Outcome matching logic per desk:
  - market-intel: correct if predicted regime matches realized regime, or if
    return direction matched regime implication (risk_on → positive return).
  - alpha-research: correct if signal direction matched realized return sign.
  - risk: correct if recommended sizing didn't result in outsized loss
    (loss < 2x expected).
  - execution: correct if realized slippage was within 2x of forecast.

Usage:
    tuner = PromptTuner()
    tuner.record_outcome(
        desk="market-intel",
        prediction={"macro_regime": "risk_on_trending", "event_risk": "none"},
        outcome={"realized_return": -0.03, "regime_after": "risk_off_trending"},
    )
    recommendations = tuner.get_recommendations()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_STATE_PATH = Path.home() / ".quant_pod" / "prompt_tuner.json"

_VALID_DESKS = frozenset({"market-intel", "alpha-research", "risk", "execution"})

# Minimum outcomes before generating a recommendation. Prevents reacting to
# small-sample noise — 5 outcomes showing the same pattern is the floor.
_MIN_OUTCOMES_FOR_RECOMMENDATION = 5

# Regime implications: what return direction each regime implies
_REGIME_DIRECTION: dict[str, str] = {
    "risk_on_trending": "positive",
    "trending_up": "positive",
    "bullish": "positive",
    "risk_off_trending": "negative",
    "trending_down": "negative",
    "bearish": "negative",
    "ranging": "neutral",
    "sideways": "neutral",
    "unknown": "neutral",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PromptRecommendation:
    """A single prompt improvement recommendation."""

    desk: str
    pattern: str  # short description of the error pattern
    evidence: str  # "7/12 regime calls were wrong when VIX > 25"
    suggested_change: str  # "Add VIX > 25 check to risk_off classification"
    confidence: float  # 0-1, based on sample size and consistency
    priority: str  # "high", "medium", "low"


@dataclass
class DeskAccuracy:
    """Per-desk accuracy statistics."""

    desk: str
    total_outcomes: int
    correct: int
    wrong: int
    accuracy_pct: float
    worst_pattern: str | None  # most common error


@dataclass
class _OutcomeRecord:
    """Single recorded prediction/outcome pair."""

    desk: str
    prediction: dict[str, Any]
    outcome: dict[str, Any]
    symbol: str | None
    timestamp: str
    correct: bool | None  # evaluated lazily


# ---------------------------------------------------------------------------
# PromptTuner
# ---------------------------------------------------------------------------


class PromptTuner:
    """
    Records desk agent predictions and realized outcomes, then analyzes
    the gap to produce actionable prompt improvement recommendations.

    State is persisted to a JSON file. All public methods are thread-safe.
    """

    _lock = Lock()

    def __init__(self, state_path: Path | str | None = None) -> None:
        self._state_path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
        self._records: list[_OutcomeRecord] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        desk: str,
        prediction: dict[str, Any],
        outcome: dict[str, Any],
        symbol: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Record a desk agent prediction and what actually happened.

        Args:
            desk: Which desk made the prediction. Must be one of
                  "market-intel", "alpha-research", "risk", "execution".
            prediction: What the desk predicted. Keys vary by desk:
                - market-intel: macro_regime, event_risk, direction
                - alpha-research: signal_direction, signal_quality, confidence
                - risk: recommended_size, expected_loss
                - execution: expected_slippage_bps, recommended_window
            outcome: What actually happened:
                - realized_return, regime_after, actual_slippage_bps, etc.
            symbol: Optional ticker for context.
            timestamp: When the prediction was made. Defaults to now.
        """
        if desk not in _VALID_DESKS:
            logger.warning(
                f"[PromptTuner] Unknown desk '{desk}'. "
                f"Valid desks: {', '.join(sorted(_VALID_DESKS))}"
            )
            return

        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        correct = _evaluate_correctness(desk, prediction, outcome)

        record = _OutcomeRecord(
            desk=desk,
            prediction=prediction,
            outcome=outcome,
            symbol=symbol,
            timestamp=ts,
            correct=correct,
        )

        with self._lock:
            self._records.append(record)
            self._persist()

        label = "CORRECT" if correct else "WRONG" if correct is not None else "UNKNOWN"
        logger.debug(
            f"[PromptTuner] Recorded {desk} outcome ({label}): "
            f"prediction={prediction}, outcome={outcome}"
        )

    def get_recommendations(
        self,
        min_outcomes: int = _MIN_OUTCOMES_FOR_RECOMMENDATION,
    ) -> list[PromptRecommendation]:
        """
        Analyze recorded outcomes to find systematic errors.

        Only generates recommendations when there are ``min_outcomes`` or more
        outcomes showing the same pattern. Returns recommendations sorted by
        priority (high first).
        """
        recommendations: list[PromptRecommendation] = []

        with self._lock:
            records = list(self._records)

        by_desk: dict[str, list[_OutcomeRecord]] = {}
        for r in records:
            by_desk.setdefault(r.desk, []).append(r)

        for desk, desk_records in by_desk.items():
            evaluated = [r for r in desk_records if r.correct is not None]
            if len(evaluated) < min_outcomes:
                continue

            desk_recs = _detect_patterns(desk, evaluated, min_outcomes)
            recommendations.extend(desk_recs)

        # Sort by priority: high > medium > low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))

        return recommendations

    def get_desk_accuracy(self, desk: str) -> DeskAccuracy:
        """
        Per-desk accuracy statistics.

        Returns a ``DeskAccuracy`` with total outcomes, correct/wrong counts,
        accuracy percentage, and the most common error pattern.
        """
        with self._lock:
            desk_records = [r for r in self._records if r.desk == desk]

        evaluated = [r for r in desk_records if r.correct is not None]
        correct_count = sum(1 for r in evaluated if r.correct)
        wrong_count = len(evaluated) - correct_count
        accuracy = (correct_count / len(evaluated) * 100.0) if evaluated else 0.0

        # Find worst pattern among wrong predictions
        worst = _find_worst_pattern(desk, [r for r in evaluated if not r.correct])

        return DeskAccuracy(
            desk=desk,
            total_outcomes=len(evaluated),
            correct=correct_count,
            wrong=wrong_count,
            accuracy_pct=round(accuracy, 1),
            worst_pattern=worst,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Write current state to JSON. Called under lock."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [
                {
                    "desk": r.desk,
                    "prediction": r.prediction,
                    "outcome": r.outcome,
                    "symbol": r.symbol,
                    "timestamp": r.timestamp,
                    "correct": r.correct,
                }
                for r in self._records
            ]
            self._state_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.warning(f"[PromptTuner] Failed to persist state: {exc}")

    def _load(self) -> None:
        """Load persisted state from JSON."""
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text())
            for entry in raw:
                self._records.append(
                    _OutcomeRecord(
                        desk=entry["desk"],
                        prediction=entry["prediction"],
                        outcome=entry["outcome"],
                        symbol=entry.get("symbol"),
                        timestamp=entry["timestamp"],
                        correct=entry.get("correct"),
                    )
                )
            logger.info(
                f"[PromptTuner] Loaded {len(self._records)} outcome records "
                f"from {self._state_path}"
            )
        except Exception as exc:
            logger.warning(f"[PromptTuner] Failed to load state: {exc}")


# ---------------------------------------------------------------------------
# Correctness evaluation per desk
# ---------------------------------------------------------------------------


def _evaluate_correctness(
    desk: str,
    prediction: dict[str, Any],
    outcome: dict[str, Any],
) -> bool | None:
    """
    Evaluate whether a desk prediction was correct given the outcome.

    Returns True (correct), False (wrong), or None (cannot evaluate — missing
    keys in prediction or outcome).
    """
    if desk == "market-intel":
        return _eval_market_intel(prediction, outcome)
    if desk == "alpha-research":
        return _eval_alpha_research(prediction, outcome)
    if desk == "risk":
        return _eval_risk(prediction, outcome)
    if desk == "execution":
        return _eval_execution(prediction, outcome)
    return None


def _eval_market_intel(
    prediction: dict[str, Any],
    outcome: dict[str, Any],
) -> bool | None:
    """
    Market-intel is correct if:
      1. Predicted regime matches realized regime, OR
      2. Return direction matched regime implication.
    """
    predicted_regime = prediction.get("macro_regime") or prediction.get("regime")
    realized_regime = outcome.get("regime_after") or outcome.get("regime")
    realized_return = outcome.get("realized_return")

    # Direct regime match
    if predicted_regime and realized_regime:
        if _normalize_regime(predicted_regime) == _normalize_regime(realized_regime):
            return True

    # Fallback: check if regime implication matched return direction
    if predicted_regime and realized_return is not None:
        implied_direction = _REGIME_DIRECTION.get(
            _normalize_regime(predicted_regime), "neutral"
        )
        if implied_direction == "positive" and realized_return > 0:
            return True
        if implied_direction == "negative" and realized_return < 0:
            return True
        if implied_direction == "neutral":
            # Neutral regimes are hard to evaluate — only wrong if big move
            return abs(realized_return) < 0.03

        return False

    return None


def _eval_alpha_research(
    prediction: dict[str, Any],
    outcome: dict[str, Any],
) -> bool | None:
    """
    Alpha-research is correct if signal direction matched realized return sign.
    """
    direction = prediction.get("signal_direction") or prediction.get("direction")
    realized_return = outcome.get("realized_return")

    if direction is None or realized_return is None:
        return None

    direction_lower = str(direction).lower()
    if direction_lower in ("long", "buy", "bullish"):
        return realized_return > 0
    if direction_lower in ("short", "sell", "bearish"):
        return realized_return < 0
    if direction_lower in ("neutral", "hold", "flat"):
        return abs(realized_return) < 0.02

    return None


def _eval_risk(
    prediction: dict[str, Any],
    outcome: dict[str, Any],
) -> bool | None:
    """
    Risk desk is correct if recommended sizing didn't result in outsized loss.
    Loss < 2x expected is acceptable.
    """
    expected_loss = prediction.get("expected_loss")
    realized_return = outcome.get("realized_return")

    if expected_loss is None or realized_return is None:
        return None

    # expected_loss is typically a negative number or a positive magnitude
    expected_magnitude = abs(float(expected_loss))

    if realized_return >= 0:
        return True  # No loss — risk desk was fine

    actual_loss = abs(realized_return)
    # Correct if actual loss was within 2x of expected
    return actual_loss <= expected_magnitude * 2.0


def _eval_execution(
    prediction: dict[str, Any],
    outcome: dict[str, Any],
) -> bool | None:
    """
    Execution desk is correct if realized slippage was within 2x of forecast.
    """
    expected_slippage = prediction.get("expected_slippage_bps")
    actual_slippage = outcome.get("actual_slippage_bps")

    if expected_slippage is None or actual_slippage is None:
        return None

    expected_mag = abs(float(expected_slippage))
    actual_mag = abs(float(actual_slippage))

    # Allow at least 5 bps of baseline noise so tiny forecasts don't auto-fail
    threshold = max(expected_mag * 2.0, 5.0)
    return actual_mag <= threshold


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def _detect_patterns(
    desk: str,
    records: list[_OutcomeRecord],
    min_outcomes: int,
) -> list[PromptRecommendation]:
    """
    Detect systematic error patterns for a desk and generate recommendations.
    """
    wrong = [r for r in records if not r.correct]
    total = len(records)
    wrong_count = len(wrong)

    if wrong_count < min_outcomes:
        return []

    recommendations: list[PromptRecommendation] = []

    if desk == "market-intel":
        recommendations.extend(_detect_regime_misclassification(wrong, total))
    elif desk == "alpha-research":
        recommendations.extend(_detect_signal_overestimation(records, wrong, total))
    elif desk == "risk":
        recommendations.extend(_detect_sizing_aggression(records, wrong, total))
    elif desk == "execution":
        recommendations.extend(_detect_timing_errors(records, wrong, total))

    return recommendations


def _detect_regime_misclassification(
    wrong_records: list[_OutcomeRecord],
    total: int,
) -> list[PromptRecommendation]:
    """
    Pattern: market-intel predicted regime X but outcome was Y more than 40%
    of the time.
    """
    # Group by predicted regime
    by_predicted: dict[str, list[_OutcomeRecord]] = {}
    for r in wrong_records:
        predicted = r.prediction.get("macro_regime") or r.prediction.get("regime")
        if predicted:
            by_predicted.setdefault(str(predicted), []).append(r)

    recommendations: list[PromptRecommendation] = []
    for predicted_regime, records in by_predicted.items():
        if len(records) < 3:
            continue

        error_rate = len(records) / total
        if error_rate < 0.40:
            continue

        # Find what the actual regime was most often
        actual_regimes: dict[str, int] = {}
        for r in records:
            actual = r.outcome.get("regime_after") or r.outcome.get("regime", "unknown")
            actual_regimes[str(actual)] = actual_regimes.get(str(actual), 0) + 1

        most_common_actual = max(actual_regimes, key=actual_regimes.get)  # type: ignore[arg-type]
        evidence = (
            f"{len(records)}/{total} regime calls for '{predicted_regime}' were wrong — "
            f"actual regime was '{most_common_actual}' "
            f"{actual_regimes[most_common_actual]} times"
        )

        confidence = min(1.0, len(records) / 20.0)  # Saturates at 20 observations
        priority = "high" if error_rate > 0.60 else "medium"

        recommendations.append(
            PromptRecommendation(
                desk="market-intel",
                pattern=f"Regime misclassification: predicted {predicted_regime}, "
                f"usually was {most_common_actual}",
                evidence=evidence,
                suggested_change=(
                    f"Add distinguishing conditions between '{predicted_regime}' "
                    f"and '{most_common_actual}' in market-intel desk prompt. "
                    f"Review ADX/ATR thresholds and VIX regime boundaries."
                ),
                confidence=round(confidence, 2),
                priority=priority,
            )
        )

    return recommendations


def _detect_signal_overestimation(
    all_records: list[_OutcomeRecord],
    wrong_records: list[_OutcomeRecord],
    total: int,
) -> list[PromptRecommendation]:
    """
    Pattern: alpha-research rated signal as "high quality" but realized return
    was negative >50% of the time.
    """
    # Filter for high-quality/high-confidence signals
    high_confidence = [r for r in all_records if _is_high_confidence(r.prediction)]

    if len(high_confidence) < _MIN_OUTCOMES_FOR_RECOMMENDATION:
        return []

    wrong_high = [r for r in high_confidence if not r.correct]
    error_rate = len(wrong_high) / len(high_confidence)

    if error_rate <= 0.50:
        return []

    confidence = min(1.0, len(high_confidence) / 15.0)
    priority = "high" if error_rate > 0.65 else "medium"

    return [
        PromptRecommendation(
            desk="alpha-research",
            pattern="Signal quality overestimation on high-confidence calls",
            evidence=(
                f"{len(wrong_high)}/{len(high_confidence)} high-confidence signals "
                f"had negative realized returns ({error_rate:.0%} error rate)"
            ),
            suggested_change=(
                "Raise the confidence threshold for 'high quality' signals in "
                "alpha-research desk prompt. Consider requiring multiple confirming "
                "indicators before assigning high confidence."
            ),
            confidence=round(confidence, 2),
            priority=priority,
        )
    ]


def _detect_sizing_aggression(
    all_records: list[_OutcomeRecord],
    wrong_records: list[_OutcomeRecord],
    total: int,
) -> list[PromptRecommendation]:
    """
    Pattern: risk desk recommended full size but outcome was loss >60% of
    the time.
    """
    full_size = [r for r in all_records if _is_full_size(r.prediction)]

    if len(full_size) < _MIN_OUTCOMES_FOR_RECOMMENDATION:
        return []

    wrong_full = [r for r in full_size if not r.correct]
    error_rate = len(wrong_full) / len(full_size)

    if error_rate <= 0.60:
        return []

    confidence = min(1.0, len(full_size) / 15.0)
    priority = "high" if error_rate > 0.75 else "medium"

    return [
        PromptRecommendation(
            desk="risk",
            pattern="Full-size recommendations resulting in frequent losses",
            evidence=(
                f"{len(wrong_full)}/{len(full_size)} full-size recommendations "
                f"resulted in losses exceeding 2x expected ({error_rate:.0%} error rate)"
            ),
            suggested_change=(
                "Default to half-Kelly sizing in risk desk prompt. Add a sizing "
                "penalty when regime confidence is below 0.7 or when multiple "
                "risk factors are present simultaneously."
            ),
            confidence=round(confidence, 2),
            priority=priority,
        )
    ]


def _detect_timing_errors(
    all_records: list[_OutcomeRecord],
    wrong_records: list[_OutcomeRecord],
    total: int,
) -> list[PromptRecommendation]:
    """
    Pattern: execution desk slippage forecasts were systematically low.
    """
    if len(wrong_records) < _MIN_OUTCOMES_FOR_RECOMMENDATION:
        return []

    error_rate = len(wrong_records) / total
    if error_rate <= 0.40:
        return []

    # Compute average slippage overshoot
    overshoots: list[float] = []
    for r in wrong_records:
        expected = r.prediction.get("expected_slippage_bps")
        actual = r.outcome.get("actual_slippage_bps")
        if expected is not None and actual is not None:
            overshoots.append(abs(float(actual)) - abs(float(expected)))

    avg_overshoot = sum(overshoots) / len(overshoots) if overshoots else 0.0
    confidence = min(1.0, len(wrong_records) / 15.0)
    priority = "medium" if error_rate < 0.60 else "high"

    return [
        PromptRecommendation(
            desk="execution",
            pattern="Slippage forecasts systematically too optimistic",
            evidence=(
                f"{len(wrong_records)}/{total} fills had slippage exceeding "
                f"2x forecast (avg overshoot: {avg_overshoot:.1f} bps)"
            ),
            suggested_change=(
                "Increase baseline slippage estimate in execution desk prompt. "
                "Consider adding time-of-day and liquidity adjustments — slippage "
                "is typically higher at open/close and in low-ADV names."
            ),
            confidence=round(confidence, 2),
            priority=priority,
        )
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_regime(regime: str) -> str:
    """Normalize regime string for comparison."""
    return regime.strip().lower().replace(" ", "_").replace("-", "_")


def _is_high_confidence(prediction: dict[str, Any]) -> bool:
    """Check if a prediction was labeled as high quality/confidence."""
    quality = str(prediction.get("signal_quality", "")).lower()
    if quality in ("high", "strong", "excellent"):
        return True

    confidence = prediction.get("confidence")
    if confidence is not None:
        try:
            return float(confidence) >= 0.75
        except (ValueError, TypeError):
            pass

    return False


def _is_full_size(prediction: dict[str, Any]) -> bool:
    """Check if the risk desk recommended full position size."""
    size = prediction.get("recommended_size")
    if size is None:
        return False

    size_str = str(size).lower()
    if size_str in ("full", "1.0", "100%"):
        return True

    try:
        return float(size) >= 0.9
    except (ValueError, TypeError):
        return False


def _find_worst_pattern(desk: str, wrong_records: list[_OutcomeRecord]) -> str | None:
    """Find the most common error pattern among wrong predictions."""
    if not wrong_records:
        return None

    if desk == "market-intel":
        # Most common regime misclassification
        misses: dict[str, int] = {}
        for r in wrong_records:
            predicted = r.prediction.get("macro_regime") or r.prediction.get("regime")
            actual = r.outcome.get("regime_after") or r.outcome.get("regime")
            if predicted and actual:
                key = f"predicted {predicted}, was {actual}"
                misses[key] = misses.get(key, 0) + 1
        return max(misses, key=misses.get) if misses else None  # type: ignore[arg-type]

    if desk == "alpha-research":
        # Most common direction miss
        misses = {}
        for r in wrong_records:
            direction = r.prediction.get("signal_direction") or r.prediction.get(
                "direction"
            )
            ret = r.outcome.get("realized_return")
            if direction and ret is not None:
                actual_dir = "positive" if ret > 0 else "negative"
                key = f"predicted {direction}, return was {actual_dir}"
                misses[key] = misses.get(key, 0) + 1
        return max(misses, key=misses.get) if misses else None  # type: ignore[arg-type]

    if desk == "risk":
        return "sizing too aggressive — losses exceeded expected"

    if desk == "execution":
        return "slippage forecast too optimistic"

    return None
