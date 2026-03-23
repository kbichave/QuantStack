# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Regime collector — HMM-primary with rule-based fallback.

Uses HMMRegimeModel as the primary regime detector, providing probabilistic
state assignments (state probabilities, expected duration, stability).
Falls back to WeeklyRegimeClassifier if HMM fitting fails or hmmlearn
is not installed.

v1.1 upgrade: HMM replaces WeeklyRegimeClassifier as the primary source.
The key improvement is that HMM provides state *probabilities*, not just a
label — enabling probabilistic position sizing and regime-conditional
synthesis weights downstream.
"""

import asyncio
from typing import Any

from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.core.hierarchy.regime_classifier import RegimeType, WeeklyRegimeClassifier

from quantstack.core.hierarchy.regime.hmm_model import (
    HMMRegimeModel,
    HMMRegimeState,
)

_MIN_BARS = 60
_HMM_MIN_BARS = 120  # HMM needs more history for stable fitting


async def collect_regime(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Classify market regime for *symbol* from locally stored daily OHLCV.

    Returns a dict with keys:
        trend_regime       : "trending_up" | "trending_down" | "ranging" | "unknown"
        volatility_regime  : "low" | "normal" | "high" | "extreme"
        confidence         : float [0, 1]
        regime_label       : str — raw regime label
        ema_alignment      : int (-1, 0, 1)
        momentum_score     : float
        bars_in_regime     : int
        --- HMM-specific (present when HMM succeeds) ---
        hmm_state          : str — HMM state name (e.g., "LOW_VOL_BULL")
        hmm_probabilities  : dict[str, float] — per-state probabilities
        hmm_stability      : float [0, 1] — how stable current regime is
        hmm_expected_duration : float — expected bars remaining in regime
        regime_source      : "hmm" | "rule_based" — which model produced the result
    """
    try:
        return await asyncio.to_thread(_collect_regime_sync, symbol, store)
    except Exception as exc:
        logger.warning(f"[regime] {symbol}: {exc}")
        return {
            "trend_regime": "unknown",
            "volatility_regime": "normal",
            "confidence": 0.0,
            "regime_source": "fallback",
        }


def _collect_regime_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    df = store.load_ohlcv(symbol, Timeframe.D1)
    if df is None or len(df) < _MIN_BARS:
        return {
            "trend_regime": "unknown",
            "volatility_regime": "normal",
            "confidence": 0.0,
            "regime_source": "insufficient_data",
        }

    # --- Try HMM first (probabilistic, richer output) ---
    hmm_result = _try_hmm_regime(df)

    # --- Always run rule-based as baseline / fallback ---
    rule_result = _rule_based_regime(df)

    if hmm_result is not None:
        # HMM succeeded — use it as primary, enrich with rule-based details
        return _merge_hmm_with_rules(hmm_result, rule_result)

    # HMM failed — fall back to rule-based
    rule_result["regime_source"] = "rule_based"
    return rule_result


def _try_hmm_regime(df: "pd.DataFrame") -> dict[str, Any] | None:
    """Attempt HMM regime detection. Returns None on any failure."""
    if len(df) < _HMM_MIN_BARS:
        return None

    try:
        model = HMMRegimeModel(lookback=min(252, len(df)), min_train_samples=80)
        model.fit(df)

        if not model.is_fitted:
            return None

        result = model.predict(df)

        # Map HMM state to QuantPod trend/vol taxonomy
        trend_regime = _hmm_state_to_trend(result.state)
        vol_regime = _hmm_state_to_vol(result.state)

        # Confidence: use regime stability (how certain HMM is about current state)
        confidence = round(result.regime_stability, 3)

        # Build probability dict with string keys for JSON serialization
        probs = {
            state.name: round(prob, 4)
            for state, prob in result.state_probabilities.items()
        }

        return {
            "trend_regime": trend_regime,
            "volatility_regime": vol_regime,
            "confidence": confidence,
            "regime_label": result.state.name,
            "hmm_state": result.state.name,
            "hmm_probabilities": probs,
            "hmm_stability": round(result.regime_stability, 3),
            "hmm_expected_duration": round(result.expected_duration, 1),
            "regime_source": "hmm",
        }
    except Exception as exc:
        logger.warning(f"[regime] HMM failed: {exc} — falling back to rule-based")
        return None


def _rule_based_regime(df: "pd.DataFrame") -> dict[str, Any]:
    """Run the original WeeklyRegimeClassifier."""
    classifier = WeeklyRegimeClassifier()
    ctx = classifier.classify(df)

    trend_regime = _map_trend(ctx.regime, ctx.ema_alignment, ctx.momentum_score)
    vol_regime = _map_vol(ctx.volatility_regime)

    return {
        "trend_regime": trend_regime,
        "volatility_regime": vol_regime,
        "confidence": round(ctx.confidence, 3),
        "regime_label": ctx.regime.value,
        "ema_alignment": ctx.ema_alignment,
        "momentum_score": round(ctx.momentum_score, 3),
        "bars_in_regime": ctx.bars_in_regime,
    }


def _merge_hmm_with_rules(hmm: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """
    Merge HMM (primary) with rule-based (secondary) results.

    HMM provides: trend, vol, confidence, probabilities, stability, duration.
    Rules provide: ema_alignment, momentum_score, bars_in_regime.

    If HMM and rules disagree on trend direction, reduce confidence by 15%
    — the disagreement signals regime ambiguity.
    """
    merged = {**hmm}  # Start with HMM output

    # Enrich with rule-based details not available from HMM
    merged["ema_alignment"] = rules.get("ema_alignment", 0)
    merged["momentum_score"] = rules.get("momentum_score", 0.0)
    merged["bars_in_regime"] = rules.get("bars_in_regime", 0)
    merged["rule_trend"] = rules.get("trend_regime", "unknown")
    merged["rule_confidence"] = rules.get("confidence", 0.0)

    # Disagreement penalty: if HMM and rules disagree on direction, reduce confidence
    hmm_trend = hmm.get("trend_regime", "unknown")
    rule_trend = rules.get("trend_regime", "unknown")

    if hmm_trend != "unknown" and rule_trend != "unknown" and hmm_trend != rule_trend:
        # Directional disagreement — regime is ambiguous
        original_conf = merged["confidence"]
        merged["confidence"] = round(max(0.1, original_conf - 0.15), 3)
        merged["regime_disagreement"] = True
        logger.debug(
            f"[regime] HMM({hmm_trend}) disagrees with rules({rule_trend}) "
            f"— confidence {original_conf:.3f} → {merged['confidence']:.3f}"
        )
    else:
        merged["regime_disagreement"] = False

    return merged


def _hmm_state_to_trend(state: "HMMRegimeState") -> str:
    """Map HMM state to QuantPod trend taxonomy."""
    name = state.name
    if "BULL" in name:
        return "trending_up"
    if "BEAR" in name:
        return "trending_down"
    return "ranging"


def _hmm_state_to_vol(state: "HMMRegimeState") -> str:
    """Map HMM state to QuantPod volatility taxonomy."""
    name = state.name
    if "HIGH_VOL" in name:
        return "high"
    if "LOW_VOL" in name:
        return "low"
    return "normal"


# --- Legacy mapping functions (kept for rule-based path) ---


def _map_trend(regime: RegimeType, ema_alignment: int, momentum_score: float) -> str:
    """Map WeeklyRegimeClassifier result to QuantPod trend taxonomy."""
    if regime == RegimeType.BULL:
        return "trending_up"
    if regime == RegimeType.BEAR:
        return "trending_down"
    if regime == RegimeType.SIDEWAYS:
        if abs(momentum_score) < 0.15:
            return "ranging"
        return "trending_up" if ema_alignment > 0 else "trending_down"
    return "unknown"


def _map_vol(vol_regime_int: int) -> str:
    """Map volatility_regime int (-1, 0, 1) to string label."""
    return {-1: "low", 0: "normal", 1: "high"}.get(vol_regime_int, "normal")
