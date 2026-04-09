# Copyright 2024 QuantStack Contributors
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
import os
from typing import Any

import numpy as np
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.core.hierarchy.regime_classifier import RegimeType, WeeklyRegimeClassifier

from quantstack.core.hierarchy.regime.hmm_model import (
    HMMRegimeModel,
    HMMRegimeState,
)
from quantstack.signal_engine.staleness import check_freshness

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
    if not check_freshness(symbol, "1d", max_days=4):
        return {}
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

        # Map HMM state to QuantStack trend/vol taxonomy
        trend_regime = _hmm_state_to_trend(result.state)
        vol_regime = _hmm_state_to_vol(result.state)

        # P05 §5.2: Vol-conditioned sub-regime (e.g. "trending_up_low_vol")
        vol_sub = _vol_sub_regime(df)
        sub_regime = f"{trend_regime}_{vol_sub}"

        # Confidence: use regime stability (how certain HMM is about current state)
        confidence = round(result.regime_stability, 3)

        # Build probability dict with string keys for JSON serialization
        probs = {
            state.name: round(prob, 4)
            for state, prob in result.state_probabilities.items()
        }

        # Transition probability: 1 - max(filtered posteriors)
        # High value = HMM uncertain about current state = possible regime transition
        transition_probability = round(1.0 - max(probs.values()), 4)

        # Second most likely state (potential transition target)
        sorted_states = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        most_likely_next = sorted_states[1][0] if len(sorted_states) > 1 else None

        return {
            "trend_regime": trend_regime,
            "volatility_regime": vol_regime,
            "sub_regime": sub_regime,
            "confidence": confidence,
            "regime_label": result.state.name,
            "hmm_state": result.state.name,
            "hmm_probabilities": probs,
            "hmm_stability": round(result.regime_stability, 3),
            "hmm_expected_duration": round(result.expected_duration, 1),
            "transition_probability": transition_probability,
            "most_likely_next_regime": most_likely_next,
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
    """Map HMM state to QuantStack trend taxonomy."""
    name = state.name
    if "BULL" in name:
        return "trending_up"
    if "BEAR" in name:
        return "trending_down"
    return "ranging"


def _hmm_state_to_vol(state: "HMMRegimeState") -> str:
    """Map HMM state to QuantStack volatility taxonomy."""
    name = state.name
    if "HIGH_VOL" in name:
        return "high"
    if "LOW_VOL" in name:
        return "low"
    return "normal"


# --- Legacy mapping functions (kept for rule-based path) ---


def _map_trend(regime: RegimeType, ema_alignment: int, momentum_score: float) -> str:
    """Map WeeklyRegimeClassifier result to QuantStack trend taxonomy."""
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


# ---------------------------------------------------------------------------
# Transition-based position sizing (Section 15)
# ---------------------------------------------------------------------------


def transition_sizing_factor(transition_probability: float | None) -> float:
    """Map transition probability to a position sizing multiplier.

    Tiers:
        P < 0.10  -> 1.0   (no adjustment)
        0.10-0.30 -> 0.75  (mild reduction)
        0.30-0.50 -> 0.50  (moderate reduction)
        P >= 0.50 -> 0.25  (severe reduction, but never zero)
    """
    if transition_probability is None:
        return 1.0
    if transition_probability < 0.10:
        return 1.0
    if transition_probability < 0.30:
        return 0.75
    if transition_probability < 0.50:
        return 0.50
    return 0.25


def transition_sizing_factor_gated(transition_probability: float | None) -> float:
    """Config-flag-gated wrapper for transition_sizing_factor."""
    if os.getenv("FEEDBACK_TRANSITION_SIZING", "false").lower() != "true":
        return 1.0
    return transition_sizing_factor(transition_probability)


# ---------------------------------------------------------------------------
# Vol-conditioned sub-regimes (Section 15)
# ---------------------------------------------------------------------------


def _vol_sub_regime(df: "pd.DataFrame") -> str:
    """Classify current vol relative to trailing 252-day distribution.

    Uses 20-day realized vol vs 30th/70th percentile of trailing 252-day
    realized vol series.

    Returns: 'low_vol' | 'normal_vol' | 'high_vol'
    """
    closes = df["close"].values
    if len(closes) < 40:
        return "normal_vol"

    # Daily log returns
    log_returns = np.diff(np.log(closes))

    # Rolling 20-day realized vol (annualized)
    window = 20
    if len(log_returns) < window:
        return "normal_vol"

    # Current 20-day realized vol
    current_vol = float(np.std(log_returns[-window:])) * np.sqrt(252)

    # Trailing 252-day series of 20-day rolling vols
    lookback = min(252, len(log_returns) - window + 1)
    vol_series = []
    for i in range(lookback):
        start = len(log_returns) - window - i
        if start < 0:
            break
        chunk = log_returns[start:start + window]
        vol_series.append(float(np.std(chunk)) * np.sqrt(252))

    if len(vol_series) < 10:
        return "normal_vol"

    p30 = float(np.percentile(vol_series, 30))
    p70 = float(np.percentile(vol_series, 70))

    if current_vol < p30:
        return "low_vol"
    if current_vol > p70:
        return "high_vol"
    return "normal_vol"
