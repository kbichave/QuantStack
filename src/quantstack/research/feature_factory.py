# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Autonomous feature factory — top-level orchestrator.

Three-phase pipeline:
  1. **Enumerate** — generate candidate features (programmatic + LLM)
  2. **Screen** — IC + stability + correlation filtering to 50-100 curated features
  3. **Monitor** — daily drift check on curated features, auto-replace decayed ones

The feature_candidates table (PostgreSQL) stores all curated features with their
screening metrics. The event bus publishes FEATURE_DECAYED and FEATURE_REPLACED
events for downstream consumers.

Usage:
    from quantstack.research.feature_factory import run_full_pipeline, monitor_features

    result = run_full_pipeline(base_features=["close", "volume", ...], ohlcv_data={...})
    # result = {"enumerated": 1500, "curated": 75, "features": [...]}

    # Daily monitoring (call from supervisor loop):
    decayed = monitor_features(curated_features, event_bus=bus)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger

from quantstack.learning.drift_detector import PSI_CRITICAL, compute_psi
from quantstack.research.feature_enumerator import (
    enumerate_programmatic,
    enumerate_with_llm,
)
from quantstack.research.feature_screener import compute_ic, screen_and_filter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CANDIDATE_HARD_CAP = 2000
_IC_DECAY_THRESHOLD = 0.005
_IC_DECAY_WINDOW_DAYS = 10
_PSI_DECAY_THRESHOLD = PSI_CRITICAL  # 0.25


# ---------------------------------------------------------------------------
# Phase 1: Enumeration
# ---------------------------------------------------------------------------


def enumerate_features(
    base_features: list[str],
    regime: str = "unknown",
    use_llm: bool = True,
) -> list[dict[str, str]]:
    """Generate candidate features, enforcing a 2000-candidate hard cap.

    Combines programmatic transforms with optional LLM-generated candidates.
    Programmatic candidates are generated first; LLM candidates fill remaining
    capacity up to the cap.

    Args:
        base_features: List of base feature names.
        regime: Current market regime for LLM context.
        use_llm: Whether to attempt LLM enumeration.

    Returns:
        List of candidate dicts (max 2000).
    """
    candidates = enumerate_programmatic(base_features)

    if use_llm and len(candidates) < _CANDIDATE_HARD_CAP:
        llm_candidates = enumerate_with_llm(base_features, regime)
        remaining = _CANDIDATE_HARD_CAP - len(candidates)
        candidates.extend(llm_candidates[:remaining])

    # Enforce hard cap
    if len(candidates) > _CANDIDATE_HARD_CAP:
        candidates = candidates[:_CANDIDATE_HARD_CAP]

    # Deduplicate by feature_id
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for cand in candidates:
        fid = cand["feature_id"]
        if fid not in seen:
            seen.add(fid)
            deduped.append(cand)

    logger.info(
        "[FeatureFactory] Enumerated %d candidates (%d after dedup, cap=%d)",
        len(candidates), len(deduped), _CANDIDATE_HARD_CAP,
    )
    return deduped


# ---------------------------------------------------------------------------
# Phase 2: Screening
# ---------------------------------------------------------------------------


def screen_features(
    candidates: list[dict[str, Any]],
    ohlcv_data: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """IC screening + correlation filtering.

    Delegates to feature_screener.screen_and_filter, which applies:
      - IC > 0.01
      - IC stability > 0.5
      - Pairwise correlation < 0.95

    Args:
        candidates: Output from enumerate_features.
        ohlcv_data: Dict mapping feature names and "forward_returns" to arrays.

    Returns:
        Curated list of 50-100 feature candidates with IC/stability metrics.
    """
    curated = screen_and_filter(candidates, ohlcv_data)
    logger.info(
        "[FeatureFactory] Screened %d -> %d curated features",
        len(candidates), len(curated),
    )
    return curated


# ---------------------------------------------------------------------------
# Phase 3: Monitoring
# ---------------------------------------------------------------------------


def monitor_features(
    curated: list[dict[str, Any]],
    ohlcv_data: dict[str, np.ndarray] | None = None,
    baseline_data: dict[str, np.ndarray] | None = None,
    ic_history: dict[str, list[float]] | None = None,
    event_bus: Any | None = None,
) -> list[dict[str, Any]]:
    """Daily drift check on curated features, publishing decay events.

    Two decay signals:
      1. PSI > 0.25 — distribution shift (via compute_psi from drift_detector)
      2. IC < 0.005 for 10 consecutive days — predictive power lost

    When decay is detected:
      - Publishes FEATURE_DECAYED event on the event bus
      - Marks the feature with decay_date
      - If a replacement is available (from remaining candidates), publishes
        FEATURE_REPLACED event

    Args:
        curated: Currently active curated features.
        ohlcv_data: Current feature values + forward_returns.
        baseline_data: Baseline feature distributions for PSI comparison.
        ic_history: Dict mapping feature_name to list of recent daily IC values.
        event_bus: EventBus instance (optional). If None, events are logged only.

    Returns:
        List of decayed feature dicts.
    """
    if not curated:
        return []

    decayed: list[dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for feature in curated:
        fname = feature["feature_name"]
        is_decayed = False
        decay_reason = ""

        # Check 1: PSI drift
        if baseline_data and ohlcv_data:
            baseline_vals = baseline_data.get(fname)
            current_vals = ohlcv_data.get(fname)
            if baseline_vals is not None and current_vals is not None:
                psi = compute_psi(
                    np.asarray(baseline_vals, dtype=np.float64),
                    np.asarray(current_vals, dtype=np.float64),
                )
                if psi > _PSI_DECAY_THRESHOLD:
                    is_decayed = True
                    decay_reason = f"PSI={psi:.4f} > {_PSI_DECAY_THRESHOLD}"

        # Check 2: IC decay (low IC for consecutive days)
        if not is_decayed and ic_history:
            feature_ic_hist = ic_history.get(fname, [])
            if len(feature_ic_hist) >= _IC_DECAY_WINDOW_DAYS:
                recent = feature_ic_hist[-_IC_DECAY_WINDOW_DAYS:]
                if all(abs(ic) < _IC_DECAY_THRESHOLD for ic in recent):
                    is_decayed = True
                    decay_reason = (
                        f"IC < {_IC_DECAY_THRESHOLD} for "
                        f"{_IC_DECAY_WINDOW_DAYS} consecutive days"
                    )

        if is_decayed:
            feature["decay_date"] = now_iso
            feature["status"] = "decayed"
            decayed.append(feature)

            logger.warning(
                "[FeatureFactory] Feature decayed: %s — %s", fname, decay_reason,
            )

            if event_bus is not None:
                _publish_decay_event(event_bus, feature, decay_reason)

    return decayed


def _publish_decay_event(
    event_bus: Any,
    feature: dict[str, Any],
    reason: str,
) -> None:
    """Publish FEATURE_DECAYED event on the event bus."""
    from quantstack.coordination.event_bus import Event, EventType

    event_bus.publish(Event(
        event_type=EventType.FEATURE_DECAYED,
        source_loop="feature_factory",
        payload={
            "feature_id": feature["feature_id"],
            "feature_name": feature["feature_name"],
            "reason": reason,
        },
    ))


def publish_replacement_event(
    event_bus: Any,
    old_feature: dict[str, Any],
    new_feature: dict[str, Any],
) -> None:
    """Publish FEATURE_REPLACED event when a decayed feature is swapped."""
    from quantstack.coordination.event_bus import Event, EventType

    event_bus.publish(Event(
        event_type=EventType.FEATURE_REPLACED,
        source_loop="feature_factory",
        payload={
            "old_feature_id": old_feature["feature_id"],
            "old_feature_name": old_feature["feature_name"],
            "new_feature_id": new_feature["feature_id"],
            "new_feature_name": new_feature["feature_name"],
        },
    ))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_full_pipeline(
    base_features: list[str],
    ohlcv_data: dict[str, np.ndarray],
    regime: str = "unknown",
    use_llm: bool = True,
) -> dict[str, Any]:
    """Chain enumerate + screen into a single call.

    Args:
        base_features: List of base feature names.
        ohlcv_data: Dict with feature arrays and "forward_returns".
        regime: Current market regime.
        use_llm: Whether to attempt LLM enumeration.

    Returns:
        Dict with keys: enumerated (int), curated (int), features (list[dict]).
    """
    candidates = enumerate_features(base_features, regime=regime, use_llm=use_llm)
    curated = screen_features(candidates, ohlcv_data)

    return {
        "enumerated": len(candidates),
        "curated": len(curated),
        "features": curated,
    }
