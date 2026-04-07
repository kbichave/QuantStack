# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
IC-based weight adjustment for signal synthesis (Section 07, Phase 7).

Computes continuous sigmoid IC factors per collector, applies IC_IR
consistency penalty, and provides a weight floor safety check.

Kill-switch: FEEDBACK_IC_WEIGHT_ADJUSTMENT (default false).
"""

from __future__ import annotations

import math
import os
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Core sigmoid function
# ---------------------------------------------------------------------------


def ic_factor(ic: float) -> float:
    """Continuous sigmoid IC factor.

    Formula: 1 / (1 + exp(-50 * (ic - 0.02)))

    - IC > 0.04: factor ~ 1.0 (full weight)
    - IC = 0.02: factor = 0.5 (half weight)
    - IC < 0.00: factor ~ 0.0 (near-zero weight)
    """
    exponent = -50.0 * (ic - 0.02)
    # Clamp to avoid overflow
    exponent = max(-100.0, min(100.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


# ---------------------------------------------------------------------------
# Per-collector IC factor computation
# ---------------------------------------------------------------------------


def compute_ic_factors(
    ic_data: dict[str, list[float]],
    min_observations: int = 21,
) -> dict[str, float]:
    """Compute IC-based weight adjustment factors for each collector.

    Args:
        ic_data: Mapping of collector_name -> list of daily IC values
                 (most recent last).
        min_observations: Minimum IC observations required. Below this,
                          factor defaults to 1.0 (cold-start).

    Returns:
        Mapping of collector_name -> ic_factor in [0.0, 1.0].
    """
    factors: dict[str, float] = {}

    for name, values in ic_data.items():
        if len(values) < min_observations:
            factors[name] = 1.0
            continue

        window = values[-21:]
        rolling_ic = sum(window) / len(window)
        factor = ic_factor(rolling_ic)

        # IC_IR consistency penalty
        std = _std(window)
        if std > 0:
            ic_ir = rolling_ic / std
        else:
            ic_ir = float("inf")  # perfectly consistent

        if ic_ir < 0.1:
            factor *= 0.7

        factors[name] = round(factor, 6)

    return factors


def compute_ic_factors_gated(
    ic_data: dict[str, list[float]],
    min_observations: int = 21,
) -> dict[str, float]:
    """Config-flag-gated wrapper for compute_ic_factors."""
    if os.getenv("FEEDBACK_IC_WEIGHT_ADJUSTMENT", "false").lower() not in ("true", "1", "yes"):
        return {name: 1.0 for name in ic_data}
    return compute_ic_factors(ic_data, min_observations)


# ---------------------------------------------------------------------------
# Weight floor safety check
# ---------------------------------------------------------------------------


def check_weight_floor(
    static_weights: dict[str, float],
    ic_factors: dict[str, float],
    floor: float = 0.1,
) -> dict[str, Any]:
    """Check if total effective weight is above the floor.

    When total effective weight < floor, fall back to static weights
    (IC adjustment disabled for this cycle).

    Returns:
        Dict with keys: floor_triggered (bool), effective_weights (dict),
        total_effective_weight (float).
    """
    effective = {}
    for name, static_w in static_weights.items():
        factor = ic_factors.get(name, 1.0)
        effective[name] = static_w * factor

    total = sum(effective.values())

    if total < floor:
        logger.warning(
            f"[ic_weights] Weight floor triggered: total={total:.4f} < {floor}. "
            "Falling back to static weights."
        )
        return {
            "floor_triggered": True,
            "effective_weights": dict(static_weights),
            "total_effective_weight": total,
        }

    return {
        "floor_triggered": False,
        "effective_weights": effective,
        "total_effective_weight": total,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
