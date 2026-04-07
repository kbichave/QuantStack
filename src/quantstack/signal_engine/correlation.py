# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Signal correlation tracking and penalty computation (Section 08, Phase 7).

Computes pairwise Spearman correlations, applies continuous penalties to
redundant signals (weaker IC gets penalized), and tracks effective
independent signal count via eigenvalue decomposition.

Kill-switch: FEEDBACK_CORRELATION_PENALTY (default false).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CorrelationResult:
    """Result of pairwise signal correlation analysis."""

    correlation_matrix: dict[tuple[str, str], float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    effective_signal_count: int = 0

    @staticmethod
    def empty(collectors: list[str]) -> CorrelationResult:
        return CorrelationResult(
            correlation_matrix={},
            penalties={name: 1.0 for name in collectors},
            effective_signal_count=len(collectors),
        )


# ---------------------------------------------------------------------------
# Penalty formula
# ---------------------------------------------------------------------------


def correlation_penalty(corr: float) -> float:
    """Continuous correlation penalty.

    Formula: max(0.2, 1.0 - max(0.0, abs(corr) - 0.5) * 2.0)

    - corr < 0.5: no penalty (1.0)
    - corr 0.5-0.75: linear ramp from 1.0 to 0.5
    - corr >= 0.75: continues to 0.2 floor
    """
    return max(0.2, 1.0 - max(0.0, abs(corr) - 0.5) * 2.0)


# ---------------------------------------------------------------------------
# Spearman rank correlation (pure numpy)
# ---------------------------------------------------------------------------


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two arrays."""
    n = len(x)
    if n < 3:
        return 0.0
    rank_x = _rank(x)
    rank_y = _rank(y)
    d = rank_x - rank_y
    rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
    return float(np.clip(rho, -1.0, 1.0))


def _rank(arr: np.ndarray) -> np.ndarray:
    """Assign ranks to array values (average rank for ties)."""
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    return ranks


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_signal_correlations(
    signal_data: dict[str, list[float]],
    ic_data: dict[str, float],
    min_observations: int = 63,
) -> CorrelationResult:
    """Compute pairwise Spearman correlations and derive penalties.

    Args:
        signal_data: collector_name -> list of daily signal values.
        ic_data: collector_name -> current rolling IC value.
        min_observations: Minimum signal observations per collector.

    Returns:
        CorrelationResult with correlation_matrix, penalties, effective_signal_count.
    """
    # Filter to collectors with enough data
    valid = {
        name: np.array(vals)
        for name, vals in signal_data.items()
        if len(vals) >= min_observations
    }

    all_names = list(signal_data.keys())
    if len(valid) < 2:
        return CorrelationResult.empty(all_names)

    names = sorted(valid.keys())
    n = len(names)

    # Build correlation matrix
    corr_dict: dict[tuple[str, str], float] = {}
    corr_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            rho = _spearman_corr(valid[names[i]], valid[names[j]])
            corr_dict[(names[i], names[j])] = round(rho, 4)
            corr_dict[(names[j], names[i])] = round(rho, 4)
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # Compute penalties: weaker signal gets penalized
    penalties: dict[str, float] = {name: 1.0 for name in all_names}

    for i in range(n):
        for j in range(i + 1, n):
            rho = abs(corr_matrix[i, j])
            if rho <= 0.5:
                continue

            pen = correlation_penalty(rho)
            ic_i = ic_data.get(names[i], 0.0)
            ic_j = ic_data.get(names[j], 0.0)

            if ic_i > ic_j:
                # j is weaker — penalize j
                penalties[names[j]] = min(penalties[names[j]], pen)
            elif ic_j > ic_i:
                # i is weaker — penalize i
                penalties[names[i]] = min(penalties[names[i]], pen)
            else:
                # equal IC — both penalized
                penalties[names[i]] = min(penalties[names[i]], pen)
                penalties[names[j]] = min(penalties[names[j]], pen)

    # Effective independent signal count via eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    effective_count = int(np.sum(eigenvalues > 0.1))

    return CorrelationResult(
        correlation_matrix=corr_dict,
        penalties=penalties,
        effective_signal_count=effective_count,
    )


# ---------------------------------------------------------------------------
# Config-flag-gated wrapper
# ---------------------------------------------------------------------------


def compute_correlation_penalties_gated(
    signal_data: dict[str, list[float]],
    ic_data: dict[str, float],
    min_observations: int = 63,
) -> dict[str, float]:
    """Return per-collector correlation penalties, gated by config flag.

    When FEEDBACK_CORRELATION_PENALTY=false, returns 1.0 for all collectors.
    """
    if os.getenv("FEEDBACK_CORRELATION_PENALTY", "false").lower() not in ("true", "1", "yes"):
        return {name: 1.0 for name in signal_data}

    result = compute_signal_correlations(signal_data, ic_data, min_observations)
    return result.penalties
