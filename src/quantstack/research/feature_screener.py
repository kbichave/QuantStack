# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2 of the autonomous feature factory: IC screening and correlation filtering.

Takes enumerated candidates and OHLCV data, computes information coefficient (IC)
and IC stability, then filters to a curated set of 50-100 features with low
cross-correlation.

IC is measured as Spearman rank correlation between feature values and forward returns.
IC stability is the inverse of the standard deviation of rolling IC — higher means
the feature's predictive power is consistent, not a fluke.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

_IC_MIN = 0.01
_STABILITY_MIN = 0.5
_CORRELATION_MAX = 0.95
_TARGET_MIN = 50
_TARGET_MAX = 100


def compute_ic(feature_values: np.ndarray, forward_returns: np.ndarray) -> float:
    """Spearman rank correlation between feature values and forward returns.

    Args:
        feature_values: 1-D array of feature observations.
        forward_returns: 1-D array of corresponding forward returns.

    Returns:
        Spearman correlation coefficient (float). Returns 0.0 if inputs are
        too short or constant.
    """
    fv = np.asarray(feature_values, dtype=np.float64).ravel()
    fr = np.asarray(forward_returns, dtype=np.float64).ravel()

    n = min(len(fv), len(fr))
    if n < 10:
        return 0.0

    fv, fr = fv[:n], fr[:n]

    # Remove NaN/inf pairs
    mask = np.isfinite(fv) & np.isfinite(fr)
    fv, fr = fv[mask], fr[mask]
    if len(fv) < 10:
        return 0.0

    # Spearman = Pearson on ranks
    fv_ranks = _rankdata(fv)
    fr_ranks = _rankdata(fr)

    fv_dm = fv_ranks - fv_ranks.mean()
    fr_dm = fr_ranks - fr_ranks.mean()

    num = float(np.sum(fv_dm * fr_dm))
    den = float(np.sqrt(np.sum(fv_dm ** 2) * np.sum(fr_dm ** 2)))

    if den == 0:
        return 0.0

    return num / den


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Rank data with average tie-breaking (pure numpy, no scipy)."""
    sorter = np.argsort(arr)
    ranks = np.empty_like(sorter, dtype=np.float64)
    ranks[sorter] = np.arange(1, len(arr) + 1, dtype=np.float64)

    # Handle ties: average rank for tied values
    sorted_arr = arr[sorter]
    # Find tie groups
    i = 0
    while i < len(sorted_arr):
        j = i + 1
        while j < len(sorted_arr) and sorted_arr[j] == sorted_arr[i]:
            j += 1
        if j > i + 1:
            # Average rank for this tie group
            avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
            for k in range(i, j):
                ranks[sorter[k]] = avg_rank
        i = j

    return ranks


def compute_ic_stability(
    feature_values: np.ndarray,
    forward_returns: np.ndarray,
    window: int = 63,
) -> float:
    """IC stability = 1 / std(rolling IC).

    Higher values mean the feature's IC is consistent across time windows.
    Returns 0.0 if there aren't enough observations for at least 3 rolling windows.

    Args:
        feature_values: 1-D array of feature observations (time-ordered).
        forward_returns: 1-D array of corresponding forward returns.
        window: Rolling window size in observations (default 63 ~ 1 quarter).

    Returns:
        Stability score (float). 0.0 if insufficient data.
    """
    fv = np.asarray(feature_values, dtype=np.float64).ravel()
    fr = np.asarray(forward_returns, dtype=np.float64).ravel()

    n = min(len(fv), len(fr))
    if n < window * 3:
        return 0.0

    fv, fr = fv[:n], fr[:n]

    rolling_ics: list[float] = []
    for start in range(0, n - window + 1, window // 2):
        end = start + window
        if end > n:
            break
        ic = compute_ic(fv[start:end], fr[start:end])
        rolling_ics.append(ic)

    if len(rolling_ics) < 3:
        return 0.0

    ic_std = float(np.std(rolling_ics))
    if ic_std == 0:
        return float("inf")

    return 1.0 / ic_std


# ---------------------------------------------------------------------------
# Correlation filtering
# ---------------------------------------------------------------------------


def _pairwise_correlation(features_matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix for columns of features_matrix."""
    if features_matrix.shape[1] == 0:
        return np.array([])

    # Standardize columns
    mu = features_matrix.mean(axis=0)
    sigma = features_matrix.std(axis=0)
    sigma[sigma == 0] = 1.0
    standardized = (features_matrix - mu) / sigma

    n = standardized.shape[0]
    corr = (standardized.T @ standardized) / n
    return corr


# ---------------------------------------------------------------------------
# Main screening pipeline
# ---------------------------------------------------------------------------


def screen_and_filter(
    candidates: list[dict[str, Any]],
    ohlcv_data: dict[str, np.ndarray],
    ic_min: float = _IC_MIN,
    stability_min: float = _STABILITY_MIN,
    correlation_max: float = _CORRELATION_MAX,
) -> list[dict[str, Any]]:
    """Screen candidates by IC, stability, and cross-correlation.

    Pipeline:
      1. Compute each candidate's feature values from ohlcv_data
      2. Filter by IC > ic_min
      3. Filter by IC stability > stability_min
      4. Remove features with pairwise correlation > correlation_max
         (keep the one with higher IC)
      5. Target 50-100 output features

    Args:
        candidates: List of candidate dicts from the enumerator.
        ohlcv_data: Dict with keys like "close", "volume", "rsi_14" mapping
                    to 1-D numpy arrays. Must include "forward_returns".
        ic_min: Minimum absolute IC to pass screening.
        stability_min: Minimum IC stability to pass screening.
        correlation_max: Maximum pairwise correlation allowed.

    Returns:
        Curated list of candidate dicts, enriched with "ic" and "ic_stability" keys.
    """
    forward_returns = ohlcv_data.get("forward_returns")
    if forward_returns is None or len(forward_returns) < 30:
        logger.warning("[FeatureScreener] No forward_returns in ohlcv_data, returning empty")
        return []

    forward_returns = np.asarray(forward_returns, dtype=np.float64)

    # Step 1-2: compute IC and filter
    ic_passed: list[dict[str, Any]] = []
    for cand in candidates:
        feature_name = cand["feature_name"]
        feature_values = ohlcv_data.get(feature_name)
        if feature_values is None:
            continue

        feature_values = np.asarray(feature_values, dtype=np.float64)
        ic = compute_ic(feature_values, forward_returns)

        if abs(ic) < ic_min:
            continue

        cand_with_ic = {**cand, "ic": ic}
        ic_passed.append(cand_with_ic)

    logger.info("[FeatureScreener] IC filter: %d / %d passed", len(ic_passed), len(candidates))

    # Step 3: stability filter
    stability_passed: list[dict[str, Any]] = []
    for cand in ic_passed:
        feature_values = np.asarray(ohlcv_data[cand["feature_name"]], dtype=np.float64)
        stability = compute_ic_stability(feature_values, forward_returns)

        if stability < stability_min:
            continue

        cand["ic_stability"] = stability
        stability_passed.append(cand)

    logger.info(
        "[FeatureScreener] Stability filter: %d / %d passed",
        len(stability_passed), len(ic_passed),
    )

    if not stability_passed:
        return []

    # Step 4: correlation filter — sort by abs(IC) descending, greedily remove correlated
    stability_passed.sort(key=lambda c: abs(c["ic"]), reverse=True)

    # Build feature matrix for correlation computation
    n_obs = len(forward_returns)
    feature_names = [c["feature_name"] for c in stability_passed]
    feature_matrix = np.column_stack([
        np.asarray(ohlcv_data[name], dtype=np.float64)[:n_obs]
        for name in feature_names
    ])

    corr_matrix = _pairwise_correlation(feature_matrix)
    n_features = len(stability_passed)

    kept_indices: list[int] = []
    dropped: set[int] = set()

    for i in range(n_features):
        if i in dropped:
            continue
        kept_indices.append(i)
        # Drop any subsequent feature too correlated with this one
        for j in range(i + 1, n_features):
            if j in dropped:
                continue
            if abs(corr_matrix[i, j]) > correlation_max:
                dropped.add(j)

    curated = [stability_passed[i] for i in kept_indices]

    logger.info(
        "[FeatureScreener] Correlation filter: %d / %d kept (target %d-%d)",
        len(curated), len(stability_passed), _TARGET_MIN, _TARGET_MAX,
    )

    # Cap at TARGET_MAX (already sorted by IC)
    if len(curated) > _TARGET_MAX:
        curated = curated[:_TARGET_MAX]

    return curated
