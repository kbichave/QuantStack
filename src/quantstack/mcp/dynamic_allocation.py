# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Dynamic portfolio-level strategy allocation.

Replaces the static regime-strategy matrix (CLAUDE.md §7) with an allocation
engine that:
  1. Uses HRP (Hierarchical Risk Parity) across strategy return streams for
     diversification — correlated strategies get reduced weight.
  2. Updates weekly based on rolling 30-day strategy performance.
  3. Penalizes correlated strategies (two momentum strategies on the same sector
     get reduced weight).
  4. Respects existing regime affinity, risk limits, and forward_testing caps.

The static ``compute_allocation()`` in ``allocation.py`` is NOT replaced — it
remains available as a fast fallback.  This module adds a ``compute_dynamic_allocation()``
that should be called from the /meta workflow when sufficient performance history exists.

Failure modes:
  - <3 strategies with return histories → fall back to static allocation.
  - Singular covariance → HRP handles this natively (uses single-linkage clustering).
  - All strategies have 0 return → equal-weight within regime-eligible set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from quantstack.mcp.allocation import _match_regime, compute_allocation as static_alloc


@dataclass
class DynamicAllocation:
    """Single strategy allocation from the dynamic engine."""

    strategy_id: str
    strategy_name: str
    capital_pct: float
    hrp_weight: float  # raw HRP-derived weight before adjustments
    regime_score: float
    rolling_sharpe: float
    correlation_penalty: float  # 0-1, how much weight was reduced due to correlation
    mode: str  # "paper" or "live"
    reasoning: str


@dataclass
class DynamicAllocationPlan:
    """Output of the dynamic allocation engine."""

    regime: str
    regime_confidence: float
    allocations: list[DynamicAllocation]
    total_allocated_pct: float
    unallocated_pct: float
    method: str  # "hrp_dynamic" or "static_fallback"
    warnings: list[str] = field(default_factory=list)
    strategy_correlation_matrix: dict[str, dict[str, float]] | None = None


def compute_dynamic_allocation(
    regime: str,
    regime_confidence: float,
    strategies: list[dict[str, Any]],
    strategy_returns: dict[str, pd.Series],
    max_gross_exposure_pct: float = 1.0,
    forward_testing_cap: float = 0.10,
    lookback_days: int = 30,
    correlation_penalty_threshold: float = 0.7,
    min_strategies_for_hrp: int = 3,
) -> DynamicAllocationPlan:
    """
    Compute dynamic capital allocations using HRP across strategy return streams.

    Args:
        regime: Current regime label.
        regime_confidence: Confidence in regime classification (0-1).
        strategies: List of strategy dicts from registry (need strategy_id, name,
                    status, regime_affinity, backtest_summary, risk_params).
        strategy_returns: Dict mapping strategy_id → daily return Series.
                         Should cover at least ``lookback_days`` trading days.
        max_gross_exposure_pct: Maximum total allocation.
        forward_testing_cap: Max allocation for forward_testing strategies.
        lookback_days: Rolling window for performance evaluation.
        correlation_penalty_threshold: Strategy pairs with correlation above this
                                       get penalized.
        min_strategies_for_hrp: Minimum eligible strategies to use HRP.
                                Below this, fall back to static allocation.

    Returns:
        DynamicAllocationPlan with per-strategy allocations.
    """
    # Filter to eligible strategies (live or forward_testing, regime match)
    eligible = []
    warnings = []

    for strat in strategies:
        status = strat.get("status", "draft")
        if status not in ("live", "forward_testing"):
            continue

        affinity = strat.get("regime_affinity") or {}
        if isinstance(affinity, str):
            try:
                affinity = json.loads(affinity)
            except (ValueError, TypeError):
                affinity = {}

        regime_score = _match_regime(regime, affinity)
        if regime_score <= 0:
            continue

        strategy_id = strat["strategy_id"]
        returns = strategy_returns.get(strategy_id)

        eligible.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strat.get("name", ""),
                "status": status,
                "regime_score": regime_score,
                "risk_params": strat.get("risk_params") or {},
                "returns": returns,
            }
        )

    if not eligible:
        return DynamicAllocationPlan(
            regime=regime,
            regime_confidence=regime_confidence,
            allocations=[],
            total_allocated_pct=0.0,
            unallocated_pct=1.0,
            method="static_fallback",
            warnings=[f"No eligible strategies for regime '{regime}'"],
        )

    # Check if we have enough return history for HRP
    strategies_with_returns = [
        s
        for s in eligible
        if s["returns"] is not None and len(s["returns"]) >= lookback_days
    ]

    if len(strategies_with_returns) < min_strategies_for_hrp:
        # Fall back to static allocation
        static_result = static_alloc(
            regime=regime,
            regime_confidence=regime_confidence,
            strategies=strategies,
            max_gross_exposure_pct=max_gross_exposure_pct,
            forward_testing_cap=forward_testing_cap,
        )
        static_allocations = []
        for a in static_result.get("allocations", []):
            static_allocations.append(
                DynamicAllocation(
                    strategy_id=a["strategy_id"],
                    strategy_name=a["strategy_name"],
                    capital_pct=a["capital_pct"],
                    hrp_weight=0.0,
                    regime_score=a["regime_score"],
                    rolling_sharpe=a["ranking_sharpe"],
                    correlation_penalty=0.0,
                    mode=a["mode"],
                    reasoning=f"Static fallback (insufficient return history). {a['reasoning']}",
                )
            )
        return DynamicAllocationPlan(
            regime=regime,
            regime_confidence=regime_confidence,
            allocations=static_allocations,
            total_allocated_pct=static_result["total_allocated_pct"],
            unallocated_pct=static_result["unallocated_pct"],
            method="static_fallback",
            warnings=[
                f"Only {len(strategies_with_returns)} strategies have {lookback_days}d return history "
                f"(need {min_strategies_for_hrp}). Using static allocation."
            ],
        )

    # Build returns matrix for HRP
    returns_df = _build_returns_matrix(strategies_with_returns, lookback_days)
    if returns_df.empty or returns_df.shape[1] < 2:
        # Can't do HRP with <2 assets
        warnings.append("Cannot compute HRP with fewer than 2 return series")
        return _equal_weight_allocation(
            eligible,
            regime,
            regime_confidence,
            max_gross_exposure_pct,
            forward_testing_cap,
            warnings,
        )

    # Compute HRP weights
    hrp_weights = _compute_hrp_weights(returns_df)

    # Compute rolling Sharpe for each strategy
    rolling_sharpes = {}
    for sid in returns_df.columns:
        rets = returns_df[sid].dropna()
        if len(rets) > 5:
            rolling_sharpes[sid] = float(
                rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)
            )
        else:
            rolling_sharpes[sid] = 0.0

    # Compute correlation matrix for penalty
    corr_matrix = returns_df.corr()
    strategy_corr = {}
    for sid in returns_df.columns:
        strategy_corr[sid] = {
            other: round(float(corr_matrix.loc[sid, other]), 4)
            for other in returns_df.columns
        }

    # Apply correlation penalty: if two strategies are highly correlated,
    # reduce the weight of the one with lower Sharpe
    correlation_penalties = _compute_correlation_penalties(
        hrp_weights,
        corr_matrix,
        rolling_sharpes,
        correlation_penalty_threshold,
    )

    # Build final allocations
    allocations = []
    total = 0.0
    strat_lookup = {s["strategy_id"]: s for s in eligible}

    for sid, raw_weight in hrp_weights.items():
        strat = strat_lookup.get(sid)
        if strat is None:
            continue

        penalty = correlation_penalties.get(sid, 0.0)
        adjusted_weight = raw_weight * (1.0 - penalty)

        # Scale by regime score
        adjusted_weight *= strat["regime_score"]

        # Cap forward_testing
        if strat["status"] == "forward_testing":
            adjusted_weight = min(adjusted_weight, forward_testing_cap)

        # Cap by strategy's own risk params
        risk_params = strat.get("risk_params", {})
        if isinstance(risk_params, str):
            try:
                risk_params = json.loads(risk_params)
            except (ValueError, TypeError):
                risk_params = {}
        max_pos = risk_params.get(
            "max_position_pct", risk_params.get("position_pct", 0.15)
        )
        adjusted_weight = min(adjusted_weight, max_pos)

        # Don't exceed total cap
        adjusted_weight = min(adjusted_weight, max_gross_exposure_pct - total)

        if adjusted_weight < 0.005:
            continue

        mode = "paper" if strat["status"] == "forward_testing" else "live"
        rolling_s = rolling_sharpes.get(sid, 0.0)

        allocations.append(
            DynamicAllocation(
                strategy_id=sid,
                strategy_name=strat["strategy_name"],
                capital_pct=round(adjusted_weight, 4),
                hrp_weight=round(raw_weight, 4),
                regime_score=round(strat["regime_score"], 2),
                rolling_sharpe=round(rolling_s, 4),
                correlation_penalty=round(penalty, 4),
                mode=mode,
                reasoning=(
                    f"HRP weight {raw_weight:.3f} × regime {strat['regime_score']:.2f} "
                    f"× (1 - corr_penalty {penalty:.2f}) = {adjusted_weight:.3f}. "
                    f"30d Sharpe={rolling_s:.2f}."
                ),
            )
        )
        total += adjusted_weight

    # Scale down if regime confidence is low
    if regime_confidence < 0.6:
        scale = regime_confidence / 0.6
        for a in allocations:
            a.capital_pct = round(a.capital_pct * scale, 4)
        total *= scale
        warnings.append(
            f"Regime confidence {regime_confidence:.0%} < 60%: allocations scaled to {scale:.0%}"
        )

    return DynamicAllocationPlan(
        regime=regime,
        regime_confidence=regime_confidence,
        allocations=allocations,
        total_allocated_pct=round(total, 4),
        unallocated_pct=round(1.0 - total, 4),
        method="hrp_dynamic",
        warnings=warnings,
        strategy_correlation_matrix=strategy_corr,
    )


# =============================================================================
# HRP implementation (Hierarchical Risk Parity)
# =============================================================================


def _compute_hrp_weights(returns_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute HRP weights from a returns matrix.

    Uses single-linkage hierarchical clustering on the correlation distance
    matrix, then recursively bisects the portfolio to allocate inversely
    proportional to cluster variance.

    Reference: Lopez de Prado, "Building Diversified Portfolios that Outperform
    Out-of-Sample" (2016).
    """
    cov = returns_df.cov().values
    corr = returns_df.corr().values
    symbols = list(returns_df.columns)
    n = len(symbols)

    if n == 1:
        return {symbols[0]: 1.0}

    # Distance matrix: sqrt(0.5 * (1 - corr))
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = list(leaves_list(link))

    # Reorder covariance
    sorted_symbols = [symbols[i] for i in sort_ix]
    sorted_cov = cov[np.ix_(sort_ix, sort_ix)]

    # Recursive bisection
    weights = pd.Series(1.0, index=sorted_symbols)
    cluster_items = [sorted_symbols]

    while cluster_items:
        new_clusters = []
        for items in cluster_items:
            if len(items) <= 1:
                continue
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]

            # Inverse-variance allocation between clusters
            left_var = _cluster_variance(left, symbols, cov)
            right_var = _cluster_variance(right, symbols, cov)

            alloc_factor = 1 - left_var / (left_var + right_var + 1e-12)

            for item in left:
                weights[item] *= alloc_factor
            for item in right:
                weights[item] *= 1 - alloc_factor

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        cluster_items = new_clusters

    # Normalize
    total = weights.sum()
    if total > 1e-9:
        weights = weights / total

    return dict(weights)


def _cluster_variance(
    items: list[str],
    all_symbols: list[str],
    cov: np.ndarray,
) -> float:
    """Compute inverse-variance-weighted portfolio variance for a cluster."""
    indices = [all_symbols.index(s) for s in items]
    cluster_cov = cov[np.ix_(indices, indices)]

    # Inverse-variance weights within cluster
    ivp = 1.0 / (np.diag(cluster_cov) + 1e-12)
    ivp = ivp / ivp.sum()

    return float(ivp @ cluster_cov @ ivp)


# =============================================================================
# Correlation penalty
# =============================================================================


def _compute_correlation_penalties(
    hrp_weights: dict[str, float],
    corr_matrix: pd.DataFrame,
    rolling_sharpes: dict[str, float],
    threshold: float,
) -> dict[str, float]:
    """
    For each pair of highly correlated strategies, penalize the weaker one.

    Penalty = max pairwise correlation above threshold × 0.5.
    The strategy with lower rolling Sharpe in each pair gets penalized.
    """
    penalties: dict[str, float] = {sid: 0.0 for sid in hrp_weights}
    sids = list(hrp_weights.keys())

    for i, sid_a in enumerate(sids):
        for sid_b in sids[i + 1 :]:
            if sid_a not in corr_matrix.index or sid_b not in corr_matrix.index:
                continue
            corr_val = abs(float(corr_matrix.loc[sid_a, sid_b]))
            if corr_val > threshold:
                excess = (corr_val - threshold) / (1.0 - threshold + 1e-9)
                penalty = min(excess * 0.5, 0.5)  # cap at 50% reduction

                sharpe_a = rolling_sharpes.get(sid_a, 0.0)
                sharpe_b = rolling_sharpes.get(sid_b, 0.0)

                # Penalize the weaker strategy
                if sharpe_a < sharpe_b:
                    penalties[sid_a] = max(penalties[sid_a], penalty)
                else:
                    penalties[sid_b] = max(penalties[sid_b], penalty)

    return penalties


# =============================================================================
# Helpers
# =============================================================================


def _build_returns_matrix(
    strategies: list[dict[str, Any]],
    lookback_days: int,
) -> pd.DataFrame:
    """Build a returns DataFrame aligned across strategies."""
    series_dict = {}
    for strat in strategies:
        returns = strat.get("returns")
        if returns is None or len(returns) == 0:
            continue
        # Take last lookback_days
        series_dict[strat["strategy_id"]] = returns.tail(lookback_days)

    if not series_dict:
        return pd.DataFrame()

    df = pd.DataFrame(series_dict)
    # Forward-fill short gaps, then drop remaining NaNs
    df = df.ffill(limit=3).dropna()
    return df


def _equal_weight_allocation(
    eligible: list[dict[str, Any]],
    regime: str,
    regime_confidence: float,
    max_gross_exposure_pct: float,
    forward_testing_cap: float,
    warnings: list[str],
) -> DynamicAllocationPlan:
    """Equal-weight fallback when HRP can't be computed."""
    n = len(eligible)
    base_weight = min(max_gross_exposure_pct / n, 0.15)

    allocations = []
    total = 0.0
    for strat in eligible:
        w = base_weight
        if strat["status"] == "forward_testing":
            w = min(w, forward_testing_cap)
        mode = "paper" if strat["status"] == "forward_testing" else "live"
        allocations.append(
            DynamicAllocation(
                strategy_id=strat["strategy_id"],
                strategy_name=strat["strategy_name"],
                capital_pct=round(w, 4),
                hrp_weight=round(1.0 / n, 4),
                regime_score=round(strat["regime_score"], 2),
                rolling_sharpe=0.0,
                correlation_penalty=0.0,
                mode=mode,
                reasoning=f"Equal-weight fallback: 1/{n} = {1.0/n:.3f}",
            )
        )
        total += w

    warnings.append("Using equal-weight allocation (insufficient data for HRP)")

    return DynamicAllocationPlan(
        regime=regime,
        regime_confidence=regime_confidence,
        allocations=allocations,
        total_allocated_pct=round(total, 4),
        unallocated_pct=round(1.0 - total, 4),
        method="equal_weight_fallback",
        warnings=warnings,
    )
