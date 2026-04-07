"""Experiment prioritization formula and queue sorting (AR-9).

Ranks competing experiments before the research graph selects which to validate.
Priority = (expected_IC * regime_fit * novelty_score) / estimated_compute_cost.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Minimum compute cost floor to avoid division by zero
_MIN_COMPUTE_COST = 100

# Default token estimates by complexity
_COMPLEXITY_TOKENS = {
    "simple": 5_000,
    "ml": 30_000,
}


def compute_priority(
    expected_ic: float,
    regime_fit: float,
    novelty_score: float,
    estimated_compute_cost: int,
) -> float:
    """Rank experiments by expected value per compute dollar.

    Parameters
    ----------
    expected_ic : float
        Prior from similar strategies in knowledge graph, or 0.03 default
        for novel hypotheses.
    regime_fit : float
        How well the hypothesis matches current regime. Range [0.0, 1.0].
    novelty_score : float
        1.0 if not in knowledge graph, 0.1 if similar hypothesis tested before.
    estimated_compute_cost : int
        Token estimate. Simple rule = 5,000. ML model = 30,000.
    """
    cost = max(estimated_compute_cost, _MIN_COMPUTE_COST)
    return (expected_ic * regime_fit * novelty_score) / cost


def _get_novelty_score(hypothesis: str) -> float:
    """Check knowledge graph for hypothesis novelty.

    Returns 1.0 when KG is not yet built (graceful degradation).
    """
    try:
        from quantstack.tools.functions.knowledge_graph import check_hypothesis_novelty
        result = check_hypothesis_novelty(hypothesis)
        return result.get("novelty_score", 1.0)
    except (ImportError, Exception):
        return 1.0


def _get_regime_fit(hypothesis: str, current_regime: str) -> float:
    """Estimate regime fit for a hypothesis. Defaults to 1.0."""
    # Regime fit requires strategy metadata not available at prioritization time.
    # Default to 1.0 (neutral) — actual regime filtering happens in backtest validation.
    return 1.0


def prioritize_experiments(
    experiments: list[dict],
    current_regime: str,
) -> list[dict]:
    """Sort experiment queue by priority (descending).

    Each experiment dict must contain: 'hypothesis', 'complexity'
    ('simple' or 'ml'), and optionally 'expected_ic'.

    Uses knowledge graph for novelty check when available (falls back
    to novelty_score=1.0 when KG is not yet built).
    """
    if not experiments:
        return []

    scored = []
    for exp in experiments:
        expected_ic = exp.get("expected_ic", 0.03)
        complexity = exp.get("complexity", "simple")
        compute_cost = _COMPLEXITY_TOKENS.get(complexity, _COMPLEXITY_TOKENS["simple"])
        novelty = _get_novelty_score(exp.get("hypothesis", ""))
        regime_fit = _get_regime_fit(exp.get("hypothesis", ""), current_regime)

        priority = compute_priority(expected_ic, regime_fit, novelty, compute_cost)
        scored.append({**exp, "_priority": priority, "_novelty_score": novelty})

    scored.sort(key=lambda x: x["_priority"], reverse=True)
    return scored
