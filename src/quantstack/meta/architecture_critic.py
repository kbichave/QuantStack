"""Quarterly architecture critic.

Identifies the graph node with the worst token-cost-to-alpha ratio and
generates a human-readable improvement proposal.  Runs at most once per
quarter to avoid churn.
"""

from __future__ import annotations

from datetime import datetime, timezone

MIN_DAYS_BETWEEN_RUNS = 90


def should_run(last_run_date: str | None) -> bool:
    """Return True if at least 90 days have elapsed since *last_run_date*.

    *last_run_date* should be ISO-8601 (e.g. ``"2026-01-01"``).  If None,
    the critic has never run and should execute.
    """
    if last_run_date is None:
        return True
    last = datetime.fromisoformat(last_run_date).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - last).days >= MIN_DAYS_BETWEEN_RUNS


def identify_bottleneck(node_stats: list[dict]) -> dict | None:
    """Find the node with the highest ``token_cost / alpha`` ratio.

    Each dict in *node_stats* must have keys ``node_id``, ``token_cost``,
    and ``alpha``.  Nodes with ``alpha <= 0`` are treated as infinite cost
    and are returned first.  Returns None if the list is empty.
    """
    if not node_stats:
        return None

    def _cost_ratio(stat: dict) -> float:
        alpha = stat.get("alpha", 0)
        if alpha <= 0:
            return float("inf")
        return stat["token_cost"] / alpha

    return max(node_stats, key=_cost_ratio)


def generate_improvement_proposal(bottleneck: dict) -> str:
    """Return a human-readable improvement proposal for *bottleneck*."""
    node_id = bottleneck.get("node_id", "unknown")
    token_cost = bottleneck.get("token_cost", 0)
    alpha = bottleneck.get("alpha", 0)

    if alpha <= 0:
        return (
            f"Node '{node_id}' consumes {token_cost:,} tokens but contributes "
            f"zero or negative alpha ({alpha}). Consider removing or replacing "
            f"this node with a cheaper deterministic alternative."
        )

    ratio = token_cost / alpha
    return (
        f"Node '{node_id}' has the worst cost/alpha ratio ({ratio:,.1f} "
        f"tokens per unit alpha). Token cost: {token_cost:,}, alpha: {alpha:.4f}. "
        f"Consider model-tier downgrade, prompt compression, or caching "
        f"repeated computations."
    )
