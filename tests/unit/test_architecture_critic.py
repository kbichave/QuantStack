"""Tests for the quarterly architecture critic."""

from __future__ import annotations

from quantstack.meta.architecture_critic import (
    MIN_DAYS_BETWEEN_RUNS,
    generate_improvement_proposal,
    identify_bottleneck,
    should_run,
)


def test_bottleneck_identified_by_cost_alpha_ratio():
    stats = [
        {"node_id": "cheap_good", "token_cost": 100, "alpha": 1.0},
        {"node_id": "expensive_bad", "token_cost": 10000, "alpha": 0.01},
        {"node_id": "moderate", "token_cost": 500, "alpha": 0.5},
    ]
    bottleneck = identify_bottleneck(stats)
    assert bottleneck is not None
    assert bottleneck["node_id"] == "expensive_bad"

    # Verify the proposal is a non-empty string.
    proposal = generate_improvement_proposal(bottleneck)
    assert isinstance(proposal, str)
    assert len(proposal) > 0


def test_quarterly_cadence_enforced():
    assert MIN_DAYS_BETWEEN_RUNS == 90
    # Never run before => should run.
    assert should_run(None) is True
    # Ran yesterday => should not run.
    assert should_run("2026-04-06") is False
    # Ran 100 days ago => should run.
    assert should_run("2025-12-28") is True


def test_no_bottleneck_when_empty_stats():
    assert identify_bottleneck([]) is None
