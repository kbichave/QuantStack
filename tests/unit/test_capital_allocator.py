"""Tests for dynamic capital allocation (Section 03)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.portfolio.capital_allocator import (
    compute_allocation_scores,
    compute_budgets,
    get_strategy_budget_remaining,
)


# -- 3.1 Allocation Scoring Tests --


def test_score_computation_known_inputs():
    """score = sharpe * capacity * (1 - correlation_penalty) * regime_fit."""
    strategies = [{"strategy_id": "s1", "status": "live", "regime_affinity": ["trending_up"]}]
    closed_trades = pd.DataFrame({
        "strategy_id": ["s1"] * 20,
        "pnl": np.random.default_rng(42).normal(100, 50, 20),
        "closed_at": pd.bdate_range("2024-01-01", periods=20),
    })
    scores = compute_allocation_scores(
        strategies=strategies,
        closed_trades=closed_trades,
        current_regime="trending_up",
        correlation_matrix=pd.DataFrame({"s1": [0.0]}, index=["s1"]),
        adv_data={"s1": 10_000_000},
    )
    assert len(scores) == 1
    assert scores[0]["strategy_id"] == "s1"
    assert scores[0]["score"] > 0
    assert scores[0]["regime_fit"] == 1.0


def test_zero_regime_fit_produces_zero_score():
    """Mismatched regime -> regime_fit=0 -> score=0."""
    strategies = [{"strategy_id": "s1", "status": "live", "regime_affinity": ["trending_up"]}]
    closed_trades = pd.DataFrame({
        "strategy_id": ["s1"] * 20,
        "pnl": np.random.default_rng(42).normal(100, 50, 20),
        "closed_at": pd.bdate_range("2024-01-01", periods=20),
    })
    scores = compute_allocation_scores(
        strategies=strategies,
        closed_trades=closed_trades,
        current_regime="trending_down",  # mismatch
        correlation_matrix=pd.DataFrame({"s1": [0.0]}, index=["s1"]),
        adv_data={"s1": 10_000_000},
    )
    assert scores[0]["score"] == 0.0
    assert scores[0]["regime_fit"] == 0.0


def test_high_correlation_penalty_reduces_score():
    """Higher correlation penalty -> lower score."""
    strategies = [
        {"strategy_id": "s1", "status": "live", "regime_affinity": ["trending_up"]},
        {"strategy_id": "s2", "status": "live", "regime_affinity": ["trending_up"]},
    ]
    closed_trades = pd.DataFrame({
        "strategy_id": ["s1"] * 20 + ["s2"] * 20,
        "pnl": list(np.random.default_rng(42).normal(100, 50, 20)) * 2,
        "closed_at": list(pd.bdate_range("2024-01-01", periods=20)) * 2,
    })
    # High correlation between s1 and s2
    high_corr = pd.DataFrame({"s1": [0.0, 0.8], "s2": [0.8, 0.0]}, index=["s1", "s2"])
    low_corr = pd.DataFrame({"s1": [0.0, 0.1], "s2": [0.1, 0.0]}, index=["s1", "s2"])

    scores_high = compute_allocation_scores(
        strategies, closed_trades, "trending_up", high_corr, {"s1": 10_000_000, "s2": 10_000_000},
    )
    scores_low = compute_allocation_scores(
        strategies, closed_trades, "trending_up", low_corr, {"s1": 10_000_000, "s2": 10_000_000},
    )

    # Higher correlation -> lower scores
    assert scores_high[0]["score"] < scores_low[0]["score"]


def test_capacity_constraint_reduces_score():
    """Low ADV relative to capital -> capacity < 1 -> lower score."""
    strategies = [{"strategy_id": "s1", "status": "live", "regime_affinity": ["trending_up"]}]
    closed_trades = pd.DataFrame({
        "strategy_id": ["s1"] * 20,
        "pnl": np.random.default_rng(42).normal(100, 50, 20),
        "closed_at": pd.bdate_range("2024-01-01", periods=20),
    })
    corr = pd.DataFrame({"s1": [0.0]}, index=["s1"])

    scores_high_adv = compute_allocation_scores(
        strategies, closed_trades, "trending_up", corr, {"s1": 10_000_000},
    )
    scores_low_adv = compute_allocation_scores(
        strategies, closed_trades, "trending_up", corr, {"s1": 100},  # very low ADV
    )

    assert scores_low_adv[0]["capacity_component"] < scores_high_adv[0]["capacity_component"]


def test_low_trade_count_uses_backtest_sharpe_with_haircut():
    """< 10 trades -> use backtest Sharpe with DSR haircut."""
    strategies = [{
        "strategy_id": "s1", "status": "live",
        "regime_affinity": ["trending_up"],
        "backtest_sharpe": 1.5,
        "dsr_penalty": 0.3,
    }]
    closed_trades = pd.DataFrame({
        "strategy_id": ["s1"] * 5,  # < 10 trades
        "pnl": [100, 200, -50, 150, 80],
        "closed_at": pd.bdate_range("2024-01-01", periods=5),
    })
    scores = compute_allocation_scores(
        strategies, closed_trades, "trending_up",
        pd.DataFrame({"s1": [0.0]}, index=["s1"]),
        {"s1": 10_000_000},
    )
    # Sharpe should be backtest / (1 + dsr_penalty)
    expected_sharpe = 1.5 / (1 + 0.3)
    assert abs(scores[0]["sharpe_component"] - expected_sharpe) < 0.01


# -- 3.2 Budget Computation Tests --


def test_budgets_sum_to_lte_total_equity():
    """Budgets sum to <= total_equity."""
    scores = [
        {"strategy_id": "s1", "score": 1.0},
        {"strategy_id": "s2", "score": 0.5},
        {"strategy_id": "s3", "score": 0.3},
    ]
    budgets = compute_budgets(scores, total_equity=100_000)
    total = sum(b["budget_dollars"] for b in budgets)
    assert total <= 100_000 + 0.01  # floating point tolerance


def test_no_single_strategy_exceeds_25pct():
    """No single strategy budget > 25% of total_equity."""
    scores = [
        {"strategy_id": "s1", "score": 10.0},  # dominant
        {"strategy_id": "s2", "score": 0.1},
        {"strategy_id": "s3", "score": 0.1},
    ]
    budgets = compute_budgets(scores, total_equity=100_000, max_strategy_allocation=0.25)
    for b in budgets:
        assert b["budget_dollars"] <= 25_000 + 0.01


def test_forward_testing_scalar_applied():
    """Forward-testing strategies get FORWARD_TESTING_SIZE_SCALAR applied."""
    scores = [
        {"strategy_id": "s1", "score": 1.0},
        {"strategy_id": "s2", "score": 1.0},
    ]
    budgets = compute_budgets(
        scores, total_equity=100_000,
        forward_testing_scalar=0.5,
        strategy_statuses={"s1": "live", "s2": "forward_testing"},
    )
    b_live = next(b for b in budgets if b["strategy_id"] == "s1")
    b_ft = next(b for b in budgets if b["strategy_id"] == "s2")
    assert b_ft["budget_dollars"] < b_live["budget_dollars"]
    assert abs(b_ft["budget_dollars"] - b_live["budget_dollars"] * 0.5) < 1.0


def test_zero_score_gets_zero_budget():
    """Zero-score strategy gets zero budget."""
    scores = [
        {"strategy_id": "s1", "score": 1.0},
        {"strategy_id": "s2", "score": 0.0},
    ]
    budgets = compute_budgets(scores, total_equity=100_000)
    b_zero = next(b for b in budgets if b["strategy_id"] == "s2")
    assert b_zero["budget_dollars"] == 0.0
    assert b_zero["budget_pct"] == 0.0


def test_all_zero_scores_no_division_error():
    """All strategies with score=0 -> all budgets=0, no ZeroDivisionError."""
    scores = [
        {"strategy_id": "s1", "score": 0.0},
        {"strategy_id": "s2", "score": 0.0},
    ]
    budgets = compute_budgets(scores, total_equity=100_000)
    assert all(b["budget_dollars"] == 0.0 for b in budgets)


# -- 3.3 + 3.4 Budget Remaining and Rebalancing --


def test_budget_remaining_computed():
    """Budget remaining = budget - deployed."""
    remaining = get_strategy_budget_remaining("s1", budget_dollars=10_000, deployed_capital=3_000)
    assert remaining == 7_000


def test_budget_remaining_does_not_go_negative():
    """If deployed > budget (post-rebalance), remaining is 0 not negative."""
    remaining = get_strategy_budget_remaining("s1", budget_dollars=5_000, deployed_capital=8_000)
    assert remaining == 0.0


def test_near_capacity_at_80pct():
    """Strategy is near capacity when deployed >= 80% of budget."""
    remaining = get_strategy_budget_remaining("s1", budget_dollars=10_000, deployed_capital=8_500)
    assert remaining == 1_500
    # The caller checks: deployed / budget >= 0.8 -> near_capacity
    assert 8_500 / 10_000 >= 0.8
