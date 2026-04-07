"""Experiment prioritization formula and queue sorting."""

from quantstack.learning.experiment_prioritizer import compute_priority, prioritize_experiments


def test_high_ic_cheap_ranked_above_low_ic_expensive():
    """An experiment with expected_IC=0.05, cost=5000 tokens ranks above one
    with expected_IC=0.02, cost=30000 tokens."""
    p1 = compute_priority(expected_ic=0.05, regime_fit=0.8, novelty_score=1.0, estimated_compute_cost=5000)
    p2 = compute_priority(expected_ic=0.02, regime_fit=0.8, novelty_score=1.0, estimated_compute_cost=30000)
    assert p1 > p2


def test_novelty_score_affects_priority():
    """A novel hypothesis (1.0) ranks above a known one (0.1), all else equal."""
    p_novel = compute_priority(expected_ic=0.03, regime_fit=0.7, novelty_score=1.0, estimated_compute_cost=10000)
    p_known = compute_priority(expected_ic=0.03, regime_fit=0.7, novelty_score=0.1, estimated_compute_cost=10000)
    assert p_novel > p_known


def test_cold_start_all_novel():
    """When novelty_score=1.0 for all, ranking is by regime_fit / compute_cost."""
    exps = [
        {"hypothesis": "A", "complexity": "simple", "expected_ic": 0.03},
        {"hypothesis": "B", "complexity": "ml", "expected_ic": 0.05},
    ]
    ranked = prioritize_experiments(exps, current_regime="trending_up")
    # All get novelty=1.0; B has higher IC but higher cost (ml=30k vs simple=5k)
    # B priority: 0.05 * 1.0 * 1.0 / 30000 = 1.67e-6
    # A priority: 0.03 * 1.0 * 1.0 / 5000 = 6.0e-6
    # A should rank above B due to better value per token
    assert ranked[0]["hypothesis"] == "A"


def test_prioritizer_empty_queue():
    """prioritize_experiments returns an empty list when given an empty queue."""
    assert prioritize_experiments([], current_regime="ranging") == []


def test_zero_cost_handled():
    """Zero compute cost doesn't cause division by zero."""
    p = compute_priority(expected_ic=0.03, regime_fit=0.8, novelty_score=1.0, estimated_compute_cost=0)
    assert p > 0  # Should use a minimum cost floor
