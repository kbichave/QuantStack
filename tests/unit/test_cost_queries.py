"""Tests for Section 03: Per-Agent Cost Tracking.

Validates:
1. AgentConfig accepts max_tokens_budget field
2. Budget enforcement tracks tokens and stops agents
3. Cost aggregation query functions
4. Langfuse metadata enrichment
"""

import pytest


# ---------------------------------------------------------------------------
# AgentConfig budget field
# ---------------------------------------------------------------------------


class TestAgentConfigBudget:
    """max_tokens_budget field in AgentConfig."""

    def test_accepts_budget_field(self):
        from quantstack.graphs.config import AgentConfig
        cfg = AgentConfig(
            name="test_agent",
            role="test",
            goal="test",
            backstory="test",
            llm_tier="heavy",
            max_tokens_budget=10000,
        )
        assert cfg.max_tokens_budget == 10000

    def test_budget_defaults_to_none(self):
        from quantstack.graphs.config import AgentConfig
        cfg = AgentConfig(
            name="test_agent",
            role="test",
            goal="test",
            backstory="test",
            llm_tier="heavy",
        )
        assert cfg.max_tokens_budget is None

    def test_budget_none_means_no_limit(self):
        from quantstack.graphs.config import AgentConfig
        cfg = AgentConfig(
            name="test_agent",
            role="test",
            goal="test",
            backstory="test",
            llm_tier="heavy",
            max_tokens_budget=None,
        )
        assert cfg.max_tokens_budget is None


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    """Token budget accumulation and enforcement."""

    def test_accumulates_tokens(self):
        from quantstack.observability.cost_queries import TokenBudgetTracker
        tracker = TokenBudgetTracker(max_tokens=5000)
        tracker.add_usage(input_tokens=100, output_tokens=50)
        assert tracker.total_tokens == 150

    def test_accumulates_across_rounds(self):
        from quantstack.observability.cost_queries import TokenBudgetTracker
        tracker = TokenBudgetTracker(max_tokens=5000)
        tracker.add_usage(input_tokens=100, output_tokens=50)
        tracker.add_usage(input_tokens=200, output_tokens=100)
        assert tracker.total_tokens == 450

    def test_exceeds_budget(self):
        from quantstack.observability.cost_queries import TokenBudgetTracker
        tracker = TokenBudgetTracker(max_tokens=1000)
        tracker.add_usage(input_tokens=600, output_tokens=500)
        assert tracker.budget_exceeded is True

    def test_under_budget(self):
        from quantstack.observability.cost_queries import TokenBudgetTracker
        tracker = TokenBudgetTracker(max_tokens=5000)
        tracker.add_usage(input_tokens=100, output_tokens=50)
        assert tracker.budget_exceeded is False

    def test_no_budget_never_exceeds(self):
        from quantstack.observability.cost_queries import TokenBudgetTracker
        tracker = TokenBudgetTracker(max_tokens=None)
        tracker.add_usage(input_tokens=100000, output_tokens=50000)
        assert tracker.budget_exceeded is False

    def test_tracks_prompt_and_completion(self):
        from quantstack.observability.cost_queries import TokenBudgetTracker
        tracker = TokenBudgetTracker(max_tokens=5000)
        tracker.add_usage(input_tokens=100, output_tokens=50)
        assert tracker.input_tokens == 100
        assert tracker.output_tokens == 50


# ---------------------------------------------------------------------------
# Cost aggregation queries
# ---------------------------------------------------------------------------


class TestCostAggregation:
    """Cost aggregation functions (compute from data, no Langfuse API needed)."""

    def test_compute_cost_from_usage(self):
        from quantstack.observability.cost_queries import compute_cost_usd
        # Sonnet pricing: $3/MTok input, $15/MTok output
        cost = compute_cost_usd(
            input_tokens=1000, output_tokens=500,
            model="anthropic/claude-sonnet-4-6",
        )
        assert cost > 0
        assert isinstance(cost, float)

    def test_compute_cost_zero_tokens(self):
        from quantstack.observability.cost_queries import compute_cost_usd
        cost = compute_cost_usd(input_tokens=0, output_tokens=0, model="anthropic/claude-sonnet-4-6")
        assert cost == 0.0

    def test_compute_cost_unknown_model_uses_default(self):
        from quantstack.observability.cost_queries import compute_cost_usd
        cost = compute_cost_usd(input_tokens=1000, output_tokens=500, model="unknown/model")
        assert cost > 0  # uses default pricing, doesn't crash

    def test_detect_anomaly_exceeds_threshold(self):
        from quantstack.observability.cost_queries import detect_cost_anomaly
        # 7-day average: $1/day, today: $5 → 5x > 3x threshold
        assert detect_cost_anomaly(
            current_cost=5.0, baseline_avg=1.0, threshold=3.0,
        ) is True

    def test_detect_anomaly_within_threshold(self):
        from quantstack.observability.cost_queries import detect_cost_anomaly
        assert detect_cost_anomaly(
            current_cost=2.0, baseline_avg=1.0, threshold=3.0,
        ) is False

    def test_detect_anomaly_zero_baseline(self):
        from quantstack.observability.cost_queries import detect_cost_anomaly
        # Zero baseline: any cost is anomalous (if > 0)
        assert detect_cost_anomaly(
            current_cost=1.0, baseline_avg=0.0, threshold=3.0,
        ) is True
