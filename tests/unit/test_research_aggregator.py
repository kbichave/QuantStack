# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ResearchAggregator — aggregates parallel agent results.
"""

import pytest
from quantstack.research.agent_aggregator import AgentResult, ResearchAggregator


def test_aggregate_complete_coverage():
    """Test aggregation when all 3 domains complete for a symbol."""
    results = [
        AgentResult(
            symbol="AAPL",
            domain="investment",
            status="success",
            strategies_registered=["aapl_investment_value"],
            models_trained=[],
            hypotheses_tested=2,
            breakthrough_features=["piotroski_f_score"],
            thesis_status="intact",
            thesis_summary="Undervalued quality growth",
            conflicts=[],
            elapsed_seconds=120.0
        ),
        AgentResult(
            symbol="AAPL",
            domain="swing",
            status="success",
            strategies_registered=["aapl_swing_momentum"],
            models_trained=["aapl_swing_lgbm"],
            hypotheses_tested=1,
            breakthrough_features=["rsi_divergence"],
            thesis_status="intact",
            thesis_summary="Bullish momentum",
            conflicts=[],
            elapsed_seconds=90.0
        ),
        AgentResult(
            symbol="AAPL",
            domain="options",
            status="success",
            strategies_registered=["aapl_options_call_spread"],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=["iv_rank"],
            thesis_status="intact",
            thesis_summary="IV compression + directional",
            conflicts=[],
            elapsed_seconds=100.0
        ),
    ]

    agg = ResearchAggregator()
    summary = agg.aggregate(results)

    assert "AAPL" in summary["symbols_complete"]
    assert len(summary["symbols_partial"]) == 0
    assert summary["total_strategies"] == 3
    assert summary["total_models"] == 1
    assert summary["total_hypotheses"] == 4
    assert summary["agents_spawned"] == 3
    assert summary["agents_succeeded"] == 3


def test_aggregate_partial_coverage():
    """Test aggregation when only 2 of 3 domains complete."""
    results = [
        AgentResult(
            symbol="TSLA",
            domain="investment",
            status="success",
            strategies_registered=["tsla_investment"],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=[],
            thesis_status="intact",
            thesis_summary="Growth story",
            conflicts=[],
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="TSLA",
            domain="swing",
            status="failure",
            strategies_registered=[],
            models_trained=[],
            hypotheses_tested=0,
            breakthrough_features=[],
            thesis_status="unknown",
            thesis_summary="",
            conflicts=["Insufficient data"],
            elapsed_seconds=30.0
        ),
        AgentResult(
            symbol="TSLA",
            domain="options",
            status="success",
            strategies_registered=["tsla_options_straddle"],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=["iv_percentile"],
            thesis_status="intact",
            thesis_summary="High vol regime",
            conflicts=[],
            elapsed_seconds=80.0
        ),
    ]

    agg = ResearchAggregator()
    summary = agg.aggregate(results)

    assert "TSLA" in summary["symbols_partial"]
    assert "TSLA" not in summary["symbols_complete"]
    assert summary["total_strategies"] == 2
    assert summary["total_hypotheses"] == 2
    assert summary["agents_spawned"] == 3
    assert summary["agents_succeeded"] == 2


def test_detect_thesis_conflict():
    """Test detection of cross-domain thesis conflicts."""
    results = [
        AgentResult(
            symbol="NVDA",
            domain="investment",
            status="success",
            strategies_registered=[],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=[],
            thesis_status="intact",
            thesis_summary="Bullish fundamentals",
            conflicts=[],
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="NVDA",
            domain="swing",
            status="success",
            strategies_registered=[],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=[],
            thesis_status="broken",
            thesis_summary="Bearish technicals",
            conflicts=[],
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="NVDA",
            domain="options",
            status="success",
            strategies_registered=[],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=[],
            thesis_status="weakening",
            thesis_summary="Vol spike",
            conflicts=[],
            elapsed_seconds=60.0
        ),
    ]

    agg = ResearchAggregator()
    summary = agg.aggregate(results)

    assert len(summary["conflicts"]) > 0
    assert any("NVDA" in conflict for conflict in summary["conflicts"])


def test_breakthrough_features_cross_domain():
    """Test identification of features appearing in multiple domains."""
    results = [
        AgentResult(
            symbol="MSFT",
            domain="investment",
            status="success",
            strategies_registered=[],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=["volume_spike", "institutional_flow"],
            thesis_status="intact",
            thesis_summary="",
            conflicts=[],
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="MSFT",
            domain="swing",
            status="success",
            strategies_registered=[],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=["volume_spike", "rsi_divergence"],
            thesis_status="intact",
            thesis_summary="",
            conflicts=[],
            elapsed_seconds=60.0
        ),
        AgentResult(
            symbol="GOOG",
            domain="options",
            status="success",
            strategies_registered=[],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=["volume_spike"],
            thesis_status="intact",
            thesis_summary="",
            conflicts=[],
            elapsed_seconds=60.0
        ),
    ]

    agg = ResearchAggregator()
    summary = agg.aggregate(results)

    # volume_spike appears in 3 results (2+ threshold)
    assert "volume_spike" in summary["breakthrough_features"]
    # institutional_flow and rsi_divergence only appear once
    assert "institutional_flow" not in summary["breakthrough_features"]
    assert "rsi_divergence" not in summary["breakthrough_features"]


def test_domain_coverage_matrix():
    """Test that domain coverage matrix accurately reflects agent results."""
    results = [
        AgentResult(symbol="AAPL", domain="investment", status="success", elapsed_seconds=60.0),
        AgentResult(symbol="AAPL", domain="swing", status="locked", elapsed_seconds=0.0),
        AgentResult(symbol="AAPL", domain="options", status="needs_more_data", elapsed_seconds=30.0),
        AgentResult(symbol="TSLA", domain="investment", status="success", elapsed_seconds=60.0),
        AgentResult(symbol="TSLA", domain="swing", status="success", elapsed_seconds=60.0),
    ]

    agg = ResearchAggregator()
    summary = agg.aggregate(results)

    coverage = summary["domain_coverage"]
    assert coverage["AAPL"]["investment"] == "success"
    assert coverage["AAPL"]["swing"] == "locked"
    assert coverage["AAPL"]["options"] == "needs_more_data"
    assert coverage["TSLA"]["investment"] == "success"
    assert coverage["TSLA"]["swing"] == "success"
    assert "options" not in coverage["TSLA"]


def test_domain_success_rate():
    """Test calculation of success rate per domain."""
    results = [
        AgentResult(symbol="AAPL", domain="investment", status="success", elapsed_seconds=60.0),
        AgentResult(symbol="TSLA", domain="investment", status="success", elapsed_seconds=60.0),
        AgentResult(symbol="NVDA", domain="investment", status="failure", elapsed_seconds=30.0),
        AgentResult(symbol="AAPL", domain="swing", status="success", elapsed_seconds=60.0),
        AgentResult(symbol="TSLA", domain="swing", status="locked", elapsed_seconds=0.0),
    ]

    agg = ResearchAggregator()
    summary = agg.aggregate(results)

    # Investment: 2 success out of 3 = 66.7%
    assert abs(summary["domain_success_rate"]["investment"] - 0.667) < 0.01
    # Swing: 1 success out of 2 = 50%
    assert summary["domain_success_rate"]["swing"] == 0.5
    # Options: no results = 0%
    assert summary["domain_success_rate"]["options"] == 0.0


def test_format_summary():
    """Test formatted summary output."""
    results = [
        AgentResult(
            symbol="AAPL",
            domain="investment",
            status="success",
            strategies_registered=["s1"],
            models_trained=["m1"],
            hypotheses_tested=2,
            breakthrough_features=["feature_a", "feature_b"],
            thesis_status="intact",
            thesis_summary="",
            conflicts=[],
            elapsed_seconds=120.0
        ),
        AgentResult(
            symbol="AAPL",
            domain="swing",
            status="success",
            strategies_registered=["s2"],
            models_trained=[],
            hypotheses_tested=1,
            breakthrough_features=["feature_a"],
            thesis_status="intact",
            thesis_summary="",
            conflicts=[],
            elapsed_seconds=90.0
        ),
    ]

    agg = ResearchAggregator()
    summary = agg.aggregate(results)
    formatted = agg.format_summary(summary)

    assert "BLITZ Mode Research Summary" in formatted
    assert "2 spawned" in formatted
    assert "2 succeeded" in formatted
    assert "AAPL" in formatted
    assert "2 strategies" in formatted
    assert "feature_a" in formatted  # Appears 2+ times


def test_empty_results():
    """Test aggregation with no results."""
    agg = ResearchAggregator()
    summary = agg.aggregate([])

    assert len(summary["symbols_complete"]) == 0
    assert len(summary["symbols_partial"]) == 0
    assert summary["total_strategies"] == 0
    assert summary["total_models"] == 0
    assert summary["total_hypotheses"] == 0
    assert len(summary["breakthrough_features"]) == 0
    assert len(summary["conflicts"]) == 0
    assert summary["agents_spawned"] == 0
    assert summary["agents_succeeded"] == 0
