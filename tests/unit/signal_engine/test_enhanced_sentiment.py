"""Tests for enhanced sentiment aggregation (Section 09)."""

from __future__ import annotations

import pytest

from quantstack.signal_engine.collectors.enhanced_sentiment import (
    compute_enhanced_sentiment,
)


def test_multi_source_aggregation_with_correct_weights():
    """Weighted: analyst=0.5, news=0.3, social=0.2."""
    result = compute_enhanced_sentiment(
        analyst_score=0.8,
        news_score=0.6,
        social_score=0.4,
    )
    # 0.8*0.5 + 0.6*0.3 + 0.4*0.2 = 0.40 + 0.18 + 0.08 = 0.66
    assert abs(result["composite_sentiment"] - 0.66) < 0.01


def test_time_decay_recent_weighted_more():
    """Recent analyst signal outweighs old one when mixing with news."""
    # Analyst=0.8 (recent) + news=0.4 -> analyst dominates more with fresh decay
    recent = compute_enhanced_sentiment(
        analyst_score=0.8,
        news_score=0.4,
        social_score=None,
        analyst_age_hours=6,
        news_age_hours=12,
    )
    # Analyst=0.8 (old) + news=0.4 -> analyst weight reduced by decay
    old = compute_enhanced_sentiment(
        analyst_score=0.8,
        news_score=0.4,
        social_score=None,
        analyst_age_hours=120,
        news_age_hours=12,
    )
    # Recent analyst contributes more -> composite closer to 0.8
    assert recent["composite_sentiment"] > old["composite_sentiment"]


def test_confidence_high_when_sources_agree():
    """3 sources all bullish -> high confidence, agreement_rate near 1."""
    result = compute_enhanced_sentiment(
        analyst_score=0.8,
        news_score=0.7,
        social_score=0.75,
    )
    assert result["source_count"] == 3
    assert result["agreement_rate"] > 0.8
    assert result["confidence"] >= 0.8


def test_confidence_lower_when_sources_disagree():
    """Mixed sources -> lower agreement and confidence."""
    result = compute_enhanced_sentiment(
        analyst_score=0.9,
        news_score=0.3,
        social_score=0.5,
    )
    assert result["agreement_rate"] < 0.8
    assert result["confidence"] < 0.8


def test_missing_sources_handled():
    """Only 1 source -> still valid, low confidence."""
    result = compute_enhanced_sentiment(
        analyst_score=None,
        news_score=0.7,
        social_score=None,
    )
    assert result["source_count"] == 1
    assert result["confidence"] == 0.3
    assert 0 <= result["composite_sentiment"] <= 1.0


def test_empty_inputs_neutral():
    """No sources -> neutral with zero confidence."""
    result = compute_enhanced_sentiment(
        analyst_score=None,
        news_score=None,
        social_score=None,
    )
    assert result["composite_sentiment"] == 0.5
    assert result["confidence"] == 0.0
    assert result["source_count"] == 0
    assert result["dominant_direction"] == "neutral"
