"""Tests for earnings catalyst options integration (Section 08)."""

from __future__ import annotations

import pytest

from quantstack.alpha_discovery.earnings_catalyst_options import (
    recommend_earnings_options_play,
)


def test_low_iv_premium_with_conviction_produces_directional_play():
    """IV < 1.2 + directional conviction -> directional options."""
    result = recommend_earnings_options_play(
        iv_premium_ratio=1.1,
        directional_conviction="bullish",
        sue_history=3.0,
    )
    assert result is not None
    assert result["play_type"] == "directional_calls"
    assert 0 < result["confidence"] <= 1.0


def test_low_iv_bearish_conviction():
    """IV < 1.2 + bearish conviction -> directional puts."""
    result = recommend_earnings_options_play(
        iv_premium_ratio=1.1,
        directional_conviction="bearish",
        sue_history=-2.5,
    )
    assert result is not None
    assert result["play_type"] == "directional_puts"


def test_high_iv_premium_no_conviction_produces_iron_condor():
    """IV > 1.5 + no conviction -> iron condor."""
    result = recommend_earnings_options_play(
        iv_premium_ratio=1.7,
        directional_conviction=None,
        sue_history=None,
    )
    assert result is not None
    assert result["play_type"] == "iron_condor"


def test_ambiguous_iv_premium_produces_no_play():
    """IV 1.2-1.5 -> no play."""
    result = recommend_earnings_options_play(
        iv_premium_ratio=1.35,
        directional_conviction="bullish",
        sue_history=2.0,
    )
    assert result is None


def test_missing_iv_data_returns_no_play():
    """No IV data -> no recommendation."""
    result = recommend_earnings_options_play(
        iv_premium_ratio=0.0,
        directional_conviction=None,
        sue_history=None,
    )
    assert result is None
