"""Tests for regime flip detection and response (section 13).

Validates:
  - Severity classification (severe, moderate, None)
  - Stop tightening math with floor enforcement
  - Action generation for severe (auto-exit) and moderate (tighten stops)
  - regime_at_entry field on MonitoredPosition
  - None stop_price handling
"""
from __future__ import annotations

import pytest

from quantstack.execution.execution_monitor import MonitoredPosition
from quantstack.execution.regime_flip import (
    classify_regime_flip,
    compute_tightened_stop,
    generate_regime_flip_actions,
)
from quantstack.holding_period import HoldingType


# ── classify_regime_flip ─────────────────────────────────────────────────


class TestClassifyRegimeFlip:
    """Severity classification for regime transitions."""

    def test_same_regime_returns_none(self):
        assert classify_regime_flip("trending_up", "trending_up") is None

    def test_unknown_entry_returns_none(self):
        """Can't assess flip from unknown entry regime."""
        assert classify_regime_flip("unknown", "trending_down") is None

    def test_trending_up_to_down_is_severe(self):
        assert classify_regime_flip("trending_up", "trending_down") == "severe"

    def test_trending_down_to_up_is_severe(self):
        assert classify_regime_flip("trending_down", "trending_up") == "severe"

    def test_trending_to_ranging_is_moderate(self):
        assert classify_regime_flip("trending_up", "ranging") == "moderate"

    def test_ranging_to_trending_is_moderate(self):
        assert classify_regime_flip("ranging", "trending_up") == "moderate"

    def test_to_unknown_is_moderate(self):
        """Transition to unknown from known regime is moderate."""
        assert classify_regime_flip("trending_up", "unknown") == "moderate"

    def test_ranging_to_ranging_returns_none(self):
        assert classify_regime_flip("ranging", "ranging") is None


# ── compute_tightened_stop ───────────────────────────────────────────────


class TestComputeTightenedStop:
    """Stop tightening math with floor enforcement."""

    def test_long_halves_distance(self):
        """Long: stop distance halved from 10 to 5."""
        # price=100, stop=90 → distance=10 → new_distance=5 → new_stop=95
        # floor = max(2*1.0, 0.01*100) = 2.0, so 5 > 2 → use 5
        result = compute_tightened_stop(100.0, 90.0, 1.0, "long")
        assert result == pytest.approx(95.0)

    def test_short_halves_distance(self):
        """Short: stop distance halved."""
        # price=100, stop=110 → distance=10 → new=5 → new_stop=105
        result = compute_tightened_stop(100.0, 110.0, 1.0, "short")
        assert result == pytest.approx(105.0)

    def test_floor_enforcement_atr(self):
        """Floor prevents tightening below 2*ATR."""
        # price=100, stop=99 → distance=1 → new=0.5 → but floor=max(2*2.0, 1.0)=4.0
        # → new_stop = 100 - 4 = 96
        result = compute_tightened_stop(100.0, 99.0, 2.0, "long")
        assert result == pytest.approx(96.0)

    def test_floor_enforcement_pct(self):
        """Floor prevents tightening below 1% of price."""
        # price=1000, stop=999 → distance=1 → new=0.5 → floor=max(2*0.1, 10.0)=10.0
        # → new_stop = 1000 - 10 = 990
        result = compute_tightened_stop(1000.0, 999.0, 0.1, "long")
        assert result == pytest.approx(990.0)

    def test_none_stop_long_sets_at_floor(self):
        """Long with no existing stop sets stop at floor distance below price."""
        # floor = max(2*3.0, 0.01*100) = 6.0
        result = compute_tightened_stop(100.0, None, 3.0, "long")
        assert result == pytest.approx(94.0)

    def test_none_stop_short_sets_at_floor(self):
        """Short with no existing stop sets stop at floor distance above price."""
        # floor = max(2*3.0, 0.01*100) = 6.0
        result = compute_tightened_stop(100.0, None, 3.0, "short")
        assert result == pytest.approx(106.0)


# ── generate_regime_flip_actions ─────────────────────────────────────────


class TestGenerateRegimeFlipActions:
    """Action generation for regime flips."""

    def test_no_flip_returns_empty(self):
        result = generate_regime_flip_actions(
            symbol="AAPL", side="long", quantity=100,
            entry_regime="trending_up", current_regime="trending_up",
            current_price=150.0, stop_price=145.0, entry_atr=2.0,
        )
        assert result["severity"] is None
        assert result["exit_order"] is None
        assert result["new_stop"] is None

    def test_severe_generates_exit_order(self):
        result = generate_regime_flip_actions(
            symbol="AAPL", side="long", quantity=100,
            entry_regime="trending_up", current_regime="trending_down",
            current_price=150.0, stop_price=145.0, entry_atr=2.0,
        )
        assert result["severity"] == "severe"
        assert result["exit_order"] is not None
        assert result["exit_order"]["symbol"] == "AAPL"
        assert result["exit_order"]["quantity"] == 100
        assert result["exit_order"]["reason"] == "regime_flip_severe"

    def test_severe_with_stop_does_not_add_new_stop(self):
        """Severe flip with existing stop: exit_order generated, no new_stop needed."""
        result = generate_regime_flip_actions(
            symbol="AAPL", side="long", quantity=100,
            entry_regime="trending_up", current_regime="trending_down",
            current_price=150.0, stop_price=145.0, entry_atr=2.0,
        )
        assert result["new_stop"] is None  # stop already exists

    def test_severe_without_stop_sets_backup(self):
        """Severe flip with no stop: exit_order + belt-and-suspenders stop."""
        result = generate_regime_flip_actions(
            symbol="AAPL", side="long", quantity=100,
            entry_regime="trending_up", current_regime="trending_down",
            current_price=150.0, stop_price=None, entry_atr=2.0,
        )
        assert result["exit_order"] is not None
        assert result["new_stop"] is not None
        # floor = max(2*2.0, 0.01*150) = 4.0 → stop at 146.0
        assert result["new_stop"] == pytest.approx(146.0)

    def test_moderate_tightens_stop(self):
        """Moderate flip tightens existing stop."""
        result = generate_regime_flip_actions(
            symbol="AAPL", side="long", quantity=100,
            entry_regime="trending_up", current_regime="ranging",
            current_price=150.0, stop_price=140.0, entry_atr=2.0,
        )
        assert result["severity"] == "moderate"
        assert result["exit_order"] is None
        # distance=10, halved=5, floor=max(4,1.5)=4 → 5 > 4 → stop=145
        assert result["new_stop"] == pytest.approx(145.0)

    def test_moderate_no_stop_sets_new(self):
        """Moderate flip with no existing stop sets one at floor."""
        result = generate_regime_flip_actions(
            symbol="AAPL", side="long", quantity=100,
            entry_regime="trending_up", current_regime="ranging",
            current_price=150.0, stop_price=None, entry_atr=2.0,
        )
        assert result["severity"] == "moderate"
        assert result["new_stop"] is not None


# ── MonitoredPosition regime_at_entry ────────────────────────────────────


class TestMonitoredPositionRegimeField:
    """regime_at_entry field on MonitoredPosition."""

    def test_default_regime_is_unknown(self):
        mp = MonitoredPosition(
            symbol="AAPL", side="long", quantity=100,
            holding_type=HoldingType.SWING,
            entry_price=150.0, entry_time=None,
        )
        assert mp.regime_at_entry == "unknown"

    def test_regime_can_be_set(self):
        mp = MonitoredPosition(
            symbol="AAPL", side="long", quantity=100,
            holding_type=HoldingType.SWING,
            entry_price=150.0, entry_time=None,
            regime_at_entry="trending_up",
        )
        assert mp.regime_at_entry == "trending_up"
