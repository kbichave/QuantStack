# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Live vs Backtest Sharpe Demotion (Section 12).
"""

from __future__ import annotations

import math
import os

import pytest

from quantstack.learning.sharpe_demotion import check_sharpe_demotion, compute_live_sharpe


class TestLiveSharpe:
    """Test rolling 21-day Sharpe computation."""

    def test_rolling_21_day_sharpe(self):
        """Verify annualized Sharpe matches hand-computed value."""
        # Simple test: 21 days of 1% daily return
        returns = [0.01] * 21
        sharpe = compute_live_sharpe(returns)

        # Hand compute:
        # mean = 0.01
        # std = 0.0 (all values identical)
        # This should return inf since mean > 0 and std = 0
        assert sharpe == float('inf')

        # More realistic: mix of positive and negative
        returns_mixed = [0.01, -0.005, 0.02, 0.015, -0.01] * 5  # 25 returns (>21)
        returns_mixed = returns_mixed[:21]  # Take exactly 21
        sharpe_mixed = compute_live_sharpe(returns_mixed)

        # Hand compute expected value
        mean = sum(returns_mixed) / len(returns_mixed)
        variance = sum((r - mean) ** 2 for r in returns_mixed) / len(returns_mixed)
        std = math.sqrt(variance)
        expected_sharpe = (mean / std) * math.sqrt(252)

        assert sharpe_mixed is not None
        assert abs(sharpe_mixed - expected_sharpe) < 1e-10

    def test_handles_fewer_than_21_days(self):
        """Returns None when insufficient data."""
        returns = [0.01] * 15  # Only 15 days
        sharpe = compute_live_sharpe(returns)
        assert sharpe is None

        # Edge case: exactly 20 days
        returns_20 = [0.01] * 20
        sharpe_20 = compute_live_sharpe(returns_20)
        assert sharpe_20 is None

        # Edge case: exactly 21 days should work
        returns_21 = [0.01, -0.005] * 10 + [0.01]  # 21 returns
        sharpe_21 = compute_live_sharpe(returns_21)
        assert sharpe_21 is not None

    def test_zero_std_returns_inf(self):
        """All identical positive returns -> std=0 -> inf."""
        returns = [0.01] * 21
        sharpe = compute_live_sharpe(returns)
        assert sharpe == float('inf')

        # Negative returns with std=0 -> -inf
        returns_neg = [-0.01] * 21
        sharpe_neg = compute_live_sharpe(returns_neg)
        assert sharpe_neg == float('-inf')

        # Zero returns with std=0 -> 0.0
        returns_zero = [0.0] * 21
        sharpe_zero = compute_live_sharpe(returns_zero)
        assert sharpe_zero == 0.0


class TestDemotionGate:
    """Test demotion trigger logic."""

    def test_triggers_when_below_50pct(self, monkeypatch):
        """Backtest=1.5, live=0.6 (40%), 21 days -> triggers."""
        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "true")
        backtest_sharpe = 1.5
        live_sharpe = 0.6  # 40% of backtest
        consecutive_days = 21

        result = check_sharpe_demotion(live_sharpe, backtest_sharpe, consecutive_days)

        assert result is not None
        assert result["triggered"] is True
        assert result["live_sharpe"] == 0.6
        assert result["backtest_sharpe"] == 1.5
        assert result["threshold"] == 0.75  # 50% of 1.5
        assert result["consecutive_days"] == 21

    def test_no_trigger_at_20_days(self, monkeypatch):
        """Same degradation but only 20 days -> no trigger."""
        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "true")
        backtest_sharpe = 1.5
        live_sharpe = 0.6  # 40% of backtest
        consecutive_days = 20

        result = check_sharpe_demotion(live_sharpe, backtest_sharpe, consecutive_days)

        assert result is None

    def test_no_trigger_when_above_50pct(self, monkeypatch):
        """Backtest=1.5, live=0.9 (60%) -> no trigger."""
        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "true")
        backtest_sharpe = 1.5
        live_sharpe = 0.9  # 60% of backtest (above 50% threshold)
        consecutive_days = 21

        result = check_sharpe_demotion(live_sharpe, backtest_sharpe, consecutive_days)

        assert result is None

    def test_boundary_exactly_50pct(self, monkeypatch):
        """Live exactly at 50% threshold -> no trigger."""
        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "true")
        backtest_sharpe = 1.5
        live_sharpe = 0.75  # Exactly 50%
        consecutive_days = 21

        result = check_sharpe_demotion(live_sharpe, backtest_sharpe, consecutive_days)

        assert result is None


class TestConfigFlag:
    """Test FEEDBACK_SHARPE_DEMOTION config flag."""

    def test_flag_false_returns_none(self, monkeypatch):
        """FEEDBACK_SHARPE_DEMOTION=false -> always None."""
        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "false")

        backtest_sharpe = 1.5
        live_sharpe = 0.6  # Would normally trigger
        consecutive_days = 21

        result = check_sharpe_demotion(live_sharpe, backtest_sharpe, consecutive_days)

        assert result is None

    def test_flag_true_allows_trigger(self, monkeypatch):
        """FEEDBACK_SHARPE_DEMOTION=true -> normal behavior."""
        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "true")

        backtest_sharpe = 1.5
        live_sharpe = 0.6
        consecutive_days = 21

        result = check_sharpe_demotion(live_sharpe, backtest_sharpe, consecutive_days)

        assert result is not None
        assert result["triggered"] is True

    def test_flag_default_is_false(self, monkeypatch):
        """Default (unset) -> treated as false (safe-off)."""
        monkeypatch.delenv("FEEDBACK_SHARPE_DEMOTION", raising=False)

        backtest_sharpe = 1.5
        live_sharpe = 0.6
        consecutive_days = 21

        result = check_sharpe_demotion(live_sharpe, backtest_sharpe, consecutive_days)

        assert result is None


class TestColdStart:
    """Test cold start behavior (insufficient live data)."""

    def test_fewer_than_21_days_skips(self):
        """live_sharpe=None (cold start) -> no trigger."""
        backtest_sharpe = 1.5
        live_sharpe = None  # Insufficient data
        consecutive_days = 25  # Even with many days

        result = check_sharpe_demotion(live_sharpe, backtest_sharpe, consecutive_days)

        assert result is None
