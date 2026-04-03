"""Tests for slippage budget pre-trade check."""

from __future__ import annotations

import pytest

from quantstack.core.execution.slippage_budget import (
    SlippageBudgetResult,
    check_slippage_budget,
)


def test_expected_slippage_from_history():
    """Expected slippage computed from historical mean shortfall."""
    # Provide historical data directly
    result = check_slippage_budget(
        expected_alpha_bps=200.0,
        historical_mean_shortfall_bps=4.2,
    )
    assert abs(result.expected_slippage_bps - 4.2) < 0.01
    assert result.source == "historical"


def test_trade_ok_when_slippage_under_1pct():
    """status="ok" when slippage < 1% of expected alpha."""
    result = check_slippage_budget(
        expected_alpha_bps=200.0,
        historical_mean_shortfall_bps=1.0,  # 0.5% of alpha
    )
    assert result.status == "ok"


def test_trade_flagged_when_slippage_1_to_2pct():
    """status="flagged" when slippage is 1-2% of expected alpha."""
    result = check_slippage_budget(
        expected_alpha_bps=200.0,
        historical_mean_shortfall_bps=3.0,  # 1.5% of alpha
    )
    assert result.status == "flagged"


def test_trade_rejected_when_slippage_over_2pct():
    """status="rejected" when slippage > 2% of expected alpha."""
    result = check_slippage_budget(
        expected_alpha_bps=100.0,
        historical_mean_shortfall_bps=3.0,  # 3% of alpha
    )
    assert result.status == "rejected"


def test_no_history_uses_default():
    """When no historical data, use global default."""
    result = check_slippage_budget(
        expected_alpha_bps=200.0,
        historical_mean_shortfall_bps=None,
        global_default_bps=2.0,
    )
    assert result.source == "default"
    assert result.expected_slippage_bps == 2.0
    assert result.status == "ok"  # 2/200 = 1% -> ok


def test_zero_alpha_never_rejects():
    """With zero alpha, status should be flagged (not divide-by-zero)."""
    result = check_slippage_budget(
        expected_alpha_bps=0.0,
        historical_mean_shortfall_bps=5.0,
    )
    assert result.status == "flagged"
