"""Tests for backtest promotion gates calibration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quantstack.calibration.deflated_sharpe import deflated_sharpe_ratio


def test_dsr_matches_expected_formula():
    """Deflated Sharpe ratio computation produces sensible values."""
    # High Sharpe with few strategies tested -> high probability
    dsr_high = deflated_sharpe_ratio(
        observed_sharpe=2.0,
        num_strategies_tested=5,
        num_returns=252,
        skewness=0.0,
        kurtosis=3.0,
    )
    assert 0.0 <= dsr_high <= 1.0
    assert dsr_high > 0.5  # Strong Sharpe should have high probability

    # Low Sharpe with many strategies -> low probability
    dsr_low = deflated_sharpe_ratio(
        observed_sharpe=0.3,
        num_strategies_tested=100,
        num_returns=252,
        skewness=0.0,
        kurtosis=3.0,
    )
    assert 0.0 <= dsr_low <= 1.0
    assert dsr_low < dsr_high


def test_dsr_increases_with_more_testing():
    """DSR adjustment increases (probability decreases) with more strategies tested."""
    dsr_5 = deflated_sharpe_ratio(
        observed_sharpe=1.0, num_strategies_tested=5,
        num_returns=252, skewness=0.0, kurtosis=3.0,
    )
    dsr_50 = deflated_sharpe_ratio(
        observed_sharpe=1.0, num_strategies_tested=50,
        num_returns=252, skewness=0.0, kurtosis=3.0,
    )
    # More testing -> lower probability (harder to beat luck)
    assert dsr_50 < dsr_5


def test_dsr_edge_cases():
    """DSR handles edge cases without errors."""
    assert deflated_sharpe_ratio(0.0, 0, 0) == 0.0
    assert deflated_sharpe_ratio(0.0, 1, 1) == 0.0
    result = deflated_sharpe_ratio(1.0, 1, 252)
    assert 0.0 <= result <= 1.0


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_backtest_gates_under_20_fallback(mock_pg_conn):
    """With < 20 strategies returns fallback Sharpe > 0.5."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchall.return_value = [(1.0, 0.0, 3.0, 252, 0.5)] * 10
    mock_conn.execute.return_value.fetchone.return_value = (10,)
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator
    cal = ThresholdCalibrator()
    result = cal.calibrate_backtest_gates()

    assert result.value == 0.5
    assert result.is_fallback is True
