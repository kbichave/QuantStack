"""Tests for signal validation threshold calibration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_mock_conn(rows, count_row=None):
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchall.return_value = rows
    mock_conn.execute.return_value.fetchone.return_value = count_row
    return mock_conn


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_roc_curve_finds_threshold(mock_pg_conn):
    """ROC curve computation from strategy IC vs OOS outcomes selects threshold."""
    # Create 50 strategies: high IC -> profitable, low IC -> unprofitable
    rows = []
    for i in range(50):
        ic = 0.01 + i * 0.001  # 0.01 to 0.06
        oos_sharpe = 0.5 if ic > 0.03 else -0.2  # profitable above IC=0.03
        rows.append((ic, oos_sharpe))

    mock_conn = _make_mock_conn(rows)
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator
    cal = ThresholdCalibrator()
    result = cal.calibrate_signal_validation()

    assert not result.is_fallback
    assert result.sample_size == 50


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_signal_validation_under_30_fallback(mock_pg_conn):
    """With < 30 strategies returns fallback IC > 0.02."""
    rows = [(0.03, 0.5)] * 15
    mock_conn = _make_mock_conn(rows)
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator
    cal = ThresholdCalibrator()
    result = cal.calibrate_signal_validation()

    assert result.value == 0.02
    assert result.is_fallback is True
