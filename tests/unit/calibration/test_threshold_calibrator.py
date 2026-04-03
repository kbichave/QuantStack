"""Tests for the core ThresholdCalibrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quantstack.calibration.models import CalibrationResult


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_calibration_result_has_all_fields(mock_pg_conn):
    """Each calibration function returns a CalibrationResult with all fields."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = None
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator

    cal = ThresholdCalibrator()
    result = cal.calibrate_position_size()

    assert isinstance(result, CalibrationResult)
    assert isinstance(result.threshold_name, str)
    assert isinstance(result.value, float)
    assert isinstance(result.confidence_interval, tuple)
    assert len(result.confidence_interval) == 2
    assert isinstance(result.sample_size, int)
    assert isinstance(result.methodology, str)
    assert isinstance(result.is_fallback, bool)


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_calibration_with_insufficient_data_returns_fallback(mock_pg_conn):
    """calibration with insufficient data returns fallback (not error)."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = None
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator

    cal = ThresholdCalibrator()

    for method in [
        cal.calibrate_position_size,
        cal.calibrate_daily_halt,
        cal.calibrate_signal_validation,
        cal.calibrate_backtest_gates,
        cal.calibrate_kelly,
    ]:
        result = method()
        assert result.is_fallback is True
        assert "fallback" in result.methodology.lower() or "insufficient" in result.methodology.lower()
