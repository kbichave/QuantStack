"""Tests for daily loss halt calibration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_conn(rows):
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchall.return_value = rows
    mock_conn.execute.return_value.fetchone.return_value = None
    return mock_conn


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_calibrate_daily_halt_with_data(mock_pg_conn):
    """calibrate_daily_halt with known P&L distribution returns a threshold."""
    rng = np.random.default_rng(42)
    daily_pnls = [(float(rng.normal(0.001, 0.01)),) for _ in range(120)]

    mock_conn = _make_mock_conn(daily_pnls)
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator
    cal = ThresholdCalibrator()
    result = cal.calibrate_daily_halt()

    assert not result.is_fallback
    assert 0.01 <= result.value <= 0.10
    assert result.sample_size == 120


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_calibrate_daily_halt_under_60_days_fallback(mock_pg_conn):
    """With < 60 days of data returns fallback 0.03."""
    daily_pnls = [(0.005,)] * 30
    mock_conn = _make_mock_conn(daily_pnls)
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator
    cal = ThresholdCalibrator()
    result = cal.calibrate_daily_halt()

    assert result.value == 0.03
    assert result.is_fallback is True
    assert "fallback" in result.methodology.lower()
