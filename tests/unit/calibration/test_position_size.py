"""Tests for position size calibration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantstack.calibration.models import CalibrationResult


def _make_mock_conn(rows):
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchall.return_value = rows
    mock_conn.execute.return_value.fetchone.return_value = None
    return mock_conn


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_calibrate_position_size_with_known_distribution(mock_pg_conn):
    """calibrate_position_size with known loss distribution returns expected threshold."""
    rng = np.random.default_rng(42)
    # Generate 100 trades: ~60% wins, ~40% losses
    n = 100
    equity = 100_000.0
    pnls = []
    for _ in range(n):
        if rng.random() < 0.6:
            pnls.append((rng.uniform(500, 3000), equity))  # wins
        else:
            pnls.append((-rng.uniform(500, 2000), equity))  # losses

    mock_conn = _make_mock_conn(pnls)
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator
    cal = ThresholdCalibrator()
    result = cal.calibrate_position_size()

    assert not result.is_fallback
    assert 0.01 <= result.value <= 0.30
    assert result.sample_size == n


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_calibrate_position_size_empty_returns_fallback(mock_pg_conn):
    """With empty closed_trades returns fallback 0.15."""
    mock_conn = _make_mock_conn([])
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator
    cal = ThresholdCalibrator()
    result = cal.calibrate_position_size()

    assert result.value == 0.15
    assert result.is_fallback is True


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_calibrate_position_size_under_50_returns_fallback(mock_pg_conn):
    """With < 50 trades returns fallback 0.15."""
    rows = [(-500.0, 100_000.0)] * 30
    mock_conn = _make_mock_conn(rows)
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator
    cal = ThresholdCalibrator()
    result = cal.calibrate_position_size()

    assert result.value == 0.15
    assert result.is_fallback is True
    assert "fallback" in result.methodology.lower()


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_calibrated_value_changes_with_distribution(mock_pg_conn):
    """Wider loss distribution => smaller position size."""
    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator

    equity = 100_000.0

    # Tight losses: small losses around -500
    tight = [(float(-500 - i * 10), equity) for i in range(60)]
    tight += [(float(1000 + i * 10), equity) for i in range(40)]

    mock_conn = _make_mock_conn(tight)
    mock_pg_conn.return_value = mock_conn
    cal = ThresholdCalibrator()
    result_tight = cal.calibrate_position_size()

    # Wide losses: large losses around -5000
    wide = [(float(-5000 - i * 100), equity) for i in range(60)]
    wide += [(float(1000 + i * 10), equity) for i in range(40)]

    mock_conn2 = _make_mock_conn(wide)
    mock_pg_conn.return_value = mock_conn2
    result_wide = cal.calibrate_position_size()

    # Wider losses should yield smaller position size
    assert result_wide.value < result_tight.value
