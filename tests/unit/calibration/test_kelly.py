"""Tests for Kelly fraction calibration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantstack.calibration.monte_carlo import compute_max_drawdowns, simulate_paths


def test_simulate_paths_produces_realistic_curves():
    """Monte Carlo simulation produces realistic equity curves."""
    rng = np.random.default_rng(42)
    daily_returns = rng.normal(0.001, 0.015, size=252)

    paths = simulate_paths(daily_returns, n_paths=1000, n_days=252, kelly_fraction=0.5)

    assert paths.shape == (1000, 252)
    assert np.all(np.isfinite(paths))
    # Starting equity should be close to 1.0
    assert np.allclose(paths[:, 0], 1.0 + daily_returns[0] * 0.5, atol=0.1)


def test_max_drawdowns_computed_correctly():
    """compute_max_drawdowns returns correct values."""
    # Simple case: equity goes up then down
    equity = np.array([[1.0, 1.1, 1.2, 1.0, 0.9]])
    dds = compute_max_drawdowns(equity)
    assert len(dds) == 1
    # Max DD = (1.2 - 0.9) / 1.2 = 0.25
    assert abs(dds[0] - 0.25) < 0.01


def test_high_win_rate_yields_higher_kelly():
    """Optimal fraction with high win rate > optimal fraction with low win rate."""
    rng = np.random.default_rng(42)

    # High win rate
    high_wr = np.where(rng.random(200) < 0.65, 0.02, -0.015)
    # Low win rate
    low_wr = np.where(rng.random(200) < 0.45, 0.02, -0.015)

    # Find max fraction where DD constraint holds for each
    def find_optimal(returns):
        best = 0.1
        for f in np.arange(0.1, 1.05, 0.05):
            p = simulate_paths(returns, n_paths=2000, n_days=252, kelly_fraction=f)
            dds = compute_max_drawdowns(p)
            if np.mean(dds > 0.15) <= 0.05:
                best = f
        return best

    optimal_high = find_optimal(high_wr)
    optimal_low = find_optimal(low_wr)

    assert optimal_high >= optimal_low


@patch("quantstack.calibration.threshold_calibrator.pg_conn")
def test_kelly_under_100_trades_fallback(mock_pg_conn):
    """With < 100 trades returns fallback 0.5 (half-Kelly)."""
    rows = [(500.0, 100_000.0)] * 50
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchall.return_value = rows
    mock_conn.execute.return_value.fetchone.return_value = None
    mock_pg_conn.return_value = mock_conn

    from quantstack.calibration.threshold_calibrator import ThresholdCalibrator
    cal = ThresholdCalibrator()
    result = cal.calibrate_kelly()

    assert result.value == 0.5
    assert result.is_fallback is True
    assert "fallback" in result.methodology.lower()


def test_simulate_paths_empty_returns():
    """simulate_paths handles empty returns array."""
    paths = simulate_paths(np.array([]), n_paths=10, n_days=10)
    assert paths.shape == (10, 10)
    assert np.all(paths == 1.0)
