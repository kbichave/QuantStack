"""Tests for the monthly threshold tuner."""

from __future__ import annotations

import yaml

from quantstack.meta.config import (
    _THRESHOLDS,
    _YAML_PATH,
    _load_thresholds,
    get_threshold,
    get_threshold_bounds,
    set_threshold,
)
from quantstack.meta.threshold_tuner import tune_threshold


def _reset_thresholds() -> None:
    """Reload thresholds from disk so tests are independent."""
    fresh = _load_thresholds()
    _THRESHOLDS.clear()
    _THRESHOLDS.update(fresh)


def test_thresholds_read_from_yaml():
    data = yaml.safe_load(_YAML_PATH.read_text())
    assert "hypothesis_critique" in data
    assert data["hypothesis_critique"]["value"] == 0.7
    assert data["backtest_sharpe"]["floor"] == 0.2


def test_threshold_lowered_on_high_false_rejection():
    _reset_thresholds()
    before = get_threshold("hypothesis_critique")
    result = tune_threshold("hypothesis_critique", false_rejection_rate=0.25, false_acceptance_rate=0.0)
    assert result is not None
    assert result < before
    _reset_thresholds()


def test_threshold_raised_on_high_false_acceptance():
    _reset_thresholds()
    before = get_threshold("hypothesis_critique")
    result = tune_threshold("hypothesis_critique", false_rejection_rate=0.0, false_acceptance_rate=0.35)
    assert result is not None
    assert result > before
    _reset_thresholds()


def test_threshold_never_below_floor():
    _reset_thresholds()
    floor, _ = get_threshold_bounds("oos_ic")
    # Force value to the floor, then try to lower further.
    set_threshold("oos_ic", floor)
    result = tune_threshold("oos_ic", false_rejection_rate=0.99, false_acceptance_rate=0.0)
    # Should not go below floor.
    assert result is None or result >= floor
    _reset_thresholds()


def test_threshold_never_above_ceiling():
    _reset_thresholds()
    _, ceiling = get_threshold_bounds("oos_ic")
    set_threshold("oos_ic", ceiling)
    result = tune_threshold("oos_ic", false_rejection_rate=0.0, false_acceptance_rate=0.99)
    assert result is None or result <= ceiling
    _reset_thresholds()


def test_threshold_unchanged_when_rates_acceptable():
    _reset_thresholds()
    result = tune_threshold("backtest_sharpe", false_rejection_rate=0.10, false_acceptance_rate=0.10)
    assert result is None
    _reset_thresholds()
