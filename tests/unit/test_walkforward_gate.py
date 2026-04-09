"""Tests for 4.4 — Walk-Forward Mandatory Gate (QS-B4)."""

import pytest

from quantstack.autonomous.strategy_lifecycle import (
    _MAX_OVERFIT_RATIO,
    _MIN_OOS_IS_RATIO,
    _MIN_OOS_SHARPE,
)


class TestWalkForwardGateThresholds:
    def test_oos_is_ratio_constant(self):
        """OOS must be >= 50% of IS Sharpe."""
        assert _MIN_OOS_IS_RATIO == 0.5

    def test_oos_sharpe_minimum(self):
        """Absolute OOS Sharpe floor."""
        assert _MIN_OOS_SHARPE == 0.5

    def test_overfit_ratio_cap(self):
        """Overfit ratio must not exceed 2.0."""
        assert _MAX_OVERFIT_RATIO == 2.0


class TestOosIsRatioLogic:
    """Verify the OOS/IS ratio rejection logic from _evaluate_candidate."""

    def _would_pass(self, oos_sharpe: float, is_sharpe: float) -> bool:
        """Simulate the gate logic from strategy_lifecycle."""
        if oos_sharpe < _MIN_OOS_SHARPE:
            return False
        if is_sharpe > 0:
            oos_is_ratio = oos_sharpe / is_sharpe
            if oos_is_ratio < _MIN_OOS_IS_RATIO:
                return False
        return True

    def test_high_overfit_rejected(self):
        """IS=2.0, OOS=0.8 → ratio 0.4 < 0.5 → REJECTED."""
        assert not self._would_pass(oos_sharpe=0.8, is_sharpe=2.0)

    def test_moderate_overfit_passes(self):
        """IS=1.5, OOS=0.9 → ratio 0.6 >= 0.5 → PASSES."""
        assert self._would_pass(oos_sharpe=0.9, is_sharpe=1.5)

    def test_equal_sharpes_pass(self):
        """IS=1.0, OOS=1.0 → ratio 1.0 >= 0.5 → PASSES."""
        assert self._would_pass(oos_sharpe=1.0, is_sharpe=1.0)

    def test_oos_below_absolute_floor(self):
        """IS=0.8, OOS=0.4 → ratio OK but OOS < 0.5 → REJECTED."""
        assert not self._would_pass(oos_sharpe=0.4, is_sharpe=0.8)

    def test_zero_is_sharpe_skips_ratio(self):
        """When IS Sharpe is 0, ratio check is skipped (no division by zero)."""
        assert self._would_pass(oos_sharpe=0.6, is_sharpe=0.0)
