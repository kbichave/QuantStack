"""Tests for EWMA volatility and vol-state hysteresis (section-03)."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.kelly_sizing import (
    VOL_STATE_ENTER_THRESHOLD,
    VOL_STATE_EXIT_THRESHOLD,
    compute_vol_state,
)
from quantstack.core.risk.position_sizing import ewma_volatility


class TestEwmaVolatility:
    def test_returns_annualized_vol(self):
        """Known returns series produces reasonable annualized vol."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 100))  # ~1% daily vol
        vol = ewma_volatility(returns)
        assert vol is not None
        # 1% daily ≈ 15.9% annualized
        assert 0.10 <= vol <= 0.30

    def test_cold_start_returns_none(self):
        """Fewer than 21 observations → None."""
        returns = pd.Series(np.random.normal(0, 0.01, 15))
        assert ewma_volatility(returns) is None

    def test_exactly_min_periods_returns_value(self):
        """Exactly 21 observations → non-None."""
        returns = pd.Series(np.random.normal(0, 0.01, 21))
        vol = ewma_volatility(returns)
        assert vol is not None
        assert vol > 0

    def test_vol_floor_applied(self):
        """Very calm series (tiny returns) → floor of 0.10."""
        returns = pd.Series(np.random.normal(0, 0.0001, 100))  # 0.01% daily
        vol = ewma_volatility(returns)
        assert vol is not None
        assert vol >= 0.10

    def test_daily_vol_no_floor(self):
        """annualize=False returns daily vol without floor."""
        returns = pd.Series(np.random.normal(0, 0.0001, 100))
        vol = ewma_volatility(returns, annualize=False)
        assert vol is not None
        assert vol < 0.01  # Daily vol should be tiny

    def test_spike_decay(self):
        """Vol spike decays — day 5 estimate is higher than day 15 after a shock."""
        np.random.seed(123)
        calm = np.random.normal(0, 0.005, 30)
        spike = np.array([0.05])  # 5% shock
        post_spike = np.random.normal(0, 0.005, 20)
        returns = pd.Series(np.concatenate([calm, spike, post_spike]))

        # Vol at day 5 after spike (index 35)
        vol_early = ewma_volatility(returns[:36])
        # Vol at day 15 after spike (index 45)
        vol_late = ewma_volatility(returns[:46])

        assert vol_early is not None and vol_late is not None
        assert vol_early > vol_late, "Spike should decay over time"


class TestComputeVolState:
    def test_enters_high_when_above_enter_threshold(self):
        """Vol at 1.6× mean → enters high from normal."""
        result = compute_vol_state(0.32, 0.20, "normal")
        assert result == "high"

    def test_stays_normal_below_enter_threshold(self):
        """Vol at 1.3× mean → stays normal."""
        result = compute_vol_state(0.26, 0.20, "normal")
        assert result == "normal"

    def test_stays_high_above_exit_threshold(self):
        """In high state, vol at 1.3× mean (above exit 1.2×) → stays high."""
        result = compute_vol_state(0.26, 0.20, "high")
        assert result == "high"

    def test_exits_high_below_exit_threshold(self):
        """In high state, vol at 1.1× mean (below exit 1.2×) → exits to normal."""
        result = compute_vol_state(0.22, 0.20, "high")
        assert result == "normal"

    def test_hysteresis_gap(self):
        """Between exit (1.2×) and enter (1.5×) thresholds, state is sticky."""
        mean_vol = 0.20
        vol_in_gap = mean_vol * 1.35  # 1.35× — between 1.2 and 1.5

        assert compute_vol_state(vol_in_gap, mean_vol, "normal") == "normal"
        assert compute_vol_state(vol_in_gap, mean_vol, "high") == "high"

    def test_zero_mean_vol_returns_normal(self):
        """Edge case: zero mean vol → normal (no division by zero)."""
        assert compute_vol_state(0.20, 0.0, "normal") == "normal"
