"""Tests for Almgren-Chriss execution cost model."""

import numpy as np
import pytest

from quantcore.execution.almgren_chriss import (
    ACCostBreakdown,
    almgren_chriss_cost_breakdown,
    almgren_chriss_expected_cost_bps,
    calibrate_from_fills,
    optimal_trajectory,
)


# ---------------------------------------------------------------------------
# almgren_chriss_expected_cost_bps
# ---------------------------------------------------------------------------


class TestExpectedCostBps:
    """Core cost estimation function."""

    def test_zero_order_returns_zero(self):
        assert almgren_chriss_expected_cost_bps(0, 1_000_000, 0.02) == 0.0

    def test_zero_volume_returns_zero(self):
        assert almgren_chriss_expected_cost_bps(100, 0, 0.02) == 0.0

    def test_zero_volatility_returns_zero(self):
        assert almgren_chriss_expected_cost_bps(100, 1_000_000, 0.0) == 0.0

    def test_negative_inputs_return_zero(self):
        assert almgren_chriss_expected_cost_bps(-100, 1_000_000, 0.02) == 0.0
        assert almgren_chriss_expected_cost_bps(100, -1_000_000, 0.02) == 0.0
        assert almgren_chriss_expected_cost_bps(100, 1_000_000, -0.02) == 0.0

    def test_cost_is_positive(self):
        cost = almgren_chriss_expected_cost_bps(500, 1_000_000, 0.02)
        assert cost > 0

    def test_cost_scales_with_order_size(self):
        small = almgren_chriss_expected_cost_bps(100, 1_000_000, 0.02)
        large = almgren_chriss_expected_cost_bps(10_000, 1_000_000, 0.02)
        assert large > small

    def test_cost_scales_with_volatility(self):
        low_vol = almgren_chriss_expected_cost_bps(500, 1_000_000, 0.01)
        high_vol = almgren_chriss_expected_cost_bps(500, 1_000_000, 0.04)
        assert high_vol > low_vol

    def test_cost_decreases_with_volume(self):
        """More liquid stock → lower cost for same order size."""
        illiquid = almgren_chriss_expected_cost_bps(500, 100_000, 0.02)
        liquid = almgren_chriss_expected_cost_bps(500, 10_000_000, 0.02)
        assert illiquid > liquid

    def test_known_value_large_cap(self):
        """1% ADV on 2% daily vol large-cap should be in 1-15 bps range."""
        cost = almgren_chriss_expected_cost_bps(
            order_shares=10_000,       # 1% of 1M ADV
            daily_volume=1_000_000,
            daily_volatility=0.02,
        )
        assert 1.0 < cost < 15.0, f"Expected 1-15 bps, got {cost:.2f}"

    def test_tiny_order_negligible_cost(self):
        """50 shares on 5M ADV should cost < 1 bps."""
        cost = almgren_chriss_expected_cost_bps(50, 5_000_000, 0.015)
        assert cost < 1.0


# ---------------------------------------------------------------------------
# almgren_chriss_cost_breakdown
# ---------------------------------------------------------------------------


class TestCostBreakdown:
    """Detailed breakdown of cost components."""

    def test_returns_dataclass(self):
        result = almgren_chriss_cost_breakdown(500, 1_000_000, 0.02)
        assert isinstance(result, ACCostBreakdown)

    def test_components_sum_to_total(self):
        result = almgren_chriss_cost_breakdown(1000, 1_000_000, 0.02)
        expected_total = (
            result.permanent_impact_bps
            + result.temporary_impact_bps
            + result.timing_risk_bps
        )
        assert abs(result.total_bps - expected_total) < 0.01

    def test_participation_pct_correct(self):
        result = almgren_chriss_cost_breakdown(10_000, 1_000_000, 0.02)
        assert abs(result.participation_pct - 1.0) < 0.001  # 10K / 1M = 1%

    def test_all_components_nonnegative(self):
        result = almgren_chriss_cost_breakdown(500, 1_000_000, 0.02)
        assert result.permanent_impact_bps >= 0
        assert result.temporary_impact_bps >= 0
        assert result.timing_risk_bps >= 0

    def test_temporary_dominates_for_small_fast_orders(self):
        """Short horizon + small order → temporary impact is primary cost."""
        result = almgren_chriss_cost_breakdown(
            500, 1_000_000, 0.02,
            execution_horizon_days=1 / 26,  # 15 minutes
        )
        assert result.temporary_impact_bps > result.permanent_impact_bps

    def test_zero_inputs_return_zero_breakdown(self):
        result = almgren_chriss_cost_breakdown(0, 1_000_000, 0.02)
        assert result.total_bps == 0.0
        assert result.participation_pct == 0.0


# ---------------------------------------------------------------------------
# optimal_trajectory
# ---------------------------------------------------------------------------


class TestOptimalTrajectory:
    """Optimal execution trajectory."""

    def test_sums_to_total(self):
        traj = optimal_trajectory(1000, 10, 1_000_000, 0.02)
        assert abs(traj.sum() - 1000) < 0.01

    def test_single_slice_returns_total(self):
        traj = optimal_trajectory(1000, 1, 1_000_000, 0.02)
        assert len(traj) == 1
        assert abs(traj[0] - 1000) < 0.01

    def test_zero_slices_returns_empty(self):
        traj = optimal_trajectory(1000, 0, 1_000_000, 0.02)
        assert len(traj) == 0

    def test_zero_shares_returns_empty(self):
        traj = optimal_trajectory(0, 10, 1_000_000, 0.02)
        assert len(traj) == 0

    def test_all_slices_nonnegative(self):
        traj = optimal_trajectory(1000, 20, 1_000_000, 0.02)
        assert (traj >= 0).all()

    def test_high_urgency_front_loads(self):
        """High risk_aversion should execute more in early slices."""
        traj = optimal_trajectory(
            1000, 10, 1_000_000, 0.02,
            risk_aversion=1.0,  # very high urgency
        )
        first_half = traj[:5].sum()
        second_half = traj[5:].sum()
        assert first_half > second_half

    def test_low_urgency_is_uniform(self):
        """Near-zero risk_aversion should be close to TWAP (uniform)."""
        traj = optimal_trajectory(
            1000, 10, 1_000_000, 0.02,
            risk_aversion=1e-12,
        )
        # Should be close to 100 per slice
        assert np.std(traj) < 20  # allow some numerical noise


# ---------------------------------------------------------------------------
# calibrate_from_fills
# ---------------------------------------------------------------------------


class TestCalibration:
    """Coefficient calibration from historical fills."""

    def test_insufficient_data_returns_defaults(self):
        gamma, eta = calibrate_from_fills(
            np.array([100.5]), np.array([100.0]),
            np.array([500]), np.array([1_000_000]),
            np.array([0.02]),
        )
        assert gamma == 0.1
        assert eta == 0.01

    def test_calibrated_coefficients_positive(self):
        rng = np.random.default_rng(42)
        n = 50
        arrival = rng.uniform(90, 110, n)
        sizes = rng.integers(100, 5000, n).astype(float)
        volumes = rng.uniform(500_000, 2_000_000, n)
        vols = rng.uniform(0.01, 0.03, n)

        # Simulate fills with known impact
        participation = sizes / volumes
        slippage_frac = 0.15 * vols * participation + 0.02 * vols * np.sqrt(participation / (1/6.5))
        fills = arrival * (1 + slippage_frac)

        gamma, eta = calibrate_from_fills(fills, arrival, sizes, volumes, vols)
        assert gamma > 0
        assert eta > 0

    def test_higher_impact_fills_produce_larger_coefficients(self):
        rng = np.random.default_rng(42)
        n = 50
        arrival = rng.uniform(95, 105, n)
        sizes = rng.integers(100, 5000, n).astype(float)
        volumes = rng.uniform(500_000, 2_000_000, n)
        vols = rng.uniform(0.01, 0.03, n)

        participation = sizes / volumes

        # Low impact scenario
        slippage_low = 0.05 * vols * participation
        fills_low = arrival * (1 + slippage_low)
        gamma_low, eta_low = calibrate_from_fills(fills_low, arrival, sizes, volumes, vols)

        # High impact scenario
        slippage_high = 0.5 * vols * participation
        fills_high = arrival * (1 + slippage_high)
        gamma_high, eta_high = calibrate_from_fills(fills_high, arrival, sizes, volumes, vols)

        assert gamma_high > gamma_low or eta_high > eta_low
