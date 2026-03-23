# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for risk tools: VaR, stress testing, risk controls."""

import numpy as np
from quantstack.core.options.models import OptionType
from quantstack.core.options.pricing import black_scholes_price
from quantstack.core.risk.controls import ExposureManager
from quantstack.core.risk.stress_testing import STRESS_SCENARIOS


class TestVaRCalculation:
    """Tests for VaR calculation."""

    def test_historical_var(self):
        """Test historical VaR calculation."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02  # 2% daily vol

        # Calculate VaR manually
        var_95 = -np.percentile(returns, 5) * 100

        assert var_95 > 0  # VaR should be positive (loss)
        assert var_95 < 10  # Should be reasonable for 2% vol

    def test_var_scaling(self):
        """Test VaR scales with sqrt(time)."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02

        var_1d = -np.percentile(returns, 5)
        var_5d = -np.percentile(returns * np.sqrt(5), 5)

        # 5-day VaR should be ~sqrt(5) times 1-day VaR
        assert 2 < var_5d / var_1d < 3  # sqrt(5) ~ 2.24


class TestStressTesting:
    """Tests for portfolio stress testing."""

    def test_stress_scenario_pnl(self):
        """Test stress scenario P&L calculation."""
        # Initial position: ATM call
        # Signature: black_scholes_price(S, K, T, r, sigma, option_type, q=0.0)
        S0, K, T, vol, r = 100, 100, 0.25, 0.20, 0.05
        initial_price = black_scholes_price(S0, K, T, r, vol, OptionType.CALL)

        # Stress scenario: -20% price, +50% vol
        S_stressed = S0 * 0.8
        vol_stressed = vol * 1.5
        stressed_price = black_scholes_price(
            S_stressed, K, T, r, vol_stressed, OptionType.CALL
        )

        # Call should lose value from price drop
        assert stressed_price < initial_price

    def test_stress_scenarios_exist(self):
        """Test predefined stress scenarios exist."""
        assert "2008_lehman" in STRESS_SCENARIOS
        assert "2020_covid_crash" in STRESS_SCENARIOS
        assert "2018_volmageddon" in STRESS_SCENARIOS


class TestRiskControls:
    """Tests for risk control checks."""

    def test_normal_state(self):
        """Test normal risk state."""
        manager = ExposureManager(
            max_concurrent_trades=5,
            max_exposure_per_symbol_pct=20.0,
        )

        can_open, reason = manager.can_open_position("SPY", 10.0, 100000)

        assert can_open

    def test_position_limit_breach(self):
        """Test position limit enforcement."""
        manager = ExposureManager(max_concurrent_trades=2)

        # Fill up positions
        manager._open_positions = {"SPY": 20, "QQQ": 20}

        can_open, reason = manager.can_open_position("AAPL", 10.0, 100000)

        assert not can_open
        assert "concurrent" in reason.lower()
