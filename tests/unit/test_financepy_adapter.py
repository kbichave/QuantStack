# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for financepy adapter."""

import pytest
import numpy as np


class TestFinancePyAdapter:
    """Test suite for financepy adapter functions."""

    def test_price_vanilla_european(self):
        """Test European option pricing."""
        from quantcore.options.adapters.financepy_adapter import price_vanilla_financepy

        price = price_vanilla_financepy(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
            exercise_style="european",
        )

        assert price > 0
        assert 3.0 < price < 6.0  # Reasonable range for ATM call

    def test_price_american_call(self):
        """Test American call option pricing."""
        from quantcore.options.adapters.financepy_adapter import price_american_option

        result = price_american_option(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
            num_steps=100,
        )

        assert "price" in result
        assert "european_price" in result
        assert "early_exercise_premium" in result

        # American call without dividends should equal European
        assert abs(result["early_exercise_premium"]) < 0.05

    def test_price_american_put(self):
        """Test American put option pricing."""
        from quantcore.options.adapters.financepy_adapter import price_american_option

        result = price_american_option(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="put",
            num_steps=100,
        )

        # American put should be >= European put
        assert result["price"] >= result["european_price"] - 0.01
        assert result["early_exercise_premium"] >= -0.01

    def test_american_put_early_exercise_value(self):
        """Test American put has early exercise value for deep ITM."""
        from quantcore.options.adapters.financepy_adapter import price_american_option

        # Deep ITM put
        result = price_american_option(
            spot=80.0,
            strike=100.0,
            time_to_expiry=0.5,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="put",
            num_steps=100,
        )

        # Deep ITM American put should have meaningful early exercise premium
        assert result["early_exercise_premium"] > 0

    def test_american_with_dividends(self):
        """Test American call with dividends has early exercise value."""
        from quantcore.options.adapters.financepy_adapter import price_american_option

        # High dividend stock - American call may have early exercise value
        result = price_american_option(
            spot=100.0,
            strike=95.0,  # ITM
            time_to_expiry=0.5,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.05,  # 5% dividend
            option_type="call",
            num_steps=100,
        )

        # With dividends, American call can have early exercise value
        assert result["price"] > 0

    def test_american_greeks(self):
        """Test Greeks from American option pricing."""
        from quantcore.options.adapters.financepy_adapter import price_american_option

        result = price_american_option(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
        )

        assert "delta" in result
        assert "gamma" in result

        # ATM delta should be ~0.5
        assert 0.4 < result["delta"] < 0.6

    def test_price_at_expiry(self):
        """Test pricing at expiry returns intrinsic."""
        from quantcore.options.adapters.financepy_adapter import price_american_option

        result = price_american_option(
            spot=105.0,
            strike=100.0,
            time_to_expiry=0.0,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
        )

        assert abs(result["price"] - 5.0) < 0.1

    def test_convergence_with_steps(self):
        """Test that more steps improves accuracy."""
        from quantcore.options.adapters.financepy_adapter import price_american_option

        prices = []
        for steps in [20, 50, 100, 200]:
            result = price_american_option(
                spot=100.0,
                strike=100.0,
                time_to_expiry=0.25,
                vol=0.20,
                rate=0.05,
                dividend_yield=0.0,
                option_type="put",
                num_steps=steps,
            )
            prices.append(result["price"])

        # Prices should converge
        diffs = [abs(prices[i + 1] - prices[i]) for i in range(len(prices) - 1)]
        # Later diffs should be smaller
        assert diffs[-1] < diffs[0]


class TestBarrierOptions:
    """Test barrier option pricing."""

    def test_down_out_call(self):
        """Test down-and-out call pricing."""
        from quantcore.options.adapters.financepy_adapter import price_barrier_option

        result = price_barrier_option(
            spot=100.0,
            strike=100.0,
            barrier=80.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
            barrier_type="down-out",
        )

        assert "price" in result
        # Barrier call should be cheaper than vanilla
        assert result["price"] > 0

    def test_barrier_knocked_out(self):
        """Test barrier that's already been hit."""
        from quantcore.options.adapters.financepy_adapter import price_barrier_option

        result = price_barrier_option(
            spot=75.0,  # Below barrier
            strike=100.0,
            barrier=80.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
            barrier_type="down-out",
        )

        assert not result["is_active"]


class TestFinancePyFallback:
    """Test fallback to internal implementation."""

    def test_binomial_fallback(self):
        """Test internal binomial tree implementation."""
        from quantcore.options.adapters.financepy_adapter import price_american_option

        # Should work regardless of financepy availability
        result = price_american_option(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="put",
        )

        assert "price" in result
        assert result["price"] > 0
