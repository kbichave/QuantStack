# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vollib adapter."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestVolibAdapter:
    """Test suite for vollib adapter functions."""

    def test_bs_price_call_atm(self):
        """Test ATM call option pricing."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vollib

        price = bs_price_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
        )

        # ATM call should have positive value
        assert price > 0
        # Sanity check: ATM call ~4-5% of spot for 25% vol, 3 months
        assert 3.0 < price < 6.0

    def test_bs_price_put_atm(self):
        """Test ATM put option pricing."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vollib

        price = bs_price_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="put",
        )

        assert price > 0
        assert 2.0 < price < 5.0

    def test_bs_price_put_call_parity(self):
        """Test put-call parity relationship."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vollib

        spot = 100.0
        strike = 100.0
        tte = 0.25
        rate = 0.05
        vol = 0.20
        div = 0.0

        call = bs_price_vollib(spot, strike, tte, vol, rate, div, "call")
        put = bs_price_vollib(spot, strike, tte, vol, rate, div, "put")

        # Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
        forward = spot * np.exp(-div * tte)
        pv_strike = strike * np.exp(-rate * tte)

        assert abs((call - put) - (forward - pv_strike)) < 0.01

    def test_bs_price_at_expiry(self):
        """Test pricing at expiry returns intrinsic value."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vollib

        # ITM call
        call_price = bs_price_vollib(105, 100, 0.0, 0.20, 0.05, 0.0, "call")
        assert call_price == 5.0

        # OTM call
        call_price = bs_price_vollib(95, 100, 0.0, 0.20, 0.05, 0.0, "call")
        assert call_price == 0.0

        # ITM put
        put_price = bs_price_vollib(95, 100, 0.0, 0.20, 0.05, 0.0, "put")
        assert put_price == 5.0

    def test_implied_vol_round_trip(self):
        """Test IV calculation round-trips correctly."""
        from quantcore.options.adapters.vollib_adapter import (
            bs_price_vollib,
            implied_vol_vollib,
        )

        spot = 100.0
        strike = 100.0
        tte = 0.25
        vol = 0.25
        rate = 0.05
        div = 0.0

        # Price option
        price = bs_price_vollib(spot, strike, tte, vol, rate, div, "call")

        # Recover IV
        recovered_iv = implied_vol_vollib(spot, strike, tte, rate, div, price, "call")

        assert recovered_iv is not None
        assert abs(recovered_iv - vol) < 0.001

    def test_implied_vol_invalid_price(self):
        """Test IV returns None for invalid prices."""
        from quantcore.options.adapters.vollib_adapter import implied_vol_vollib

        # Price below intrinsic
        iv = implied_vol_vollib(
            spot=110.0,
            strike=100.0,
            time_to_expiry=0.25,
            rate=0.05,
            dividend_yield=0.0,
            option_price=5.0,  # Below intrinsic of ~10
            option_type="call",
        )

        assert iv is None

    def test_greeks_call(self):
        """Test Greeks calculation for call option."""
        from quantcore.options.adapters.vollib_adapter import greeks_vollib

        greeks = greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
        )

        # Delta should be ~0.5 for ATM call
        assert 0.4 < greeks["delta"] < 0.6

        # Gamma should be positive
        assert greeks["gamma"] > 0

        # Theta should be negative (time decay)
        assert greeks["theta"] < 0

        # Vega should be positive
        assert greeks["vega"] > 0

    def test_greeks_put(self):
        """Test Greeks for put option."""
        from quantcore.options.adapters.vollib_adapter import greeks_vollib

        greeks = greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="put",
        )

        # Delta should be ~-0.5 for ATM put
        assert -0.6 < greeks["delta"] < -0.4

        # Gamma same as call
        assert greeks["gamma"] > 0

    def test_greeks_at_expiry(self):
        """Test Greeks at expiry."""
        from quantcore.options.adapters.vollib_adapter import greeks_vollib

        greeks = greeks_vollib(
            spot=105.0,
            strike=100.0,
            time_to_expiry=0.0,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
        )

        # ITM call at expiry has delta = 1
        assert greeks["delta"] == 1.0
        assert greeks["gamma"] == 0.0

    def test_option_type_normalization(self):
        """Test option type variations are handled."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vollib

        price_call = bs_price_vollib(100, 100, 0.25, 0.20, 0.05, 0.0, "call")
        price_c = bs_price_vollib(100, 100, 0.25, 0.20, 0.05, 0.0, "c")
        price_CALL = bs_price_vollib(100, 100, 0.25, 0.20, 0.05, 0.0, "CALL")

        assert price_call == price_c
        assert price_call == price_CALL

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vollib

        with pytest.raises(ValueError):
            bs_price_vollib(-100, 100, 0.25, 0.20, 0.05, 0.0, "call")

        with pytest.raises(ValueError):
            bs_price_vollib(100, -100, 0.25, 0.20, 0.05, 0.0, "call")

        with pytest.raises(ValueError):
            bs_price_vollib(100, 100, 0.25, -0.20, 0.05, 0.0, "call")

    def test_vectorized_pricing(self):
        """Test vectorized pricing function."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vectorized

        spots = [100.0, 100.0, 100.0]
        strikes = [95.0, 100.0, 105.0]
        ttes = [0.25, 0.25, 0.25]
        vols = [0.20, 0.20, 0.20]

        prices = bs_price_vectorized(spots, strikes, ttes, vols)

        assert len(prices) == 3
        # ITM should be most expensive
        assert prices[0] > prices[1] > prices[2]


class TestVolibAdapterFallback:
    """Test fallback to internal implementation when vollib unavailable."""

    def test_fallback_pricing(self):
        """Test fallback to internal BS when vollib fails."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vollib

        # This should work regardless of vollib availability
        price = bs_price_vollib(100, 100, 0.25, 0.20, 0.05, 0.0, "call")
        assert price > 0

    def test_fallback_greeks(self):
        """Test fallback Greeks calculation."""
        from quantcore.options.adapters.vollib_adapter import greeks_vollib

        greeks = greeks_vollib(100, 100, 0.25, 0.20, 0.05, 0.0, "call")

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks
