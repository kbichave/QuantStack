# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the options engine that powers MCP tools."""


class TestOptionsEngine:
    """Tests for the options engine that powers MCP tools."""

    def test_price_option_dispatch_european(self):
        """Test European option pricing through engine."""
        from quantcore.options.engine import price_option_dispatch

        result = price_option_dispatch(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
            exercise_style="european",
        )

        assert "price" in result
        assert result["price"] > 0
        assert "greeks" in result

    def test_price_option_dispatch_american(self):
        """Test American option pricing through engine."""
        from quantcore.options.engine import price_option_dispatch

        result = price_option_dispatch(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="put",
            exercise_style="american",
        )

        assert "price" in result
        assert result["price"] > 0

    def test_compute_greeks_dispatch(self):
        """Test Greeks computation through engine."""
        from quantcore.options.engine import compute_greeks_dispatch

        result = compute_greeks_dispatch(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
        )

        assert "greeks" in result
        assert "delta" in result["greeks"]
        assert 0.4 < result["greeks"]["delta"] < 0.6  # ATM call

    def test_compute_iv_dispatch(self):
        """Test IV computation through engine."""
        from quantcore.options.engine import compute_iv_dispatch

        result = compute_iv_dispatch(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            rate=0.05,
            dividend_yield=0.0,
            option_price=5.0,
            option_type="call",
        )

        assert "implied_volatility" in result or "error" in result
        if "implied_volatility" in result:
            assert 0.1 < result["implied_volatility"] < 0.5


class TestQuickFunctions:
    """Tests for quick convenience functions."""

    def test_quick_price(self):
        """Test quick option pricing."""
        from quantcore.options.engine import quick_price

        price = quick_price(100, 100, 30, 0.20, "call")

        assert price > 0
        assert price < 10  # Reasonable for ATM 30-day call

    def test_quick_greeks(self):
        """Test quick Greeks computation."""
        from quantcore.options.engine import quick_greeks

        greeks = quick_greeks(100, 100, 30, 0.20, "call")

        assert "delta" in greeks
        assert "gamma" in greeks
        assert 0.4 < greeks["delta"] < 0.6  # ATM call

    def test_quick_iv(self):
        """Test quick IV computation."""
        # Price an option first
        from quantcore.options.engine import quick_iv, quick_price

        price = quick_price(100, 100, 30, 0.25, "call")

        # Recover IV
        iv = quick_iv(100, 100, 30, price, "call")

        assert iv is not None
        assert abs(iv - 0.25) < 0.01  # Should be close to original vol
