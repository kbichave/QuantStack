# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for adapter functions used by MCP tools."""

import pandas as pd


class TestAdaptersForMCP:
    """Tests for adapter functions used by MCP tools."""

    def test_analyze_structure_long_call(self):
        """Test structure analysis for single call."""
        from quantstack.core.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 100.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert "structure_type" in result
        assert result["structure_type"] == "Long Call"
        assert "greeks" in result

    def test_analyze_structure_spread(self):
        """Test structure analysis for spread."""
        from quantstack.core.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 95.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                },
                {
                    "option_type": "call",
                    "strike": 105.0,
                    "expiry_days": 30,
                    "quantity": -1,
                    "iv": 0.18,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert "structure_type" in result
        assert result["structure_type"] == "Bull Call Spread"
        assert result["is_defined_risk"]

    def test_portfolio_stats_ffn(self, sample_equity_curve):
        """Test portfolio stats computation."""
        from quantstack.core.analytics.adapters.ffn_adapter import (
            compute_portfolio_stats_ffn,
        )

        result = compute_portfolio_stats_ffn(sample_equity_curve)

        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result

    def test_sabr_surface_fit(self):
        """Test SABR surface fitting."""
        from quantstack.core.options.adapters.pysabr_adapter import fit_sabr_surface

        quotes = pd.DataFrame(
            {
                "strike": [90, 95, 100, 105, 110],
                "iv": [0.28, 0.24, 0.22, 0.23, 0.26],
            }
        )

        result = fit_sabr_surface(
            quotes=quotes,
            forward=100.0,
            time_to_expiry=30 / 365,
            beta=1.0,
        )

        assert "params" in result or "params_dict" in result
        assert "fit_quality" in result

    def test_american_option_pricing(self):
        """Test American option pricing via adapter."""
        from quantstack.core.options.adapters.financepy_adapter import (
            price_american_option,
        )

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
        assert "early_exercise_premium" in result
