# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for error handling in MCP tool implementations."""

import pandas as pd
import pytest


class TestErrorHandling:
    """Tests for error handling in MCP tool implementations."""

    def test_structure_empty_legs_error(self):
        """Test error handling for empty legs."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [],
        }

        result = analyze_structure_quantsbin(spec)

        assert "error" in result

    def test_portfolio_stats_insufficient_data(self):
        """Test error handling for insufficient data."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        result = compute_portfolio_stats_ffn([100])

        assert "error" in result

    def test_sabr_insufficient_points(self):
        """Test SABR fitting with insufficient data points."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        quotes = pd.DataFrame({"strike": [100], "iv": [0.22]})

        with pytest.raises(ValueError):
            fit_sabr_surface(quotes, 100.0, 30 / 365)

    def test_vollib_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vollib

        with pytest.raises(ValueError):
            bs_price_vollib(-100, 100, 0.25, 0.20, 0.05, 0.0, "call")  # Negative spot
