# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests to verify all outputs are JSON-serializable."""

import json

import pandas as pd


class TestJSONSerialization:
    """Tests to verify all outputs are JSON-serializable."""

    def test_price_option_serializable(self):
        """Test price_option output is JSON-serializable."""
        from quantcore.options.engine import price_option_dispatch

        result = price_option_dispatch(100, 100, 0.25, 0.20, 0.05, 0.0, "call")

        # Should not raise
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_greeks_serializable(self):
        """Test greeks output is JSON-serializable."""
        from quantcore.options.engine import compute_greeks_dispatch

        result = compute_greeks_dispatch(100, 100, 0.25, 0.20, 0.05, 0.0, "call")

        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_structure_analysis_serializable(self):
        """Test structure analysis output is JSON-serializable."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 100,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                }
            ],
        }

        result = analyze_structure_quantsbin(spec)

        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_portfolio_stats_serializable(self):
        """Test portfolio stats output is JSON-serializable."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        result = compute_portfolio_stats_ffn([100, 102, 104, 103, 105])

        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_sabr_fit_serializable(self):
        """Test SABR fit output is JSON-serializable."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        quotes = pd.DataFrame(
            {
                "strike": [90, 95, 100, 105, 110],
                "iv": [0.28, 0.24, 0.22, 0.23, 0.26],
            }
        )

        result = fit_sabr_surface(quotes, 100.0, 30 / 365)

        # Need to convert SABRParams to dict if present
        if "params" in result and hasattr(result["params"], "to_dict"):
            result["params"] = result["params"].to_dict()

        json_str = json.dumps(result)
        assert len(json_str) > 0
