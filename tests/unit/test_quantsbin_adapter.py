# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quantsbin adapter."""

import pytest
import numpy as np


class TestQuantsbinAdapter:
    """Test suite for quantsbin adapter functions."""

    def test_analyze_long_call(self):
        """Test analysis of single long call."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 450.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 450.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                }
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert result["structure_type"] == "Long Call"
        assert result["num_legs"] == 1
        assert "payoff_profile" in result
        assert "greeks" in result
        assert result["greeks"]["delta"] > 0

    def test_analyze_bull_call_spread(self):
        """Test bull call spread analysis."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 450.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 445.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                },
                {
                    "option_type": "call",
                    "strike": 455.0,
                    "expiry_days": 30,
                    "quantity": -1,
                    "iv": 0.18,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert result["structure_type"] == "Bull Call Spread"
        assert result["is_defined_risk"]
        assert result["max_loss"] < 0  # Has limited loss
        assert result["max_profit"] > 0

    def test_analyze_iron_condor(self):
        """Test iron condor analysis."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 450.0,
            "legs": [
                {
                    "option_type": "put",
                    "strike": 430.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.22,
                },
                {
                    "option_type": "put",
                    "strike": 440.0,
                    "expiry_days": 30,
                    "quantity": -1,
                    "iv": 0.20,
                },
                {
                    "option_type": "call",
                    "strike": 460.0,
                    "expiry_days": 30,
                    "quantity": -1,
                    "iv": 0.18,
                },
                {
                    "option_type": "call",
                    "strike": 470.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert result["structure_type"] == "Iron Condor"
        assert result["is_defined_risk"]
        assert len(result["break_evens"]) == 2

    def test_analyze_straddle(self):
        """Test long straddle analysis."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 450.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 450.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                },
                {
                    "option_type": "put",
                    "strike": 450.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert result["structure_type"] == "Long Straddle"
        assert len(result["break_evens"]) == 2

    def test_payoff_profile(self):
        """Test payoff profile generation."""
        from quantcore.options.adapters.quantsbin_adapter import (
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

        result = analyze_structure_quantsbin(spec, price_range_pct=0.30, num_points=50)

        profile = result["payoff_profile"]

        assert len(profile["prices"]) == 50
        assert len(profile["payoffs"]) == 50
        assert min(profile["prices"]) < 100
        assert max(profile["prices"]) > 100

    def test_greeks_aggregation(self):
        """Test Greeks are properly aggregated."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        # Bull call spread - delta should be positive but less than single deep ITM call
        # Long 95 call (ITM) has higher delta, short 105 call (OTM) reduces it
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

        # Delta should be positive (bullish spread)
        assert result["greeks"]["delta"] > 0

        # Net delta for bull spread should be reasonable (less than 100 = single ITM call)
        assert result["greeks"]["delta"] < 100

    def test_break_even_calculation(self):
        """Test break-even point calculation."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "TEST",
            "underlying_price": 100.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 100.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "premium": 3.0,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        # Break-even should be around strike + premium = 103
        assert len(result["break_evens"]) == 1
        assert 102 < result["break_evens"][0] < 104

    def test_probability_of_profit(self):
        """Test POP estimation."""
        from quantcore.options.adapters.quantsbin_adapter import (
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

        # ATM call should have ~40-50% POP
        pop = result.get("probability_of_profit")
        if pop is not None:
            assert 30 < pop < 60

    def test_empty_structure(self):
        """Test handling of empty structure."""
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

    def test_build_structure_from_template(self):
        """Test building structure from template."""
        from quantcore.options.adapters.quantsbin_adapter import (
            build_structure_from_template,
        )

        spec = build_structure_from_template(
            template_name="bull_call_spread",
            underlying_symbol="SPY",
            underlying_price=450.0,
            atm_strike=450.0,
            strike_width=5.0,
            expiry_days=30,
        )

        assert spec.underlying_symbol == "SPY"
        assert len(spec.legs) == 2
        assert spec.legs[0].quantity == 1
        assert spec.legs[1].quantity == -1

    def test_get_standard_structures(self):
        """Test getting structure templates."""
        from quantcore.options.adapters.quantsbin_adapter import get_standard_structures

        templates = get_standard_structures()

        assert "bull_call_spread" in templates
        assert "iron_condor" in templates
        assert "long_straddle" in templates


class TestStructureIdentification:
    """Test structure type identification."""

    def test_identify_short_put(self):
        """Test short put identification."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [
                {
                    "option_type": "put",
                    "strike": 95.0,
                    "expiry_days": 30,
                    "quantity": -1,
                    "iv": 0.20,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert result["structure_type"] == "Short Put"

    def test_identify_bear_put_spread(self):
        """Test bear put spread identification."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [
                {
                    "option_type": "put",
                    "strike": 105.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                },
                {
                    "option_type": "put",
                    "strike": 95.0,
                    "expiry_days": 30,
                    "quantity": -1,
                    "iv": 0.18,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert result["structure_type"] == "Bear Put Spread"
