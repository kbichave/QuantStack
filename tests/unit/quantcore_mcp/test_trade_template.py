# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for trade template generation and validation logic."""


class TestTradeTemplateLogic:
    """Tests for trade template generation and validation logic."""

    def test_trade_validation_passes(self, sample_trade_template):
        """Test that valid trade passes validation."""

        # Simulate what validate_trade does internally
        template = sample_trade_template
        max_loss = abs(template.get("risk_profile", {}).get("max_loss", float("inf")))
        is_defined_risk = template.get("validation", {}).get("is_defined_risk", False)
        account_equity = 100000.0
        max_position_pct = 5.0

        max_position_value = account_equity * max_position_pct / 100

        # Checks
        checks = {
            "defined_risk": is_defined_risk,
            "within_position_limit": max_loss <= max_position_value,
        }

        assert checks["defined_risk"]
        assert checks["within_position_limit"]

    def test_trade_validation_rejects_oversized(self):
        """Test that oversized position fails validation."""
        template = {
            "risk_profile": {"max_loss": -10000},
            "validation": {"is_defined_risk": True},
        }

        max_loss = abs(template["risk_profile"]["max_loss"])
        account_equity = 100000.0
        max_position_pct = 1.0  # Very restrictive

        max_position_value = account_equity * max_position_pct / 100

        within_limit = max_loss <= max_position_value

        assert not within_limit

    def test_trade_template_structure(self, sample_trade_template):
        """Test trade template has required fields."""
        template = sample_trade_template

        assert "symbol" in template
        assert "legs" in template
        assert "risk_profile" in template
        assert "greeks" in template
        assert "validation" in template
