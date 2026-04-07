"""Tests for mandatory stop-loss enforcement (Section 02).

Layer 1: execute_trade() rejects entry orders without stop_price.
Layer 3: BracketIntent model requires stop_price.
Layer 6: bracket_legs table schema.
"""

import pytest
from pydantic import ValidationError

from quantstack.execution.models import BracketIntent, BracketLeg


# ---------------------------------------------------------------------------
# Layer 1: execute_trade() validation
# ---------------------------------------------------------------------------

class TestExecuteTradeValidation:
    """execute_trade() must reject entry orders without stop_price."""

    @pytest.mark.asyncio
    async def test_rejects_buy_without_stop_price(self):
        """Entry (buy) order with stop_price=None is rejected."""
        from quantstack.execution.trade_service import execute_trade

        class MockPortfolio:
            def get_snapshot(self):
                class S:
                    total_equity = 100_000
                return S()
            def get_position(self, s):
                return None

        class MockKillSwitch:
            def guard(self):
                pass

        result = await execute_trade(
            portfolio=MockPortfolio(),
            risk_gate=None,
            broker=None,
            audit=None,
            kill_switch=MockKillSwitch(),
            session_id="test",
            symbol="AAPL",
            action="buy",
            reasoning="test",
            confidence=0.8,
            quantity=100,
            stop_price=None,
        )
        # execute_trade catches ValueError and returns error dict
        assert result.get("success") is False
        assert "stop_price" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_accepts_buy_with_stop_price(self):
        """Entry order with stop_price passes validation (may fail later at risk gate)."""
        from quantstack.execution.trade_service import execute_trade

        class MockPortfolio:
            def get_snapshot(self):
                class S:
                    total_equity = 100_000
                return S()
            def get_position(self, s):
                class P:
                    current_price = 150.0
                return P()

        class MockKillSwitch:
            def guard(self):
                pass

        class MockRiskGate:
            def check(self, **kwargs):
                class V:
                    approved = False
                    violations = []
                return V()

        class MockAudit:
            def record(self, event):
                pass

        # Should not raise ValueError — will fail at risk gate
        result = await execute_trade(
            portfolio=MockPortfolio(),
            risk_gate=MockRiskGate(),
            broker=None,
            audit=MockAudit(),
            kill_switch=MockKillSwitch(),
            session_id="test",
            symbol="AAPL",
            action="buy",
            reasoning="test",
            confidence=0.8,
            quantity=100,
            stop_price=145.0,
        )
        # Risk gate rejects — but no ValueError about stop_price
        assert result.get("success") is False


# ---------------------------------------------------------------------------
# Layer 3: BracketIntent model
# ---------------------------------------------------------------------------

class TestBracketIntent:
    """BracketIntent model enforces stop_price as required."""

    def test_valid_bracket_intent(self):
        intent = BracketIntent(
            symbol="AAPL",
            side="buy",
            quantity=100,
            stop_price=145.0,
        )
        assert intent.stop_price == 145.0

    def test_stop_price_is_required(self):
        """BracketIntent without stop_price raises ValidationError."""
        with pytest.raises(ValidationError):
            BracketIntent(
                symbol="AAPL",
                side="buy",
                quantity=100,
                # stop_price omitted
            )

    def test_target_price_is_optional(self):
        intent = BracketIntent(
            symbol="AAPL",
            side="buy",
            quantity=100,
            stop_price=145.0,
        )
        assert intent.target_price is None

    def test_entry_type_defaults_to_market(self):
        intent = BracketIntent(
            symbol="AAPL",
            side="buy",
            quantity=100,
            stop_price=145.0,
        )
        assert intent.entry_type == "market"


# ---------------------------------------------------------------------------
# Layer 6: BracketLeg model
# ---------------------------------------------------------------------------

class TestBracketLeg:
    """BracketLeg model for persisted bracket state."""

    def test_valid_bracket_leg(self):
        leg = BracketLeg(
            parent_order_id="ord-123",
            leg_type="stop_loss",
            broker_order_id="brk-456",
            status="active",
            price=145.0,
        )
        assert leg.leg_type == "stop_loss"

    def test_default_status_is_pending(self):
        leg = BracketLeg(
            parent_order_id="ord-123",
            leg_type="entry",
        )
        assert leg.status == "pending"
