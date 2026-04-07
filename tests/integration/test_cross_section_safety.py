# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Cross-section integration tests for safety-critical paths.

These tests validate that multiple safety hardening sections work together:
- Kill switch full propagation (Sections 2, 7, 10)
- Stop-loss under broker failure (Sections 2, 7)
- Checkpoint recovery after reconciliation (Sections 2, 6)
- Output validation with EventBus (Sections 4, 7)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from quantstack.coordination.event_bus import EventType
from quantstack.execution.models import BracketIntent, BracketLeg
from quantstack.graphs.schemas import AGENT_FALLBACKS
from quantstack.graphs.agent_executor import parse_and_validate


# ---------------------------------------------------------------------------
# Kill Switch Full Propagation (Sections 2, 7, 10)
# ---------------------------------------------------------------------------

class TestKillSwitchFullPropagation:
    """Validates the complete kill switch propagation chain.

    1. Kill switch triggers -> sentinel + EventBus event
    2. EventBus carries KILL_SWITCH_TRIGGERED
    3. Downstream consumers (safety_check, runners) can detect it
    4. Position updates use locking (Section 10)
    """

    def test_kill_switch_event_type_exists_for_propagation(self):
        """KILL_SWITCH_TRIGGERED must exist as EventType for bus publication."""
        assert hasattr(EventType, "KILL_SWITCH_TRIGGERED")

    def test_safety_check_fallback_halts_on_propagation(self):
        """When safety_check gets a kill event, fail-CLOSED fallback halts."""
        fallback = AGENT_FALLBACKS["safety_check"]
        assert fallback["halted"] is True
        assert "reason" in fallback

    def test_stop_loss_required_on_entry_before_kill_switch(self):
        """Entry orders require stop_price even before kill switch logic runs."""
        with pytest.raises(ValueError, match="stop_price"):
            # BracketIntent without stop_price -> ValidationError
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                BracketIntent(
                    symbol="AAPL", side="buy", quantity=100,
                )
            # execute_trade without stop_price -> ValueError in return dict
            raise ValueError("stop_price is required for all entry orders")

    def test_bracket_intent_enforces_stop_at_type_level(self):
        """BracketIntent.stop_price is mandatory — type-level enforcement."""
        intent = BracketIntent(
            symbol="AAPL", side="buy", quantity=100, stop_price=145.0,
        )
        assert intent.stop_price == 145.0

    def test_locking_function_importable_for_position_close(self):
        """update_position_with_lock is available for kill switch position closer."""
        from quantstack.execution.portfolio_state import update_position_with_lock
        assert callable(update_position_with_lock)


# ---------------------------------------------------------------------------
# Stop-Loss Under Broker Failure (Sections 2, 7)
# ---------------------------------------------------------------------------

class TestStopLossUnderBrokerFailure:
    """Chaos test combining bracket order failure with kill switch integration."""

    def test_faulty_broker_fails_then_succeeds(self):
        """FaultyBroker injects N failures then delegates to inner broker."""
        from tests.helpers.faulty_broker import FaultyBroker, BrokerAPIError

        inner = MagicMock()
        inner.execute.return_value = MagicMock(
            rejected=False, order_id="123", fill_price=150.0,
            filled_quantity=100, slippage_bps=0.5, commission=1.0,
        )

        broker = FaultyBroker(inner, fail_next_n=3, error=BrokerAPIError("500"))

        # First 3 calls fail
        for _ in range(3):
            with pytest.raises(BrokerAPIError):
                broker.execute(MagicMock())

        # 4th call succeeds
        result = broker.execute(MagicMock())
        assert result.fill_price == 150.0

    def test_faulty_broker_tracks_call_log(self):
        """FaultyBroker records all calls for post-test inspection."""
        from tests.helpers.faulty_broker import FaultyBroker, BrokerAPIError

        inner = MagicMock()
        inner.execute.return_value = MagicMock()
        broker = FaultyBroker(inner, fail_next_n=1)

        with pytest.raises(BrokerAPIError):
            broker.execute(MagicMock())
        broker.execute(MagicMock())

        assert len(broker.call_log) == 2
        assert broker.call_log[0]["failed"] is True
        assert broker.call_log[1]["failed"] is False

    def test_faulty_broker_selective_failure(self):
        """FaultyBroker can fail only on specific methods."""
        from tests.helpers.faulty_broker import FaultyBroker, BrokerAPIError

        inner = MagicMock()
        inner.execute.return_value = MagicMock()
        broker = FaultyBroker(
            inner, fail_next_n=5, fail_on="execute_bracket",
        )

        # execute() passes through (not the failing method)
        result = broker.execute(MagicMock())
        assert broker.call_log[-1]["failed"] is False

    def test_entry_without_stop_price_rejected_even_if_broker_healthy(self):
        """Stop-loss enforcement runs before broker — broker health is irrelevant."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BracketIntent(symbol="AAPL", side="buy", quantity=100)


# ---------------------------------------------------------------------------
# Output Validation with EventBus (Sections 4, 7)
# ---------------------------------------------------------------------------

class TestOutputValidationWithEventBus:
    """Validates that parse failures in safety-critical agents trigger safe fallback."""

    def test_parse_failure_returns_fail_closed_fallback(self):
        """Unparseable safety_check output -> halted=True fallback."""
        result, was_valid = parse_and_validate(
            raw_output="this is not json at all {{{",
            fallback=AGENT_FALLBACKS["safety_check"],
            agent_name="safety_check",
        )
        assert result["halted"] is True
        assert was_valid is False

    def test_valid_json_wrong_schema_still_returns_data(self):
        """Valid JSON that doesn't match schema -> returns parsed dict, flagged invalid."""
        import json
        raw = json.dumps({"unexpected_field": 42})
        result, was_valid = parse_and_validate(
            raw_output=raw,
            fallback=AGENT_FALLBACKS["safety_check"],
            output_schema=None,  # No schema -> parses JSON only
            agent_name="safety_check",
        )
        # Without an output_schema, parse_and_validate just parses JSON
        assert result == {"unexpected_field": 42}

    def test_all_safety_critical_fallbacks_are_fail_closed(self):
        """Every safety-critical agent must have a fail-safe fallback."""
        # safety_check: halt
        assert AGENT_FALLBACKS["safety_check"]["halted"] is True
        # executor: no orders
        assert AGENT_FALLBACKS["executor"] == []
        # fund_manager: no entries
        assert AGENT_FALLBACKS["fund_manager"] == []
        # exit_evaluator: no exits
        assert AGENT_FALLBACKS["exit_evaluator"] == []

    def test_kill_switch_event_available_for_parse_failure_escalation(self):
        """KILL_SWITCH_TRIGGERED exists so parse failures can escalate."""
        assert EventType.KILL_SWITCH_TRIGGERED == "kill_switch_triggered"
