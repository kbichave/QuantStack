# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 06: EventBus Extension — New Event Types.

Verifies SIGNAL_DEGRADATION, SIGNAL_CONFLICT, and AGENT_DEGRADATION
are valid EventType members with correct string serialization.
"""

from __future__ import annotations

import pytest

from quantstack.coordination.event_bus import EventType


class TestNewEventTypes:
    """Verify new event types exist and serialize correctly."""

    def test_signal_degradation_exists(self):
        assert EventType.SIGNAL_DEGRADATION.value == "signal_degradation"

    def test_signal_conflict_exists(self):
        assert EventType.SIGNAL_CONFLICT.value == "signal_conflict"

    def test_agent_degradation_exists(self):
        assert EventType.AGENT_DEGRADATION.value == "agent_degradation"

    def test_str_compatible(self):
        """EventType inherits from str — direct string comparison works."""
        assert EventType.SIGNAL_DEGRADATION == "signal_degradation"
        assert EventType.SIGNAL_CONFLICT == "signal_conflict"
        assert EventType.AGENT_DEGRADATION == "agent_degradation"

    def test_reconstruct_from_string(self):
        """EventBus stores .value in DB and reconstructs via EventType(etype)."""
        assert EventType("signal_degradation") is EventType.SIGNAL_DEGRADATION
        assert EventType("signal_conflict") is EventType.SIGNAL_CONFLICT
        assert EventType("agent_degradation") is EventType.AGENT_DEGRADATION


class TestExistingEventTypesUnchanged:
    """Regression: pre-existing event types still have correct values."""

    @pytest.mark.parametrize("member,value", [
        ("STRATEGY_PROMOTED", "strategy_promoted"),
        ("STRATEGY_RETIRED", "strategy_retired"),
        ("STRATEGY_DEMOTED", "strategy_demoted"),
        ("MODEL_TRAINED", "model_trained"),
        ("DEGRADATION_DETECTED", "degradation_detected"),
        ("REGIME_CHANGE", "regime_change"),
        ("IC_DECAY", "ic_decay"),
        ("MODEL_DEGRADATION", "model_degradation"),
        ("RISK_WARNING", "risk_warning"),
        ("KILL_SWITCH_TRIGGERED", "kill_switch_triggered"),
    ])
    def test_existing_type_value(self, member, value):
        assert getattr(EventType, member).value == value
