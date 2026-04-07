"""Tests for kill switch -> EventBus publication."""
import pytest
from quantstack.coordination.event_bus import EventType


def test_kill_switch_triggered_in_event_type():
    """KILL_SWITCH_TRIGGERED is a valid EventType member."""
    assert hasattr(EventType, "KILL_SWITCH_TRIGGERED")
    assert EventType.KILL_SWITCH_TRIGGERED.value == "kill_switch_triggered"


def test_kill_switch_triggered_is_string_enum():
    """KILL_SWITCH_TRIGGERED participates in string comparisons (str, Enum)."""
    assert EventType.KILL_SWITCH_TRIGGERED == "kill_switch_triggered"
    assert isinstance(EventType.KILL_SWITCH_TRIGGERED, str)
