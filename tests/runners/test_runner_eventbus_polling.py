"""Tests for runner-level EventBus polling."""
import pytest
from quantstack.coordination.event_bus import EventType


def test_kill_switch_event_type_exists():
    """Verify KILL_SWITCH_TRIGGERED exists for runner polling."""
    assert EventType.KILL_SWITCH_TRIGGERED == "kill_switch_triggered"


def test_kill_switch_event_type_in_poll_filter():
    """EventType can be used in a list filter (as runners do)."""
    filter_types = [EventType.KILL_SWITCH_TRIGGERED]
    assert len(filter_types) == 1
    assert filter_types[0].value == "kill_switch_triggered"
