"""Tests for Phase 10 event bus extensions.

Each test verifies that a new EventType value can be published and polled
through the existing EventBus infrastructure without modification to the
bus itself.
"""

import pytest
from quantstack.coordination.event_bus import EventBus, Event, EventType
from quantstack.coordination.event_schemas import (
    EVENT_PAYLOAD_SCHEMAS,
    validate_payload,
    ToolAddedPayload,
    ConsensusReachedPayload,
)


# --- Enum membership ---

PHASE10_EVENT_TYPES = [
    "TOOL_ADDED",
    "TOOL_DISABLED",
    "EXPERIMENT_COMPLETED",
    "FEATURE_DECAYED",
    "FEATURE_REPLACED",
    "MANDATE_ISSUED",
    "META_OPTIMIZATION_APPLIED",
    "CONSENSUS_REQUIRED",
    "CONSENSUS_REACHED",
]


@pytest.mark.parametrize("member_name", PHASE10_EVENT_TYPES)
def test_new_event_type_is_valid_enum_member(member_name: str) -> None:
    """Each new EventType value must be a valid enum member."""
    assert hasattr(EventType, member_name)
    member = EventType[member_name]
    assert isinstance(member, EventType)


# --- Payload schema coverage ---

def test_all_phase10_events_have_schemas() -> None:
    """Every Phase 10 event type should have a registered payload schema."""
    for name in PHASE10_EVENT_TYPES:
        et = EventType[name]
        assert et in EVENT_PAYLOAD_SCHEMAS, f"Missing schema for {name}"


# --- validate_payload ---

def test_validate_payload_passes_for_valid_tool_added() -> None:
    payload = {"tool_name": "my_tool", "source": "synthesis"}
    assert validate_payload(EventType.TOOL_ADDED, payload) is True


def test_validate_payload_fails_for_missing_key() -> None:
    payload = {"tool_name": "my_tool"}  # missing 'source'
    assert validate_payload(EventType.TOOL_ADDED, payload) is False


def test_validate_payload_passes_for_unknown_event_type() -> None:
    """Event types without a schema always pass validation."""
    assert validate_payload(EventType.LOOP_HEARTBEAT, {"anything": True}) is True


def test_validate_payload_consensus_reached() -> None:
    payload = {"decision_id": "d1", "consensus_level": "unanimous", "final_sizing": 0.5}
    assert validate_payload(EventType.CONSENSUS_REACHED, payload) is True


# --- Publish + poll round-trip tests (require DB) ---

def test_tool_added_publishes_and_polls(event_bus: EventBus) -> None:
    """TOOL_ADDED event round-trips through publish -> poll."""
    event = Event(
        event_type=EventType.TOOL_ADDED,
        source_loop="supervisor",
        payload={"tool_name": "new_tool", "source": "synthesis"},
    )
    event_bus.publish(event)
    events = event_bus.poll("test_consumer", event_types=[EventType.TOOL_ADDED])
    assert len(events) == 1
    assert events[0].event_type == EventType.TOOL_ADDED
    assert events[0].payload["tool_name"] == "new_tool"


def test_tool_disabled_publishes_and_polls(event_bus: EventBus) -> None:
    """TOOL_DISABLED event round-trips through publish -> poll."""
    event = Event(
        event_type=EventType.TOOL_DISABLED,
        source_loop="supervisor",
        payload={"tool_name": "broken_tool", "reason": "high failure rate", "success_rate": 0.3},
    )
    event_bus.publish(event)
    events = event_bus.poll("test_consumer", event_types=[EventType.TOOL_DISABLED])
    assert len(events) == 1
    assert events[0].payload["success_rate"] == 0.3


def test_experiment_completed_publishes_and_polls(event_bus: EventBus) -> None:
    """EXPERIMENT_COMPLETED event round-trips through publish -> poll."""
    event = Event(
        event_type=EventType.EXPERIMENT_COMPLETED,
        source_loop="research",
        payload={"experiment_id": "exp-1", "status": "winner", "oos_ic": 0.05},
    )
    event_bus.publish(event)
    events = event_bus.poll("test_consumer", event_types=[EventType.EXPERIMENT_COMPLETED])
    assert len(events) == 1
    assert events[0].payload["status"] == "winner"


def test_mandate_issued_publishes_and_polls(event_bus: EventBus) -> None:
    """MANDATE_ISSUED event round-trips through publish -> poll."""
    event = Event(
        event_type=EventType.MANDATE_ISSUED,
        source_loop="supervisor",
        payload={"mandate_id": "m-1", "key_directives": "reduce exposure"},
    )
    event_bus.publish(event)
    events = event_bus.poll("test_consumer", event_types=[EventType.MANDATE_ISSUED])
    assert len(events) == 1
    assert events[0].payload["mandate_id"] == "m-1"


def test_meta_optimization_applied_publishes_and_polls(event_bus: EventBus) -> None:
    """META_OPTIMIZATION_APPLIED event round-trips through publish -> poll."""
    event = Event(
        event_type=EventType.META_OPTIMIZATION_APPLIED,
        source_loop="supervisor",
        payload={"agent_id": "swing_agent", "change_type": "prompt", "change_summary": "shorter"},
    )
    event_bus.publish(event)
    events = event_bus.poll("test_consumer", event_types=[EventType.META_OPTIMIZATION_APPLIED])
    assert len(events) == 1
    assert events[0].payload["change_type"] == "prompt"


def test_consensus_required_and_reached_publish_and_poll(event_bus: EventBus) -> None:
    """CONSENSUS_REQUIRED and CONSENSUS_REACHED events round-trip correctly."""
    req = Event(
        event_type=EventType.CONSENSUS_REQUIRED,
        source_loop="trading",
        payload={"signal_id": "s1", "symbol": "AAPL", "notional": 10000.0},
    )
    reached = Event(
        event_type=EventType.CONSENSUS_REACHED,
        source_loop="trading",
        payload={"decision_id": "d1", "consensus_level": "unanimous", "final_sizing": 1.0},
    )
    event_bus.publish(req)
    event_bus.publish(reached)
    events = event_bus.poll(
        "test_consumer",
        event_types=[EventType.CONSENSUS_REQUIRED, EventType.CONSENSUS_REACHED],
    )
    assert len(events) == 2
    types = {e.event_type for e in events}
    assert EventType.CONSENSUS_REQUIRED in types
    assert EventType.CONSENSUS_REACHED in types


def test_feature_decayed_and_replaced_publish_and_poll(event_bus: EventBus) -> None:
    """FEATURE_DECAYED and FEATURE_REPLACED events round-trip correctly."""
    decayed = Event(
        event_type=EventType.FEATURE_DECAYED,
        source_loop="research",
        payload={"feature_id": "f1", "psi": 0.3, "ic_current": 0.01},
    )
    replaced = Event(
        event_type=EventType.FEATURE_REPLACED,
        source_loop="research",
        payload={"old_feature_id": "f1", "new_feature_id": "f2"},
    )
    event_bus.publish(decayed)
    event_bus.publish(replaced)
    events = event_bus.poll(
        "test_consumer",
        event_types=[EventType.FEATURE_DECAYED, EventType.FEATURE_REPLACED],
    )
    assert len(events) == 2
