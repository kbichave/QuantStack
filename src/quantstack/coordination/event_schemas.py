"""Payload schemas for Phase 10 event bus events.

Each TypedDict documents the expected payload structure for a Phase 10
EventType. Publishers should construct payloads matching these schemas.
Consumers can use ``validate_payload()`` for defensive checks.

The event bus itself stores payloads as JSONB and does not enforce schemas.
"""

from __future__ import annotations

from typing import TypedDict

from quantstack.coordination.event_bus import EventType


class ToolAddedPayload(TypedDict):
    tool_name: str
    source: str  # "synthesis" or "manual"


class ToolDisabledPayload(TypedDict):
    tool_name: str
    reason: str
    success_rate: float


class ExperimentCompletedPayload(TypedDict):
    experiment_id: str
    status: str  # "winner" or "rejected"
    oos_ic: float | None


class FeatureDecayedPayload(TypedDict):
    feature_id: str
    psi: float
    ic_current: float


class FeatureReplacedPayload(TypedDict):
    old_feature_id: str
    new_feature_id: str


class MandateIssuedPayload(TypedDict):
    mandate_id: str
    key_directives: str


class MetaOptimizationAppliedPayload(TypedDict):
    agent_id: str
    change_type: str  # "prompt", "threshold", or "tool"
    change_summary: str


class ConsensusRequiredPayload(TypedDict):
    signal_id: str
    symbol: str
    notional: float


class ConsensusReachedPayload(TypedDict):
    decision_id: str
    consensus_level: str  # "unanimous", "majority", or "minority"
    final_sizing: float


EVENT_PAYLOAD_SCHEMAS: dict[EventType, type] = {
    EventType.TOOL_ADDED: ToolAddedPayload,
    EventType.TOOL_DISABLED: ToolDisabledPayload,
    EventType.EXPERIMENT_COMPLETED: ExperimentCompletedPayload,
    EventType.FEATURE_DECAYED: FeatureDecayedPayload,
    EventType.FEATURE_REPLACED: FeatureReplacedPayload,
    EventType.MANDATE_ISSUED: MandateIssuedPayload,
    EventType.META_OPTIMIZATION_APPLIED: MetaOptimizationAppliedPayload,
    EventType.CONSENSUS_REQUIRED: ConsensusRequiredPayload,
    EventType.CONSENSUS_REACHED: ConsensusReachedPayload,
}


def validate_payload(event_type: EventType, payload: dict) -> bool:
    """Check that required keys are present in the payload.

    Returns True if the payload contains all keys defined in the schema.
    Returns True for event types without a registered schema (no constraint).
    """
    schema = EVENT_PAYLOAD_SCHEMAS.get(event_type)
    if schema is None:
        return True
    required_keys = set(schema.__annotations__)
    return required_keys.issubset(payload.keys())
