# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic models for the compliance audit trail.

Every agent decision — from IC analysis through to SuperTrader execution —
is captured as a DecisionEvent and stored append-only in PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation by an agent."""

    tool_name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""  # Truncated to 500 chars — not full data
    latency_ms: int | None = None
    success: bool = True
    error: str | None = None


class IndicatorAttribution(BaseModel):
    """
    Records which indicator contributed to an agent's decision and how strongly.

    Analogous to SHAP values for tree models: positive weight = indicator
    pushed toward the action taken; negative = indicator disagreed.

    Population: filled by the agent or post-processor that has access to
    the raw indicator values before they enter the LLM context.
    """

    indicator: str  # e.g. "RSI_14", "MACD_signal", "ADX"
    value: float  # Raw computed value
    signal: str  # "bullish", "bearish", "neutral"
    weight: float  # 0..1 — how strongly this indicator drove the decision
    threshold: float | None = (
        None  # Reference threshold used (e.g. RSI > 70 = overbought)
    )


class DecisionEvent(BaseModel):
    """
    A single agent decision captured for compliance audit.

    Immutable once created. Stored append-only.

    Hierarchy:
        IC analysis → Pod synthesis → Assistant brief → SuperTrader decision
    """

    # Identity
    event_id: str  # UUID
    session_id: str  # Groups all events in one trading session
    event_type: str  # "ic_analysis", "pod_synthesis", "assistant_brief",
    #                                #  "super_trader_decision", "execution", "risk_rejection"

    # Who
    agent_name: str
    agent_role: str  # "ic", "pod_manager", "assistant", "super_trader", "risk_gate"

    # What
    symbol: str | None = None
    action: str | None = None  # "buy", "sell", "hold", "close", None for analysis-only
    confidence: float | None = None

    # Input context
    input_context_hash: str = ""  # SHA256 of the full input context (not stored in DB)
    market_data_snapshot: dict[str, Any] = Field(
        default_factory=dict
    )  # Key metrics only
    portfolio_snapshot: dict[str, Any] = Field(default_factory=dict)  # Positions + cash

    # Tool calls made during this decision
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # Output
    output_summary: str = ""  # Human-readable summary of the decision
    output_structured: dict[str, Any] = Field(
        default_factory=dict
    )  # Pydantic model dump

    # Risk
    risk_approved: bool | None = None
    risk_violations: list[str] = Field(default_factory=list)

    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    decision_latency_ms: int | None = None

    # Lineage (trace a decision back to its source ICs)
    parent_event_ids: list[str] = Field(default_factory=list)

    # Attribution — which indicators drove this decision (SHAP-style)
    # Populated for IC-level and pod-level events where indicator values are available.
    indicator_attributions: list[IndicatorAttribution] = Field(default_factory=list)

    # IC dissent — for pod_synthesis and super_trader_decision events:
    # list of ICs that disagreed with the consensus action.
    # Format: "momentum_ic: HOLD (0.45 conf)"
    # Used to highlight low-agreement situations so the system scales down position size.
    ic_dissent: list[str] = Field(default_factory=list)


class AuditQuery(BaseModel):
    """Query parameters for audit log searches."""

    symbol: str | None = None
    agent_name: str | None = None
    event_type: str | None = None
    action: str | None = None
    session_id: str | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    limit: int = 100
