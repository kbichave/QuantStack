"""LangGraph state schemas for all graph pipelines.

Each TypedDict defines the complete state contract for a graph. Nodes
read fields they need and return dicts with the fields they update.

Append-only fields use Annotated[list[T], operator.add] so values
accumulate across nodes rather than being overwritten. Nodes must
return [] (not None) for append-only fields they don't want to update,
or omit the field entirely from their return dict.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    """State for a single research pipeline cycle."""

    # Input
    cycle_number: int
    regime: str
    # Pipeline
    context_summary: str
    selected_domain: str
    selected_symbols: list[str]
    hypothesis: str
    validation_result: dict
    backtest_id: str
    ml_experiment_id: str
    registered_strategy_id: str
    # Accumulation
    errors: Annotated[list[str], operator.add]
    decisions: Annotated[list[dict], operator.add]


class TradingState(TypedDict):
    """State for a single trading pipeline cycle."""

    # Input
    cycle_number: int
    regime: str
    portfolio_context: dict
    # Pipeline
    daily_plan: str
    position_reviews: list[dict]
    exit_orders: list[dict]
    entry_candidates: list[dict]
    risk_verdicts: list[dict]
    fund_manager_decisions: list[dict]
    options_analysis: list[dict]
    entry_orders: list[dict]
    reflection: str
    # Accumulation
    errors: Annotated[list[str], operator.add]
    decisions: Annotated[list[dict], operator.add]


class SupervisorState(TypedDict):
    """State for a single supervisor pipeline cycle."""

    cycle_number: int
    health_status: dict
    diagnosed_issues: list[dict]
    recovery_actions: list[dict]
    strategy_lifecycle_actions: list[dict]
    scheduled_task_results: list[dict]
    # Accumulation
    errors: Annotated[list[str], operator.add]
