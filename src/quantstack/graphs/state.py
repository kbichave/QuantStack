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
    regime_detail: dict
    # Pipeline
    context_summary: str
    selected_domain: str
    selected_symbols: list[str]
    hypothesis: str
    validation_result: dict
    backtest_id: str
    ml_experiment_id: str
    registered_strategy_id: str
    # Self-critique loop (WI-8)
    hypothesis_confidence: float
    hypothesis_critique: str
    hypothesis_attempts: int
    # Fan-out accumulator (WI-7)
    validation_results: Annotated[list[dict], operator.add]
    # research_queue task_ids consumed this cycle (marked done in knowledge_update)
    queued_task_ids: list[str]
    # Accumulation
    errors: Annotated[list[str], operator.add]
    decisions: Annotated[list[dict], operator.add]


class SymbolValidationState(TypedDict):
    """State received by each Send()-spawned validation worker (WI-7)."""

    symbol_hypothesis: dict
    validation_results: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]


class TradingState(TypedDict):
    """State for a single trading pipeline cycle."""

    # Input
    cycle_number: int
    regime: str
    portfolio_context: dict
    # Data refresh (runs before safety_check every cycle)
    data_refresh_summary: dict
    # Pre-market intelligence (populated by market_intel node)
    market_context: dict
    # Earnings detection (populated by plan_day when earnings within 14 days)
    earnings_symbols: list[str]
    earnings_analysis: dict
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
    # Post-trade quality scoring (populated by reflection node)
    trade_quality_scores: list[dict]
    # Portfolio construction (deterministic optimizer output)
    portfolio_target_weights: dict
    # Covariance from previous cycle, stored as nested list (ndarray not JSON-serializable)
    last_covariance: list  # [] if none yet
    # Volatility state for hysteresis across cycles ("normal" or "high")
    vol_state: str
    # P&L attribution context injected into reflection prompt; symbol → aggregated summary
    attribution_contexts: dict
    # Accumulation
    errors: Annotated[list[str], operator.add]
    decisions: Annotated[list[dict], operator.add]


class SupervisorState(TypedDict):
    """State for a single supervisor pipeline cycle."""

    cycle_number: int
    health_status: dict
    diagnosed_issues: list[dict]
    recovery_actions: list[dict]
    strategy_pipeline_report: dict  # draft → backtested pipeline results
    strategy_lifecycle_actions: list[dict]
    scheduled_task_results: list[dict]
    eod_refresh_summary: dict
    risk_snapshot: dict  # populated by risk monitoring nodes
    # Accumulation
    errors: Annotated[list[str], operator.add]
