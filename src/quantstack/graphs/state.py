"""LangGraph state schemas for all graph pipelines.

Each Pydantic BaseModel defines the complete state contract for a graph.
Nodes read fields they need and return dicts with the fields they update.

``extra="forbid"`` catches typos and undeclared keys at merge time rather
than downstream when a stale value causes a bad trade.

Append-only fields use ``Annotated[list[T], operator.add]`` so values
accumulate across nodes rather than being overwritten. Nodes must
return [] (not None) for append-only fields they don't want to update,
or omit the field entirely from their return dict.
"""
from __future__ import annotations

import operator
from typing import Annotated

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_VALID_VOL_STATES = {"", "low", "normal", "high", "extreme"}
_VALID_REGIMES = {"", "trending_up", "trending_down", "ranging", "unknown"}


class _DictCompatMixin:
    """Provides dict-like access for Pydantic models used as LangGraph state."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


class ResearchState(_DictCompatMixin, BaseModel):
    """State for a single research pipeline cycle."""

    model_config = ConfigDict(extra="forbid")

    # Input
    cycle_number: int = 0
    regime: str = ""
    regime_detail: dict = {}
    # Pipeline
    context_summary: str = ""
    selected_domain: str = ""
    selected_symbols: list[str] = []
    hypothesis: str = ""
    validation_result: dict = {}
    backtest_id: str = ""
    ml_experiment_id: str = ""
    registered_strategy_id: str = ""
    # Self-critique loop (WI-8)
    hypothesis_confidence: float = 0.0
    hypothesis_critique: str = ""
    hypothesis_attempts: int = 0
    # Fan-out accumulator (WI-7)
    validation_results: Annotated[list[dict], operator.add] = []
    # research_queue task_ids consumed this cycle (marked done in knowledge_update)
    queued_task_ids: list[str] = []
    # Budget discipline (AR-9)
    token_budget_remaining: int = 50_000
    cost_budget_remaining: float = 0.50
    tokens_consumed: int = 0
    cost_consumed: float = 0.0
    # Accumulation
    errors: Annotated[list[str], operator.add] = []
    decisions: Annotated[list[dict], operator.add] = []

    @field_validator("cycle_number")
    @classmethod
    def _cycle_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("cycle_number must be >= 0")
        return v

    @field_validator("regime")
    @classmethod
    def _regime_known(cls, v: str) -> str:
        if v and v not in _VALID_REGIMES:
            raise ValueError(f"regime must be one of {_VALID_REGIMES}, got {v!r}")
        return v


class SymbolValidationState(BaseModel):
    """State received by each Send()-spawned validation worker (WI-7)."""

    model_config = ConfigDict(extra="forbid")

    symbol_hypothesis: dict = {}
    validation_results: Annotated[list[dict], operator.add] = []
    errors: Annotated[list[str], operator.add] = []


class TradingState(_DictCompatMixin, BaseModel):
    """State for a single trading pipeline cycle."""

    model_config = ConfigDict(extra="forbid")

    # Input
    cycle_number: int = 0
    regime: str = ""
    portfolio_context: dict = {}
    # Data refresh (runs before safety_check every cycle)
    data_refresh_summary: dict = {}
    # Pre-market intelligence (populated by market_intel node)
    market_context: dict = {}
    # Earnings detection (populated by plan_day when earnings within 14 days)
    earnings_symbols: list[str] = []
    earnings_analysis: dict = {}
    # Pipeline
    daily_plan: str = ""
    position_reviews: list[dict] = []
    exit_orders: list[dict] = []
    entry_candidates: list[dict] = []
    risk_verdicts: list[dict] = []
    fund_manager_decisions: list[dict] = []
    options_analysis: list[dict] = []
    entry_orders: list[dict] = []
    reflection: str = ""
    # Post-trade quality scoring (populated by reflection node)
    trade_quality_scores: list[dict] = []
    # Portfolio construction (deterministic optimizer output)
    portfolio_target_weights: dict = {}
    # Covariance from previous cycle, stored as nested list (ndarray not JSON-serializable)
    last_covariance: list = []
    # Volatility state for hysteresis across cycles
    vol_state: str = ""
    # IC-adjusted Kelly alpha signal values (written by risk_sizing, read by risk gate router + portfolio_construction)
    alpha_signals: list = []
    # Filtered entry candidates corresponding 1:1 with alpha_signals (written by risk_sizing)
    alpha_signal_candidates: list[dict] = []
    # P&L attribution context injected into reflection prompt; symbol -> aggregated summary
    attribution_contexts: dict = {}
    # Compaction briefs (written by compaction nodes, read by downstream agents)
    parallel_brief: dict | None = None
    pre_execution_brief: dict | None = None
    # Cycle P&L attribution (populated by attribution_node after reflect)
    cycle_attribution: dict = {}
    # Operating mode set by safety_check router (market/extended/overnight/weekend)
    operating_mode: str = ""
    # Budget discipline (AR-9)
    token_budget_remaining: int = 30_000
    cost_budget_remaining: float = 0.20
    tokens_consumed: int = 0
    cost_consumed: float = 0.0
    # Accumulation
    errors: Annotated[list[str], operator.add] = []
    decisions: Annotated[list[dict], operator.add] = []

    @field_validator("cycle_number")
    @classmethod
    def _cycle_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("cycle_number must be >= 0")
        return v

    @field_validator("regime")
    @classmethod
    def _regime_known(cls, v: str) -> str:
        if v and v not in _VALID_REGIMES:
            raise ValueError(f"regime must be one of {_VALID_REGIMES}, got {v!r}")
        return v

    @field_validator("vol_state")
    @classmethod
    def _vol_state_known(cls, v: str) -> str:
        if v not in _VALID_VOL_STATES:
            raise ValueError(f"vol_state must be one of {_VALID_VOL_STATES}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _exit_orders_require_reviews(self) -> TradingState:
        if self.exit_orders and not self.position_reviews:
            raise ValueError(
                "exit_orders is non-empty but position_reviews is empty — "
                "exits must derive from position reviews"
            )
        return self


class SupervisorState(_DictCompatMixin, BaseModel):
    """State for a single supervisor pipeline cycle."""

    model_config = ConfigDict(extra="forbid")

    cycle_number: int = 0
    health_status: dict = {}
    diagnosed_issues: list[dict] = []
    recovery_actions: list[dict] = []
    strategy_pipeline_report: dict = {}  # draft -> backtested pipeline results
    strategy_lifecycle_actions: list[dict] = []
    scheduled_task_results: list[dict] = []
    eod_refresh_summary: dict = {}
    risk_snapshot: dict = {}  # populated by risk monitoring nodes
    # Accumulation
    errors: Annotated[list[str], operator.add] = []

    @field_validator("cycle_number")
    @classmethod
    def _cycle_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("cycle_number must be >= 0")
        return v
