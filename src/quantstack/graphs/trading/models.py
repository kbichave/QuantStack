"""Typed output models for every trading graph node.

Each model defines the exact subset of TradingState fields that a node
writes. ``extra="forbid"`` prevents writes to fields the node doesn't own.
``safe_default()`` provides a typed neutral fallback for circuit breaker /
error blocking use.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MarketIntelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    market_context: dict = {}
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> MarketIntelOutput:
        return cls(market_context={})


class EarningsAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    earnings_analysis: dict = {}
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> EarningsAnalysisOutput:
        return cls(earnings_analysis={})


class DataRefreshOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_refresh_summary: dict = {}
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> DataRefreshOutput:
        return cls(
            data_refresh_summary={"skipped": True, "reason": "node_unavailable"},
            errors=["[data_refresh] node unavailable (circuit breaker or failure)"],
        )


class SafetyCheckOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> SafetyCheckOutput:
        return cls(
            decisions=[{"node": "safety_check", "halted": True, "error": "node_unavailable"}],
            errors=["[safety_check] node unavailable (circuit breaker or failure)"],
        )


class PlanDayOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    daily_plan: str = ""
    earnings_symbols: list[str] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> PlanDayOutput:
        return cls(daily_plan="Plan unavailable — using neutral bias")


class PositionReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position_reviews: list[dict] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> PositionReviewOutput:
        return cls(
            position_reviews=[],
            errors=["[position_review] node unavailable (circuit breaker or failure)"],
        )


class ExecuteExitsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exit_orders: list[dict] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> ExecuteExitsOutput:
        return cls(
            exit_orders=[],
            errors=["[execute_exits] node unavailable (circuit breaker or failure)"],
        )


class EntryScanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_candidates: list[dict] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> EntryScanOutput:
        return cls(entry_candidates=[])


class MergeParallelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def safe_default(cls) -> MergeParallelOutput:
        return cls()


class MergePreExecutionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def safe_default(cls) -> MergePreExecutionOutput:
        return cls()


class ResolveSymbolConflictsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_candidates: list[dict] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> ResolveSymbolConflictsOutput:
        return cls(entry_candidates=[])


class RiskSizingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha_signals: list = []
    alpha_signal_candidates: list = []
    vol_state: str = "normal"
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> RiskSizingOutput:
        return cls(
            alpha_signals=[],
            alpha_signal_candidates=[],
            errors=["[risk_sizing] node unavailable (circuit breaker or failure)"],
        )


class PortfolioConstructionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    portfolio_target_weights: dict = {}
    risk_verdicts: list[dict] = []
    last_covariance: list = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> PortfolioConstructionOutput:
        return cls(portfolio_target_weights={}, risk_verdicts=[], last_covariance=[])


class PortfolioReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fund_manager_decisions: list[dict] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> PortfolioReviewOutput:
        return cls(fund_manager_decisions=[])


class OptionsAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    options_analysis: list[dict] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> OptionsAnalysisOutput:
        return cls(options_analysis=[])


class ExecuteEntriesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_orders: list[dict] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> ExecuteEntriesOutput:
        return cls(entry_orders=[])


class ReflectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reflection: str = ""
    trade_quality_scores: list[dict] = []
    attribution_contexts: dict = {}
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> ReflectionOutput:
        return cls(reflection="Reflection skipped")
