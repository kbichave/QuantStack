"""Typed output models for every research graph node.

Each model defines the exact subset of ResearchState fields that a node
writes. ``extra="forbid"`` prevents writes to fields the node doesn't own.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ContextLoadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context_summary: str = ""
    regime: str = ""
    regime_detail: dict = {}
    hypothesis_attempts: int = 0
    hypothesis_confidence: float = 0.0
    hypothesis_critique: str = ""
    queued_task_ids: list[str] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> ContextLoadOutput:
        return cls(context_summary="Context unavailable", regime="unknown", regime_detail={})


class DomainSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_domain: str = ""
    selected_symbols: list[str] = []
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> DomainSelectionOutput:
        return cls(selected_domain="swing", selected_symbols=[])


class HypothesisGenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis: str = ""
    hypothesis_attempts: int = 1
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> HypothesisGenerationOutput:
        return cls(hypothesis="", hypothesis_attempts=1)


class HypothesisCritiqueOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis_confidence: float = 0.0
    hypothesis_critique: str = ""
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> HypothesisCritiqueOutput:
        return cls(hypothesis_confidence=0.0, hypothesis_critique="Critique unavailable")


class SignalValidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    validation_result: dict = {}
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> SignalValidationOutput:
        return cls(validation_result={"passed": False, "reason": "node_unavailable"})


class BacktestValidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backtest_id: str = ""
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> BacktestValidationOutput:
        return cls(backtest_id="")


class MlExperimentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ml_experiment_id: str = ""
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> MlExperimentOutput:
        return cls(ml_experiment_id="")


class StrategyRegistrationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registered_strategy_id: str = ""
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> StrategyRegistrationOutput:
        return cls(registered_strategy_id="")


class KnowledgeUpdateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> KnowledgeUpdateOutput:
        return cls()


class ValidateSymbolOutput(BaseModel):
    """Output for Send()-spawned validation workers (SymbolValidationState)."""
    model_config = ConfigDict(extra="forbid")
    validation_results: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> ValidateSymbolOutput:
        return cls(validation_results=[])


class FilterResultsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    validation_result: dict = {}
    decisions: list[dict] = []

    @classmethod
    def safe_default(cls) -> FilterResultsOutput:
        return cls(validation_result={"passed": False, "reason": "filter unavailable"})
