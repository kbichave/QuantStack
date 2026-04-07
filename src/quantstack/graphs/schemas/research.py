"""Output schemas for research graph agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QuantResearchOutput(BaseModel):
    """Output schema for quant_researcher agent."""

    summary: str = ""
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class HypothesisOutput(BaseModel):
    """Output schema for hypothesis generation."""

    hypothesis: str = ""
    mechanism: str = ""
    prediction: str = ""
    falsification_criteria: str = ""


class HypothesisCritiqueOutput(BaseModel):
    """Output schema for hypothesis_critic agent."""

    confidence: float = 0.0
    critique: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class BacktestResultOutput(BaseModel):
    """Output schema for execution_researcher agent."""

    backtest_id: str = "bt-unknown"
    passed: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)


class ExperimentResultOutput(BaseModel):
    """Output schema for ml_scientist agent."""

    experiment_id: str = "exp-unknown"
    model_type: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)


class StrategyRegistrationOutput(BaseModel):
    """Output schema for strategy_rd agent."""

    strategy_id: str = "strat-unknown"
    domain: str = ""
    status: str = ""


class CommunityIntelOutput(BaseModel):
    """Output schema for community_intel agent."""

    ideas: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class DomainResearchOutput(BaseModel):
    """Output schema for domain_researcher agent."""

    domain: str = "swing"
    symbols: list[str] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)


class ValidationResultOutput(BaseModel):
    """Output schema for validation steps."""

    passed: bool = False
    reason: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
