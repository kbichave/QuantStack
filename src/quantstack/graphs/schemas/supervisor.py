"""Output schemas for supervisor graph agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthCheckOutput(BaseModel):
    """Output schema for health_monitor agent."""

    overall: str = "unknown"
    checks: list[dict[str, Any]] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)


class DiagnosticOutput(BaseModel):
    """Output schema for diagnostic sub-steps."""

    root_cause: str = ""
    severity: str = "unknown"
    affected_components: list[str] = Field(default_factory=list)


class RecoveryActionOutput(BaseModel):
    """Output schema for self_healer agent."""

    actions: list[dict[str, Any]] = Field(default_factory=list)
    success: bool = False
    summary: str = ""


class PortfolioRiskOutput(BaseModel):
    """Output schema for portfolio_risk_monitor agent."""

    risk_level: str = "unknown"
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class StrategyPromoterOutput(BaseModel):
    """Output schema for strategy_promoter agent."""

    promotions: list[dict[str, Any]] = Field(default_factory=list)
    retirements: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
