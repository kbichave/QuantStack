"""Output schemas for trading graph agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MarketIntelOutput(BaseModel):
    """Output schema for market_intel agent."""

    headlines: list[str] = Field(default_factory=list)
    risk_alerts: list[str] = Field(default_factory=list)
    event_calendar: list[dict[str, Any]] = Field(default_factory=list)
    sector_news: dict[str, Any] = Field(default_factory=dict)
    sentiment: str = "neutral"


class DailyPlanOutput(BaseModel):
    """Output schema for daily_planner agent."""

    plan: str = ""
    entry_candidates: list[dict[str, Any]] = Field(default_factory=list)
    exit_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    risk_assessment: str = ""


class SafetyCheckOutput(BaseModel):
    """Output schema for safety_check agent.

    CRITICAL: Fail-safe default is halted=True. A parse failure must
    HALT the system, never allow it to proceed.
    """

    halted: bool = True  # Fail CLOSED
    reason: str = ""


class PositionReviewOutput(BaseModel):
    """Output schema for position_monitor agent."""

    analyses: list[dict[str, Any]] = Field(default_factory=list)


class ExitDecisionOutput(BaseModel):
    """Output schema for exit_evaluator agent."""

    exits: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: str = ""


class TradeDebateOutput(BaseModel):
    """Output schema for trade_debater agent."""

    reviews: list[dict[str, Any]] = Field(default_factory=list)


class EntrySignalOutput(BaseModel):
    """Output schema for fund_manager / entry_scan agent."""

    signals: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: str = ""


class OptionsAnalysisOutput(BaseModel):
    """Output schema for options_analyst agent."""

    analysis: list[dict[str, Any]] = Field(default_factory=list)


class EarningsAnalysisOutput(BaseModel):
    """Output schema for earnings_analyst agent."""

    analysis: list[dict[str, Any]] = Field(default_factory=list)


class ExecutionOrderOutput(BaseModel):
    """Output schema for executor agent."""

    orders: list[dict[str, Any]] = Field(default_factory=list)


class TradeReflectionOutput(BaseModel):
    """Output schema for trade_reflector agent."""

    reflection: str = ""
    lessons: list[str] = Field(default_factory=list)
    adjustments: list[dict[str, Any]] = Field(default_factory=list)
