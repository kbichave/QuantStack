# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Structured output schemas for research pods.

Every pod produces typed, validated output — not prose. The deterministic
pipeline consumes these schemas directly. If the LLM produces invalid
output, Pydantic rejects it and the pod returns a safe default.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Alpha Researcher Schemas
# =============================================================================


class Hypothesis(BaseModel):
    """A single testable hypothesis from the Alpha Researcher."""

    hypothesis_id: str = ""
    thesis: str = Field(..., description="Clear, testable statement about a market relationship")
    source: Literal[
        "factor_decay", "regime_gap", "cross_asset", "literature",
        "feature_interaction", "failure_analysis", "shap_pattern",
    ] = "literature"
    target_regimes: list[str] = Field(default_factory=lambda: ["trending_up", "ranging"])
    target_symbols: list[str] = Field(default_factory=lambda: ["SPY"])
    entry_rules: list[dict[str, Any]] = Field(default_factory=list)
    exit_rules: list[dict[str, Any]] = Field(default_factory=list)
    feature_tiers: list[str] = Field(default_factory=lambda: ["technical"])
    expected_sharpe: float = 0.5
    priority: int = Field(default=5, ge=1, le=10)
    rationale: str = ""


class FailureAnalysis(BaseModel):
    """Structured analysis of why an experiment failed."""

    experiment_id: str
    root_cause: Literal[
        "weak_signal", "infrequent_trades", "regime_mismatch",
        "overfitting", "feature_noise", "label_mismatch",
        "execution_slippage", "correlation_with_existing", "unknown",
    ]
    evidence: str  # What data supports this diagnosis
    lesson: str  # What we learned
    next_action: Literal["modify_and_retry", "abandon", "pivot_hypothesis", "need_more_data"]
    modifications: list[dict[str, Any]] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Full output of the Alpha Researcher pod."""

    plan_id: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=5)
    failure_analyses: list[FailureAnalysis] = Field(default_factory=list)
    investigations_to_abandon: list[str] = Field(default_factory=list)
    breakthrough_features: list[str] = Field(default_factory=list)
    context_summary: str = ""  # Brief summary of what the researcher observed


# =============================================================================
# ML Scientist Schemas
# =============================================================================


class MLExperiment(BaseModel):
    """A single experiment proposed by the ML Scientist."""

    experiment_id: str = ""
    experiment_type: Literal[
        "retrain", "feature_ablation", "label_experiment",
        "architecture_change", "ensemble_experiment", "hyperparameter_tune",
        "incremental_update", "feature_tier_change",
    ]
    symbol: str
    hypothesis: str  # "Removing low-SHAP features will improve OOS AUC"
    success_criteria: str  # "AUC improves by >0.02 OOS"
    failure_analysis_plan: str  # "If AUC drops, check which features were important"
    config: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)


class MLExperimentPlan(BaseModel):
    """Full output of the ML Scientist pod."""

    plan_id: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    experiments: list[MLExperiment] = Field(default_factory=list, max_length=5)
    retrain_symbols: list[str] = Field(default_factory=list)
    feature_drops: list[str] = Field(default_factory=list)
    model_promotions: list[dict[str, Any]] = Field(default_factory=list)
    context_summary: str = ""


# =============================================================================
# Execution Researcher Schemas
# =============================================================================


class ExecutionRecommendation(BaseModel):
    """A single recommendation from the Execution Researcher."""

    recommendation_type: Literal[
        "timing", "sizing", "correlation_warning",
        "factor_exposure", "regime_allocation", "slippage_pattern",
    ]
    description: str
    action: str  # Specific action to take
    evidence: str  # Data supporting this recommendation
    impact_estimate: str  # Expected P&L impact
    priority: Literal["critical", "high", "medium", "low"] = "medium"


class ExecutionResearchPlan(BaseModel):
    """Full output of the Execution Researcher pod."""

    plan_id: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    recommendations: list[ExecutionRecommendation] = Field(default_factory=list)
    strategy_correlations: dict[str, float] = Field(default_factory=dict)
    factor_exposure: dict[str, float] = Field(default_factory=dict)
    avg_slippage_bps: float = 0.0
    worst_time_of_day: str = ""
    context_summary: str = ""
