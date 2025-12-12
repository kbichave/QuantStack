# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic output schemas for CrewAI Trading Crew.

These schemas define structured outputs for agents and tasks,
enabling CrewAI to validate and parse LLM responses automatically.
"""

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# KEY LEVEL
# =============================================================================


class TaskEnvelope(BaseModel):
    """Task + asset metadata shared across pods."""

    asset_class: Literal["equities", "options", "futures", "fx_crypto"] = "equities"
    instrument_type: str = "equity"
    task_intent: Literal[
        "analysis", "backtest", "live_signal", "hedge", "execution_check"
    ] = "analysis"
    symbol: Optional[str] = None
    notes: str = ""
    priority: Literal["routine", "high", "urgent"] = "routine"

    @classmethod
    def from_inputs(cls, inputs: Dict[str, Any]) -> "TaskEnvelope":
        """
        Normalize task envelope from raw inputs.

        Accepts either a TaskEnvelope, a dict under `task_envelope`, or
        top-level keys (asset_class, instrument_type, task_intent).
        """
        if isinstance(inputs.get("task_envelope"), TaskEnvelope):
            return inputs["task_envelope"]

        envelope_data = inputs.get("task_envelope", {}) or {}
        if isinstance(envelope_data, TaskEnvelope):
            return envelope_data

        merged: Dict[str, Any] = {}
        merged.update(envelope_data if isinstance(envelope_data, dict) else {})

        # Pull top-level fallbacks
        merged.setdefault("asset_class", inputs.get("asset_class", "equities"))
        merged.setdefault("instrument_type", inputs.get("instrument_type"))
        merged.setdefault("task_intent", inputs.get("task_intent", "analysis"))
        merged.setdefault("symbol", inputs.get("symbol"))
        merged.setdefault("notes", inputs.get("notes", ""))
        merged.setdefault("priority", inputs.get("priority", "routine"))

        normalized_asset = str(merged.get("asset_class", "equities")).lower()
        if normalized_asset in {"equity", "equities"}:
            normalized_asset = "equities"
        elif normalized_asset in {"option", "options"}:
            normalized_asset = "options"
        elif normalized_asset in {"future", "futures"}:
            normalized_asset = "futures"
        elif normalized_asset in {"fx", "forex", "crypto", "fx_crypto"}:
            normalized_asset = "fx_crypto"
        merged["asset_class"] = normalized_asset

        default_instrument = {
            "equities": "equity",
            "options": "option",
            "futures": "future",
            "fx_crypto": "fx_or_crypto",
        }[normalized_asset]

        merged["instrument_type"] = merged.get("instrument_type") or default_instrument
        merged["task_intent"] = str(merged.get("task_intent", "analysis")).lower()
        merged["priority"] = str(merged.get("priority", "routine")).lower()

        return cls(**merged)


class KeyLevel(BaseModel):
    """A significant price level identified by analysis."""

    price: float
    level_type: Literal["support", "resistance", "pivot", "target", "stop"]
    strength: float = Field(ge=0, le=1, default=0.5)
    source: str = ""  # Which indicator/analysis identified this


# =============================================================================
# ANALYSIS NOTE - Output from individual analysts
# =============================================================================


class AnalysisNote(BaseModel):
    """
    Structured analysis output from an analyst agent.

    This is RAW ANALYSIS, not a trading signal. It provides:
    - Observations about current market state
    - Key levels and zones
    - Directional bias with conviction
    - Detailed rationale
    - Quantitative technical metrics
    """

    symbol: str
    analyst_name: str
    analysis_type: str  # e.g., "trend", "momentum", "volatility", "structure"

    # Market state observations
    current_state: str  # e.g., "uptrend with pullback", "range-bound near resistance"

    # Directional bias (NOT a signal - just the analyst's view)
    bias: Literal["bullish", "bearish", "neutral"]
    bias_conviction: float = Field(ge=0, le=1, default=0.5)

    # Key levels identified
    key_levels: List[KeyLevel] = Field(default_factory=list)

    # Detailed observations
    observations: List[str] = Field(default_factory=list)

    # Risk factors identified
    risk_factors: List[str] = Field(default_factory=list)

    # Full reasoning
    rationale: str = ""

    # Quantitative technical metrics
    technical_metrics: Dict[str, float] = Field(default_factory=dict)
    # e.g., {"rsi_14": 65.2, "adx_14": 28.5, "atr_14": 2.3, "sma_20": 472.5}

    # Time horizon for this analysis
    time_horizon: Literal["intraday", "swing", "position"] = "swing"

    # Metadata
    confidence: float = Field(ge=0, le=1, default=0.5)
    timeframe_focus: str = "daily"
    data_quality: Literal["good", "limited", "stale"] = "good"


# =============================================================================
# POD RESEARCH NOTE - Synthesized output from a pod
# =============================================================================


class PodResearchNote(BaseModel):
    """
    Synthesized research note from an entire analysis pod.

    Combines multiple analyst perspectives into a unified view.
    """

    pod_name: str
    symbol: str
    analysis_date: date

    # Pod's synthesis
    executive_summary: str

    # Combined view
    market_view: str
    bias: Literal["bullish", "bearish", "neutral"]
    conviction: float = Field(ge=0, le=1, default=0.5)

    # Aggregated key levels
    key_levels: List[KeyLevel] = Field(default_factory=list)

    # Key observations
    key_observations: List[str] = Field(default_factory=list)

    # Risk factors
    risk_factors: List[str] = Field(default_factory=list)

    # Conditions that would change the view
    invalidation_conditions: List[str] = Field(default_factory=list)

    # Full reasoning chain
    reasoning: str = ""

    # Aggregated quantitative metrics
    aggregated_metrics: Dict[str, float] = Field(default_factory=dict)

    # Metric consensus interpretation
    metric_consensus: Dict[str, str] = Field(default_factory=dict)
    # e.g., {"momentum": "bullish", "trend_strength": "strong", "volatility": "normal"}

    # Time horizon
    time_horizon: Literal["intraday", "swing", "position"] = "swing"

    # Metadata
    ic_analyses_count: int = 0


# =============================================================================
# SYMBOL BRIEF - Per-symbol consolidated brief
# =============================================================================


class SymbolBrief(BaseModel):
    """
    Consolidated brief for a single symbol.

    Combines analysis from multiple sources into a single view.
    """

    symbol: str

    # Consolidated market view
    market_summary: str

    # Aggregated bias
    consensus_bias: Literal[
        "strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"
    ]
    consensus_conviction: float = Field(ge=0, le=1, default=0.5)

    # Agreement level
    pod_agreement: Literal["unanimous", "strong", "moderate", "mixed", "conflicting"]

    # Key levels (deduplicated)
    critical_levels: List[KeyLevel] = Field(default_factory=list)

    # Most important observations
    key_observations: List[str] = Field(default_factory=list)

    # Risk factors to monitor
    risk_factors: List[str] = Field(default_factory=list)

    # Actionable insights
    actionable_insights: List[str] = Field(default_factory=list)

    # Metadata
    contributing_pods: List[str] = Field(default_factory=list)
    analysis_quality: Literal["high", "medium", "low"] = "medium"


# =============================================================================
# DAILY BRIEF - Chief Strategist output
# =============================================================================


class DailyBrief(BaseModel):
    """
    The Chief Strategist's Daily Brief.

    Primary input to the SuperTrader's decision-making process.
    Contains consolidated research from all analysts.
    """

    date: date

    # Executive summary
    market_overview: str

    # Overall market bias
    market_bias: Literal["bullish", "bearish", "neutral"]
    market_conviction: float = Field(ge=0, le=1, default=0.5)

    # Risk environment
    risk_environment: Literal["low", "normal", "elevated", "high"]

    # Symbol-specific briefs
    symbol_briefs: List[SymbolBrief] = Field(default_factory=list)

    # Top opportunities (ranked)
    top_opportunities: List[str] = Field(default_factory=list)  # Symbol names

    # Top risks
    key_risks: List[str] = Field(default_factory=list)

    # Strategic recommendations
    strategic_notes: str = ""

    # Metadata
    pods_reporting: int = 0
    total_analyses: int = 0
    overall_confidence: float = Field(ge=0, le=1, default=0.5)


# =============================================================================
# TRADE DECISION - SuperTrader output
# =============================================================================


class TradeDecision(BaseModel):
    """Structured output from SuperTrader."""

    action: Literal["buy", "sell", "hold", "close"]
    symbol: str
    confidence: float = Field(ge=0, le=1)
    position_size: Literal["full", "half", "quarter", "none"] = "full"
    holding_period: Literal["intraday", "swing", "position"] = "swing"

    # Execution details
    entry_type: Literal["market", "limit", "stop"] = "market"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Reasoning
    reasoning: str = ""
    signals_used: List[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None

    # MTF context summary
    mtf_alignment: float = 0.5
    dominant_timeframe: str = "daily"


# =============================================================================
# RISK VERDICT - Risk Consultant output
# =============================================================================


class RiskVerdict(BaseModel):
    """
    The Risk Consultant's verdict on a trade proposal.

    Options:
    - APPROVE: Trade can proceed as proposed
    - SCALE: Trade can proceed but with reduced size
    - VETO: Trade is rejected entirely
    """

    status: Literal["APPROVE", "SCALE", "VETO"]

    # For SCALE verdicts
    approved_quantity: int = 0
    scale_factor: float = Field(default=1.0, ge=0, le=1)

    # Risk assessment
    risk_score: float = Field(ge=0, le=1, default=0.5)  # 0=low risk, 1=high risk

    # Reasoning
    reasoning: str = ""
    concerns: List[str] = Field(default_factory=list)

    # Specific limit breaches
    limit_breaches: List[str] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
