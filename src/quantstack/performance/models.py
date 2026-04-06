"""Pydantic / TypedDict models for the performance subsystem."""

from __future__ import annotations

from typing import TypedDict


class TradeQualityScore(TypedDict):
    """Structured quality assessment for a closed trade.

    Produced by the LLM-as-judge evaluator in the reflection node.
    All float fields are 0-1 where 1 is best.
    """

    execution_quality: float  # fill quality, slippage, order management
    thesis_accuracy: float  # did the entry thesis play out?
    risk_management: float  # risk contained within parameters?
    timing_quality: float  # entry/exit timing relative to the move
    sizing_quality: float  # position sizing vs conviction level
    overall_score: float  # composite assessment
    justification: str  # LLM reasoning for the scores


class CommunityDiscovery(TypedDict):
    """Structured output from the community intel 3-pass scan.

    Each discovery includes which iteration found it (1=broad scan,
    2=gap-fill, 3=refinement).
    """

    title: str
    source: str  # URL or platform identifier
    category: str  # "strategy", "indicator", "model", "risk_technique"
    asset_class: str  # "equity", "options", "crypto", "multi-asset", etc.
    summary: str
    empirical_evidence: str  # what backtest/paper results support this
    implementation_path: str  # how to implement with available data/tools
    novelty_vs_registry: str  # how it differs from strategies already in registry
    iteration_found: int  # 1 (initial scan), 2 (gap-fill), or 3 (refinement)
