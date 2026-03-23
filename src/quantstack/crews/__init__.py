# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Crew schemas and pure-Python utilities.

TradingCrew (CrewAI) was removed in v0.6.0 — replaced by SignalEngine.
This package now exports only the Pydantic schemas (used system-wide)
and the decoder/registry modules (pure Python, no CrewAI).
"""

from quantstack.shared.schemas import (
    AnalysisNote,
    DailyBrief,
    KeyLevel,
    PodResearchNote,
    RiskVerdict,
    SymbolBrief,
    TaskEnvelope,
    TradeDecision,
)

__all__ = [
    "AnalysisNote",
    "PodResearchNote",
    "DailyBrief",
    "SymbolBrief",
    "TradeDecision",
    "RiskVerdict",
    "KeyLevel",
    "TaskEnvelope",
]
