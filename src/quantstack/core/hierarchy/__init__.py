"""Hierarchical regime and trend classification module."""

from quantstack.core.hierarchy.alignment import AlignmentResult, HierarchicalAlignment
from quantstack.core.hierarchy.cascade import SignalCascade
from quantstack.core.hierarchy.regime_classifier import (
    RegimeContext,
    RegimeType,
    WeeklyRegimeClassifier,
)
from quantstack.core.hierarchy.swing_context import (
    SwingContext,
    SwingContextAnalyzer,
    SwingPhase,
)
from quantstack.core.hierarchy.trend_filter import DailyTrendFilter, TrendDirection
from quantstack.core.hierarchy.wave_context import (
    MultiTimeframeWaveContext,
    WaveContextAnalyzer,
    WaveContextSummary,
)

__all__ = [
    "WeeklyRegimeClassifier",
    "RegimeType",
    "RegimeContext",
    "DailyTrendFilter",
    "TrendDirection",
    "SwingContextAnalyzer",
    "SwingPhase",
    "SwingContext",
    "HierarchicalAlignment",
    "AlignmentResult",
    "SignalCascade",
    "WaveContextAnalyzer",
    "WaveContextSummary",
    "MultiTimeframeWaveContext",
]
