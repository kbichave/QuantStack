"""Hierarchical regime and trend classification module."""

from quantcore.hierarchy.alignment import AlignmentResult, HierarchicalAlignment
from quantcore.hierarchy.cascade import SignalCascade
from quantcore.hierarchy.regime_classifier import (
    RegimeContext,
    RegimeType,
    WeeklyRegimeClassifier,
)
from quantcore.hierarchy.swing_context import (
    SwingContext,
    SwingContextAnalyzer,
    SwingPhase,
)
from quantcore.hierarchy.trend_filter import DailyTrendFilter, TrendDirection
from quantcore.hierarchy.wave_context import (
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
