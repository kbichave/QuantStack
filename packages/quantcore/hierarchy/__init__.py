"""Hierarchical regime and trend classification module."""

from quantcore.hierarchy.regime_classifier import (
    WeeklyRegimeClassifier,
    RegimeType,
    RegimeContext,
)
from quantcore.hierarchy.trend_filter import DailyTrendFilter, TrendDirection
from quantcore.hierarchy.swing_context import (
    SwingContextAnalyzer,
    SwingPhase,
    SwingContext,
)
from quantcore.hierarchy.alignment import HierarchicalAlignment, AlignmentResult
from quantcore.hierarchy.cascade import SignalCascade
from quantcore.hierarchy.wave_context import (
    WaveContextAnalyzer,
    WaveContextSummary,
    MultiTimeframeWaveContext,
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
