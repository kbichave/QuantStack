"""Event-based labeling module for trade outcome classification."""

from quantstack.core.labeling.event_labeler import EventLabeler, TradeOutcome
from quantstack.core.labeling.wave_event_labeler import (
    WaveEventLabel,
    WaveEventLabeler,
    WavePerformanceAnalyzer,
)

__all__ = [
    "EventLabeler",
    "TradeOutcome",
    "WaveEventLabeler",
    "WaveEventLabel",
    "WavePerformanceAnalyzer",
]
