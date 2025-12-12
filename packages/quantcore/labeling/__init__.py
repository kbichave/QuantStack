"""Event-based labeling module for trade outcome classification."""

from quantcore.labeling.event_labeler import EventLabeler, TradeOutcome
from quantcore.labeling.wave_event_labeler import (
    WaveEventLabeler,
    WaveEventLabel,
    WavePerformanceAnalyzer,
)

__all__ = [
    "EventLabeler",
    "TradeOutcome",
    "WaveEventLabeler",
    "WaveEventLabel",
    "WavePerformanceAnalyzer",
]
