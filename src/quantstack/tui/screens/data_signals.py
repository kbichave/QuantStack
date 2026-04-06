"""Data & Signals tab container."""
from textual.app import ComposeResult
from textual.containers import ScrollableContainer

from quantstack.tui.widgets.data_signals import (
    DataHealthMatrixWidget,
    MarketCalendarWidget,
    SignalEngineWidget,
)


class DataSignalsTab(ScrollableContainer):
    """Data & Signals tab — calendar, health matrix, signal engine."""

    def compose(self) -> ComposeResult:
        yield MarketCalendarWidget()
        yield DataHealthMatrixWidget()
        yield SignalEngineWidget()
