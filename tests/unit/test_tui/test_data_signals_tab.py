"""Tests for Data & Signals tab widgets."""
from datetime import date, datetime

from quantstack.tui.queries.calendar import EarningsEvent
from quantstack.tui.queries.signals import Signal
from quantstack.tui.widgets.data_signals import (
    DataHealthMatrixWidget,
    MarketCalendarWidget,
    SignalEngineWidget,
)


class TestMarketCalendarWidget:
    def test_renders_events(self):
        w = MarketCalendarWidget()
        w.update_view({
            "earnings": [EarningsEvent("AAPL", date(2026, 7, 24), 1.50, None, None)],
        })

    def test_handles_empty(self):
        w = MarketCalendarWidget()
        w.update_view({"earnings": []})

    def test_handles_none(self):
        w = MarketCalendarWidget()
        w.update_view(None)


class TestDataHealthMatrixWidget:
    def test_renders_with_data(self):
        w = DataHealthMatrixWidget()
        now = datetime.now()
        w.update_view({
            "ohlcv": {"AAPL": now, "NVDA": now},
            "news": {"AAPL": now},
            "sentiment": {},
            "options": {},
            "insider": {},
            "macro": {},
        })

    def test_handles_empty(self):
        w = DataHealthMatrixWidget()
        w.update_view(None)


class TestSignalEngineWidget:
    def test_renders_signals(self):
        w = SignalEngineWidget()
        w.update_view([
            Signal("AAPL", "BUY", 0.85, 5.0, datetime.now(), {"ml": 0.7, "sentiment": 0.6}),
        ])

    def test_handles_empty(self):
        w = SignalEngineWidget()
        w.update_view([])

    def test_handles_none(self):
        w = SignalEngineWidget()
        w.update_view(None)
