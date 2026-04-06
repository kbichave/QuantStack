"""Tests for Risk console widgets."""
from datetime import datetime

from quantstack.tui.queries.risk import EquityAlert, RiskEvent, RiskSnapshot
from quantstack.tui.widgets.risk import RiskAlertsWidget, RiskEventsWidget, RiskMetricsWidget


class TestRiskMetricsWidget:
    def test_renders_snapshot(self):
        w = RiskMetricsWidget()
        snap = RiskSnapshot(0.5, 0.3, 0.2, 0.15, 0.1, 2000, -0.08, datetime.now())
        w.update_view(snap)

    def test_handles_none(self):
        w = RiskMetricsWidget()
        w.update_view(None)

    def test_color_coding(self):
        w = RiskMetricsWidget()
        snap = RiskSnapshot(1.2, 0.9, 0.35, 0.65, 0.45, 6000, -0.20, datetime.now())
        w.update_view(snap)


class TestRiskEventsWidget:
    def test_renders_events(self):
        w = RiskEventsWidget()
        w.update_view([
            RiskEvent("risk_rejection", "TSLA", "Volatility too high", datetime.now()),
        ])

    def test_handles_empty(self):
        w = RiskEventsWidget()
        w.update_view([])

    def test_handles_none(self):
        w = RiskEventsWidget()
        w.update_view(None)


class TestRiskAlertsWidget:
    def test_renders_alerts(self):
        w = RiskAlertsWidget()
        w.update_view([
            EquityAlert(1, "drawdown", "active", "Drawdown exceeds 5%", datetime.now(), None),
        ])

    def test_handles_empty(self):
        w = RiskAlertsWidget()
        w.update_view([])

    def test_handles_none(self):
        w = RiskAlertsWidget()
        w.update_view(None)
