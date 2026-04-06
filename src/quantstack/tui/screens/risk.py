"""Risk console container (embeddable in Overview or standalone)."""
from textual.app import ComposeResult
from textual.containers import ScrollableContainer

from quantstack.tui.widgets.risk import RiskAlertsWidget, RiskEventsWidget, RiskMetricsWidget


class RiskConsole(ScrollableContainer):
    """Risk console — metrics, events, alerts."""

    def compose(self) -> ComposeResult:
        yield RiskMetricsWidget()
        yield RiskEventsWidget()
        yield RiskAlertsWidget()
