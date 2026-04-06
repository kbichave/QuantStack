"""Risk console widgets — metrics, events, alerts."""
from __future__ import annotations

from typing import Any

from rich.table import Table
from rich.text import Text

from quantstack.db import pg_conn
from quantstack.tui.base import RefreshableWidget


RISK_LIMITS = {
    "gross_exposure": 1.0,
    "net_exposure": 0.8,
    "concentration": 0.3,
    "correlation": 0.6,
    "sector_exposure": 0.4,
    "var_1d": 5000.0,
    "max_drawdown": 0.15,
}


class RiskMetricsWidget(RefreshableWidget):
    """Risk snapshot: current values vs limits with color coding."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.risk import fetch_risk_snapshot
            return fetch_risk_snapshot(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No risk data available", style="dim"))
            return
        table = Table(show_edge=False, box=None, title="Risk Metrics")
        table.add_column("Metric")
        table.add_column("Current", justify="right")
        table.add_column("Limit", justify="right")
        metrics = [
            ("Gross Exposure", data.gross_exposure, RISK_LIMITS["gross_exposure"]),
            ("Net Exposure", data.net_exposure, RISK_LIMITS["net_exposure"]),
            ("Concentration", data.concentration, RISK_LIMITS["concentration"]),
            ("Correlation", data.correlation, RISK_LIMITS["correlation"]),
            ("Sector Exposure", data.sector_exposure, RISK_LIMITS["sector_exposure"]),
            ("VaR (1d)", data.var_1d, RISK_LIMITS["var_1d"]),
            ("Max Drawdown", data.max_drawdown, RISK_LIMITS["max_drawdown"]),
        ]
        for name, value, limit in metrics:
            ratio = abs(value) / limit if limit else 0
            if ratio > 1.0:
                color = "red"
            elif ratio > 0.75:
                color = "yellow"
            else:
                color = "green"
            if name == "VaR (1d)":
                table.add_row(name, Text(f"${value:,.0f}", style=color), f"${limit:,.0f}")
            else:
                table.add_row(name, Text(f"{value:.1%}", style=color), f"{limit:.0%}")
        table.caption = f"Snapshot: {data.snapshot_at.strftime('%H:%M:%S')}" if data.snapshot_at else ""
        self.update(table)


class RiskEventsWidget(RefreshableWidget):
    """Recent risk events (rejections, alerts, breaches)."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.risk import fetch_risk_events
            return fetch_risk_events(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No risk events", style="dim"))
            return
        table = Table(show_edge=False, box=None)
        for col in ["Time", "Type", "Symbol", "Details"]:
            table.add_column(col)
        color_map = {
            "risk_rejection": "red",
            "drawdown_alert": "yellow",
            "correlation_alert": "yellow",
        }
        for ev in data[:20]:
            color = color_map.get(ev.event_type, "")
            table.add_row(
                ev.created_at.strftime("%m/%d %H:%M") if ev.created_at else "?",
                Text(ev.event_type, style=color),
                ev.symbol or "",
                ev.details[:60] if ev.details else "",
            )
        self.update(table)


class RiskAlertsWidget(RefreshableWidget):
    """Equity alerts with clearance status."""

    REFRESH_TIER = "T2"
    TAB_ID = "tab-overview"

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            from quantstack.tui.queries.risk import fetch_equity_alerts
            return fetch_equity_alerts(conn)

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No equity alerts", style="dim"))
            return
        table = Table(show_edge=False, box=None)
        for col in ["ID", "Type", "Status", "Message", "Created"]:
            table.add_column(col)
        for a in data[:20]:
            color = "red" if a.status == "active" else "green"
            table.add_row(
                str(a.alert_id),
                a.alert_type,
                Text(a.status, style=color),
                a.message[:50] if a.message else "",
                a.created_at.strftime("%m/%d %H:%M") if a.created_at else "?",
            )
        self.update(table)
