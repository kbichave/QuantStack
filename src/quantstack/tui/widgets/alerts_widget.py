"""Alerts widget for the TUI Overview tab.

Displays unresolved system alerts sorted by severity (emergency first).
T1 refresh tier + ALWAYS_ON — alerts surface regardless of active tab.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.text import Text

from quantstack.db import pg_conn
from quantstack.tui.base import RefreshableWidget

# Severity → Rich style mapping
SEVERITY_STYLES: dict[str, str] = {
    "emergency": "bold red",
    "critical": "red",
    "warning": "yellow",
    "info": "dim",
}

_ALERTS_QUERY = """\
SELECT id, category, severity, status, title, created_at
FROM system_alerts
WHERE status != 'resolved'
ORDER BY
    CASE severity
        WHEN 'emergency' THEN 1
        WHEN 'critical' THEN 2
        WHEN 'warning' THEN 3
        WHEN 'info' THEN 4
    END,
    created_at DESC
LIMIT 20
"""


def _format_age(seconds: float) -> str:
    """Human-readable age string from seconds elapsed."""
    s = int(seconds)
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


class AlertsCompact(RefreshableWidget):
    """Unresolved system alerts — severity-sorted, color-coded."""

    REFRESH_TIER = "T1"
    TAB_ID = "tab-overview"
    ALWAYS_ON = True

    def fetch_data(self) -> Any:
        with pg_conn() as conn:
            rows = conn.execute(_ALERTS_QUERY).fetchall()
        return [
            {
                "id": r[0],
                "category": r[1],
                "severity": r[2],
                "status": r[3],
                "title": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    def update_view(self, data: Any) -> None:
        if not data:
            self.update(Text("No active alerts", style="dim"))
            return

        now = datetime.now()
        result = Text()
        for i, alert in enumerate(data[:5]):
            if i > 0:
                result.append("\n")
            style = SEVERITY_STYLES.get(alert["severity"], "dim")
            sev = alert["severity"].upper()[:4]
            title = alert["title"][:60]
            age = ""
            if alert.get("created_at"):
                created = alert["created_at"]
                if hasattr(created, "replace"):
                    created = created.replace(tzinfo=None)
                elapsed = (now - created).total_seconds()
                age = _format_age(elapsed)
            result.append(f"[{sev}] ", style=style)
            result.append(title)
            if age:
                result.append(f"  {age}", style="dim")
        self.update(result)
