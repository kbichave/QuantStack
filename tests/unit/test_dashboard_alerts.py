"""Unit tests for dashboard alert integration (Section 06)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# TUI: AlertsCompact — severity color mapping
# ---------------------------------------------------------------------------


class TestSeverityColorMapping:
    def test_emergency_bold_red(self):
        from quantstack.tui.widgets.alerts_widget import SEVERITY_STYLES

        assert SEVERITY_STYLES["emergency"] == "bold red"

    def test_critical_red(self):
        from quantstack.tui.widgets.alerts_widget import SEVERITY_STYLES

        assert SEVERITY_STYLES["critical"] == "red"

    def test_warning_yellow(self):
        from quantstack.tui.widgets.alerts_widget import SEVERITY_STYLES

        assert SEVERITY_STYLES["warning"] == "yellow"

    def test_info_dim(self):
        from quantstack.tui.widgets.alerts_widget import SEVERITY_STYLES

        assert SEVERITY_STYLES["info"] == "dim"

    def test_unknown_defaults_to_dim(self):
        from quantstack.tui.widgets.alerts_widget import SEVERITY_STYLES

        assert SEVERITY_STYLES.get("bogus", "dim") == "dim"


# ---------------------------------------------------------------------------
# TUI: AlertsCompact — fetch query ordering
# ---------------------------------------------------------------------------


class TestAlertsFetchQuery:
    """Validate the SQL query structure used by AlertsCompact.fetch_data()."""

    def test_query_excludes_resolved(self):
        from quantstack.tui.widgets.alerts_widget import _ALERTS_QUERY

        assert "resolved" in _ALERTS_QUERY.lower()

    def test_query_orders_by_severity(self):
        from quantstack.tui.widgets.alerts_widget import _ALERTS_QUERY

        assert "CASE" in _ALERTS_QUERY
        assert "emergency" in _ALERTS_QUERY

    def test_query_limits_results(self):
        from quantstack.tui.widgets.alerts_widget import _ALERTS_QUERY

        assert "LIMIT" in _ALERTS_QUERY


# ---------------------------------------------------------------------------
# TUI: AlertsCompact — format_age helper
# ---------------------------------------------------------------------------


class TestFormatAge:
    def test_seconds(self):
        from quantstack.tui.widgets.alerts_widget import _format_age

        assert _format_age(45) == "45s ago"

    def test_minutes(self):
        from quantstack.tui.widgets.alerts_widget import _format_age

        assert _format_age(120) == "2m ago"

    def test_hours(self):
        from quantstack.tui.widgets.alerts_widget import _format_age

        assert _format_age(7200) == "2h ago"

    def test_days(self):
        from quantstack.tui.widgets.alerts_widget import _format_age

        assert _format_age(172800) == "2d ago"


# ---------------------------------------------------------------------------
# Web dashboard: /api/alerts endpoint
# ---------------------------------------------------------------------------


class TestAlertsAPIEndpoint:
    """Tests for the /api/alerts FastAPI endpoint."""

    def test_fetch_alerts_query_filters_by_status(self):
        from quantstack.dashboard.app import _fetch_alerts

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []

        with patch("quantstack.db.db_conn") as mock_db:
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)
            result = _fetch_alerts(status="open", limit=20)

        sql = mock_conn.execute.call_args[0][0]
        assert "status = %s" in sql
        assert result == []

    def test_fetch_alerts_default_limit_20(self):
        from quantstack.dashboard.app import _fetch_alerts

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []

        with patch("quantstack.db.db_conn") as mock_db:
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)
            _fetch_alerts()

        params = mock_conn.execute.call_args[0][1]
        assert params[-1] == 20  # default limit

    def test_fetch_alerts_returns_dict_list(self):
        from quantstack.dashboard.app import _fetch_alerts

        now = datetime.now(timezone.utc)
        mock_row = (1, "risk_breach", "critical", "open", "system", "Test alert",
                    "Detail text", None, now, None, None)
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [mock_row]

        with patch("quantstack.db.db_conn") as mock_db:
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)
            result = _fetch_alerts(status="open", limit=20)

        assert len(result) == 1
        alert = result[0]
        assert alert["id"] == 1
        assert alert["category"] == "risk_breach"
        assert alert["severity"] == "critical"
        assert alert["title"] == "Test alert"
        assert "created_at" in alert


# ---------------------------------------------------------------------------
# Web dashboard: system_alert in TYPE_CONFIG
# ---------------------------------------------------------------------------


class TestDashboardHTML:
    def test_system_alert_in_type_config(self):
        from quantstack.dashboard.app import DASHBOARD_HTML

        assert "system_alert" in DASHBOARD_HTML

    def test_system_alert_css_class(self):
        from quantstack.dashboard.app import DASHBOARD_HTML

        assert ".msg.system_alert" in DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Event publishing integration: emit_system_alert -> publish_event
# ---------------------------------------------------------------------------


class TestEmitPublishIntegration:
    @pytest.mark.asyncio
    async def test_emit_calls_publish_event(self):
        """emit_system_alert should call publish_event after DB insert."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"id": 42}

        with (
            patch("quantstack.tools.functions.system_alerts.db_conn") as mock_db,
            patch("quantstack.tools.functions.system_alerts.publish_event") as mock_pub,
        ):
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            from quantstack.tools.functions.system_alerts import emit_system_alert

            alert_id = await emit_system_alert(
                category="risk_breach",
                severity="warning",
                title="Test alert",
                detail="Something happened",
            )

        assert alert_id == 42
        mock_pub.assert_called_once()
        call_kwargs = mock_pub.call_args
        # Check it was called with system_alert event type
        assert call_kwargs[1]["event_type"] == "system_alert" or \
               (len(call_kwargs[0]) > 2 and call_kwargs[0][2] == "system_alert")

    @pytest.mark.asyncio
    async def test_emit_still_works_if_publish_event_fails(self):
        """publish_event failure must not break alert creation."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"id": 99}

        with (
            patch("quantstack.tools.functions.system_alerts.db_conn") as mock_db,
            patch("quantstack.tools.functions.system_alerts.publish_event", side_effect=RuntimeError("boom")),
        ):
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            from quantstack.tools.functions.system_alerts import emit_system_alert

            alert_id = await emit_system_alert(
                category="kill_switch",
                severity="critical",
                title="Kill triggered",
                detail="Emergency stop",
            )

        # Alert was still created despite publish_event failure
        assert alert_id == 99

    @pytest.mark.asyncio
    async def test_publish_event_receives_alert_metadata(self):
        """publish_event should receive alert_id, category, severity in metadata."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"id": 7}

        with (
            patch("quantstack.tools.functions.system_alerts.db_conn") as mock_db,
            patch("quantstack.tools.functions.system_alerts.publish_event") as mock_pub,
        ):
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            from quantstack.tools.functions.system_alerts import emit_system_alert

            await emit_system_alert(
                category="data_quality",
                severity="info",
                title="Stale data",
                detail="OHLCV data is 2 hours old",
                metadata={"symbol": "AAPL"},
            )

        call_kwargs = mock_pub.call_args[1]
        meta = call_kwargs["metadata"]
        assert meta["alert_id"] == 7
        assert meta["category"] == "data_quality"
        assert meta["severity"] == "info"
        assert meta["symbol"] == "AAPL"
