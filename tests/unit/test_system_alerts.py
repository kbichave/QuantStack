"""Unit tests for system-level alert lifecycle tools and internal helper.

Tests mock the database layer so they run without PostgreSQL.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from quantstack.tools.functions.system_alerts import (
    ALLOWED_CATEGORIES,
    ALLOWED_SEVERITIES,
    SEVERITY_ORDER,
    emit_system_alert,
)
from quantstack.tools.langchain.system_alert_tools import (
    acknowledge_alert,
    create_system_alert,
    escalate_alert,
    query_system_alerts,
    resolve_alert,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SENTINEL = object()


def _make_mock_conn(rows=None, fetchone_val=_SENTINEL):
    """Create a mock db connection context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    if fetchone_val is not _SENTINEL:
        cursor.fetchone.return_value = fetchone_val
    if rows is not None:
        cursor.fetchall.return_value = rows
    conn.execute.return_value = cursor
    return conn


def _db_conn_patch(conn):
    """Patch db_conn to yield the given mock connection."""
    from contextlib import contextmanager

    @contextmanager
    def _mock_db_conn():
        yield conn

    return _mock_db_conn


# ---------------------------------------------------------------------------
# Shared constants tests
# ---------------------------------------------------------------------------


class TestSharedConstants:
    def test_allowed_categories_complete(self):
        expected = {
            "risk_breach", "service_failure", "kill_switch", "data_quality",
            "performance_degradation", "factor_drift", "ack_timeout", "thesis_review",
        }
        assert ALLOWED_CATEGORIES == expected

    def test_allowed_severities_complete(self):
        assert ALLOWED_SEVERITIES == {"info", "warning", "critical", "emergency"}

    def test_severity_order_ascending(self):
        assert SEVERITY_ORDER == ["info", "warning", "critical", "emergency"]


# ---------------------------------------------------------------------------
# emit_system_alert (internal helper)
# ---------------------------------------------------------------------------


class TestEmitSystemAlert:
    def test_validates_invalid_category(self):
        with pytest.raises(ValueError, match="Invalid alert category"):
            asyncio.get_event_loop().run_until_complete(
                emit_system_alert("bogus", "info", "title", "detail")
            )

    def test_validates_invalid_severity(self):
        with pytest.raises(ValueError, match="Invalid alert severity"):
            asyncio.get_event_loop().run_until_complete(
                emit_system_alert("risk_breach", "bogus", "title", "detail")
            )

    def test_inserts_row_and_returns_id(self):
        conn = _make_mock_conn(fetchone_val={"id": 42})
        with patch("quantstack.tools.functions.system_alerts.db_conn", _db_conn_patch(conn)):
            alert_id = asyncio.get_event_loop().run_until_complete(
                emit_system_alert("risk_breach", "critical", "Test alert", "Detail text", source="risk_gate")
            )
        assert alert_id == 42
        call_args = conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "INSERT INTO system_alerts" in sql
        assert params[0] == "risk_breach"
        assert params[1] == "critical"
        assert params[2] == "risk_gate"
        assert params[3] == "Test alert"
        assert params[4] == "Detail text"

    def test_metadata_serialized_as_json(self):
        conn = _make_mock_conn(fetchone_val={"id": 1})
        meta = {"threshold": 0.5, "positions": ["AAPL"]}
        with patch("quantstack.tools.functions.system_alerts.db_conn", _db_conn_patch(conn)):
            asyncio.get_event_loop().run_until_complete(
                emit_system_alert("data_quality", "warning", "title", "detail", metadata=meta)
            )
        params = conn.execute.call_args[0][1]
        assert json.loads(params[5]) == meta

    def test_metadata_none_passes_null(self):
        conn = _make_mock_conn(fetchone_val={"id": 1})
        with patch("quantstack.tools.functions.system_alerts.db_conn", _db_conn_patch(conn)):
            asyncio.get_event_loop().run_until_complete(
                emit_system_alert("data_quality", "info", "title", "detail")
            )
        params = conn.execute.call_args[0][1]
        assert params[5] is None


# ---------------------------------------------------------------------------
# create_system_alert (LangChain tool)
# ---------------------------------------------------------------------------


class TestCreateSystemAlert:
    def test_returns_alert_id_on_success(self):
        conn = _make_mock_conn(fetchone_val={"id": 99})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                create_system_alert.ainvoke({
                    "category": "risk_breach",
                    "severity": "critical",
                    "title": "Risk limit breached",
                    "detail": "Position size exceeded 5% of portfolio",
                })
            )
        data = json.loads(result)
        assert data["alert_id"] == 99
        assert data["status"] == "open"

    def test_rejects_invalid_category(self):
        result = asyncio.get_event_loop().run_until_complete(
            create_system_alert.ainvoke({
                "category": "invalid_cat",
                "severity": "info",
                "title": "t",
                "detail": "d",
            })
        )
        data = json.loads(result)
        assert "error" in data
        assert "invalid_cat" in data["error"]

    def test_rejects_invalid_severity(self):
        result = asyncio.get_event_loop().run_until_complete(
            create_system_alert.ainvoke({
                "category": "risk_breach",
                "severity": "invalid_sev",
                "title": "t",
                "detail": "d",
            })
        )
        data = json.loads(result)
        assert "error" in data
        assert "invalid_sev" in data["error"]


# ---------------------------------------------------------------------------
# acknowledge_alert
# ---------------------------------------------------------------------------


class TestAcknowledgeAlert:
    def test_sets_acknowledged_status(self):
        conn = _make_mock_conn(fetchone_val={"status": "open"})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                acknowledge_alert.ainvoke({"alert_id": 1, "agent_name": "self_healer"})
            )
        data = json.loads(result)
        assert data["status"] == "acknowledged"
        # Verify UPDATE was called
        update_call = conn.execute.call_args_list[1]
        assert "acknowledged" in update_call[0][0]
        assert "self_healer" in update_call[0][1]

    def test_idempotent_on_already_acknowledged(self):
        conn = _make_mock_conn(fetchone_val={"status": "acknowledged"})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                acknowledge_alert.ainvoke({"alert_id": 1, "agent_name": "self_healer"})
            )
        data = json.loads(result)
        assert data["status"] == "acknowledged"
        assert "no change" in data["message"].lower()
        # Only the SELECT was called, no UPDATE
        assert conn.execute.call_count == 1

    def test_idempotent_on_resolved(self):
        conn = _make_mock_conn(fetchone_val={"status": "resolved"})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                acknowledge_alert.ainvoke({"alert_id": 1, "agent_name": "self_healer"})
            )
        data = json.loads(result)
        assert data["status"] == "resolved"

    def test_not_found(self):
        conn = _make_mock_conn(fetchone_val=None)
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                acknowledge_alert.ainvoke({"alert_id": 999, "agent_name": "test"})
            )
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()


# ---------------------------------------------------------------------------
# escalate_alert
# ---------------------------------------------------------------------------


class TestEscalateAlert:
    def test_bumps_severity_warning_to_critical(self):
        conn = _make_mock_conn(fetchone_val={"severity": "warning", "detail": "original detail"})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                escalate_alert.ainvoke({"alert_id": 1, "reason": "Issue worsening"})
            )
        data = json.loads(result)
        assert data["old_severity"] == "warning"
        assert data["new_severity"] == "critical"
        assert data["status"] == "escalated"

    def test_bumps_info_to_warning(self):
        conn = _make_mock_conn(fetchone_val={"severity": "info", "detail": ""})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                escalate_alert.ainvoke({"alert_id": 1, "reason": "needs attention"})
            )
        data = json.loads(result)
        assert data["new_severity"] == "warning"

    def test_emergency_ceiling(self):
        conn = _make_mock_conn(fetchone_val={"severity": "emergency", "detail": "already max"})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                escalate_alert.ainvoke({"alert_id": 1, "reason": "still bad"})
            )
        data = json.loads(result)
        assert data["old_severity"] == "emergency"
        assert data["new_severity"] == "emergency"

    def test_reason_appended_to_detail(self):
        conn = _make_mock_conn(fetchone_val={"severity": "warning", "detail": "initial"})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            asyncio.get_event_loop().run_until_complete(
                escalate_alert.ainvoke({"alert_id": 1, "reason": "metrics degrading"})
            )
        update_call = conn.execute.call_args_list[1]
        updated_detail = update_call[0][1][1]
        assert "[ESCALATION] metrics degrading" in updated_detail
        assert "initial" in updated_detail

    def test_not_found(self):
        conn = _make_mock_conn(fetchone_val=None)
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                escalate_alert.ainvoke({"alert_id": 999, "reason": "test"})
            )
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# resolve_alert
# ---------------------------------------------------------------------------


class TestResolveAlert:
    def test_sets_resolved_status(self):
        conn = _make_mock_conn(fetchone_val={"status": "acknowledged"})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                resolve_alert.ainvoke({"alert_id": 1, "resolution": "Restarted the service"})
            )
        data = json.loads(result)
        assert data["status"] == "resolved"

    def test_idempotent_on_already_resolved(self):
        conn = _make_mock_conn(fetchone_val={"status": "resolved"})
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                resolve_alert.ainvoke({"alert_id": 1, "resolution": "Already fixed"})
            )
        data = json.loads(result)
        assert data["status"] == "resolved"
        assert "no change" in data["message"].lower()
        assert conn.execute.call_count == 1

    def test_not_found(self):
        conn = _make_mock_conn(fetchone_val=None)
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                resolve_alert.ainvoke({"alert_id": 999, "resolution": "test"})
            )
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# query_system_alerts
# ---------------------------------------------------------------------------


class TestQuerySystemAlerts:
    def _make_alert_row(self, **overrides):
        base = {
            "id": 1,
            "category": "risk_breach",
            "severity": "critical",
            "status": "open",
            "source": "supervisor",
            "title": "Test alert",
            "detail": "Some detail",
            "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
            "acknowledged_by": None,
            "acknowledged_at": None,
            "resolved_at": None,
        }
        base.update(overrides)
        return base

    def test_returns_formatted_alert_list(self):
        rows = [self._make_alert_row(id=1), self._make_alert_row(id=2, severity="warning")]
        conn = _make_mock_conn(rows=rows)
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                query_system_alerts.ainvoke({})
            )
        assert "Found 2 alert(s)" in result
        assert "#1" in result
        assert "#2" in result

    def test_no_results_message(self):
        conn = _make_mock_conn(rows=[])
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            result = asyncio.get_event_loop().run_until_complete(
                query_system_alerts.ainvoke({})
            )
        assert "No system alerts found" in result

    def test_severity_filter_passed_to_query(self):
        conn = _make_mock_conn(rows=[])
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            asyncio.get_event_loop().run_until_complete(
                query_system_alerts.ainvoke({"severity": "critical"})
            )
        sql = conn.execute.call_args[0][0]
        params = conn.execute.call_args[0][1]
        assert "severity = %s" in sql
        assert "critical" in params

    def test_status_filter_passed_to_query(self):
        conn = _make_mock_conn(rows=[])
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            asyncio.get_event_loop().run_until_complete(
                query_system_alerts.ainvoke({"status": "open"})
            )
        sql = conn.execute.call_args[0][0]
        params = conn.execute.call_args[0][1]
        assert "status = %s" in sql
        assert "open" in params

    def test_category_filter_passed_to_query(self):
        conn = _make_mock_conn(rows=[])
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            asyncio.get_event_loop().run_until_complete(
                query_system_alerts.ainvoke({"category": "factor_drift"})
            )
        sql = conn.execute.call_args[0][0]
        params = conn.execute.call_args[0][1]
        assert "category = %s" in sql
        assert "factor_drift" in params

    def test_since_hours_filter(self):
        conn = _make_mock_conn(rows=[])
        with patch("quantstack.tools.langchain.system_alert_tools.db_conn", _db_conn_patch(conn)):
            asyncio.get_event_loop().run_until_complete(
                query_system_alerts.ainvoke({"since_hours": 48})
            )
        params = conn.execute.call_args[0][1]
        cutoff = params[0]
        # The cutoff should be approximately 48 hours ago
        expected = datetime.now(timezone.utc) - timedelta(hours=48)
        assert abs((cutoff - expected).total_seconds()) < 5


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_all_five_tools_registered(self):
        from quantstack.tools.registry import TOOL_REGISTRY

        expected = [
            "create_system_alert",
            "acknowledge_alert",
            "escalate_alert",
            "resolve_alert",
            "query_system_alerts",
        ]
        for name in expected:
            assert name in TOOL_REGISTRY, f"{name} not in TOOL_REGISTRY"
