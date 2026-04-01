# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for quantstack.monitoring.alpha_monitor — Sprint 3.

Tests the classification logic of AlphaMonitor._classify_agent()
without hitting Discord webhooks or the database on disk.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from quantstack.monitoring.alpha_monitor import (
    AlertSeverity,
    AlphaMonitor,
    DegradationAlert,
    DegradationReport,
)


@pytest.fixture
def monitor() -> AlphaMonitor:
    # Disable Discord notifications in tests
    return AlphaMonitor(webhook_url=None, min_observations=5)


def _make_entry(
    agent_id: str = "test_agent",
    rolling_ic_30: float = 0.05,
    ic_trend: str = "STABLE",
    icir: float = 0.5,
    n_ic_observations: int = 20,
    needs_retraining: bool = False,
) -> dict:
    return {
        "agent_id": agent_id,
        "rolling_ic_30": rolling_ic_30,
        "ic_trend": ic_trend,
        "icir": icir,
        "n_ic_observations": n_ic_observations,
        "needs_retraining": needs_retraining,
        "ic": rolling_ic_30,
    }


# ---------------------------------------------------------------------------
# _classify_agent
# ---------------------------------------------------------------------------


class TestClassifyAgent:
    def test_healthy_agent_returns_none(self, monitor):
        """Positive IC, stable — should return None (no alert)."""
        entry = _make_entry(rolling_ic_30=0.05, ic_trend="STABLE")

        # Need a dummy tracker that has needs_retraining() method
        class _FakeTracker:
            def needs_retraining(self, agent_id):
                return False

        result = monitor._classify_agent(entry, _FakeTracker())
        assert result is None

    def test_negative_ic_triggers_critical(self, monitor):
        """Rolling IC < 0 → CRITICAL alert."""
        entry = _make_entry(rolling_ic_30=-0.02, ic_trend="DECAYING")

        class _FakeTracker:
            def needs_retraining(self, agent_id):
                return True

        result = monitor._classify_agent(entry, _FakeTracker())
        assert result is not None
        assert result.severity == AlertSeverity.CRITICAL

    def test_decaying_ic_near_zero_triggers_warning(self, monitor):
        """IC < 0.01 AND trend DECAYING → WARNING."""
        entry = _make_entry(rolling_ic_30=0.005, ic_trend="DECAYING")

        class _FakeTracker:
            def needs_retraining(self, agent_id):
                return False

        result = monitor._classify_agent(entry, _FakeTracker())
        assert result is not None
        assert result.severity == AlertSeverity.WARNING

    def test_insufficient_observations_returns_none(self, monitor):
        """Fewer than min_observations → no alert (avoid false positives)."""
        entry = _make_entry(rolling_ic_30=-0.05, n_ic_observations=3)

        class _FakeTracker:
            def needs_retraining(self, agent_id):
                return True

        result = monitor._classify_agent(entry, _FakeTracker())
        assert result is None

    def test_improving_ic_returns_none(self, monitor):
        entry = _make_entry(rolling_ic_30=0.08, ic_trend="IMPROVING")

        class _FakeTracker:
            def needs_retraining(self, agent_id):
                return False

        result = monitor._classify_agent(entry, _FakeTracker())
        assert result is None


# ---------------------------------------------------------------------------
# DegradationReport properties
# ---------------------------------------------------------------------------


class TestDegradationReportProperties:
    def test_has_critical_true(self):
        alert = DegradationAlert(
            agent_id="a",
            severity=AlertSeverity.CRITICAL,
            message="test",
            rolling_ic_30=-0.05,
            icir=0.1,
            ic_trend="DECAYING",
            needs_retraining=True,
        )
        report = DegradationReport(
            checked_at=datetime.now(),
            n_agents_checked=1,
            alerts=[alert],
            all_agents_ic_summary=[],
        )
        assert report.has_critical
        assert report.overall_status == "critical"

    def test_has_warning_true(self):
        alert = DegradationAlert(
            agent_id="b",
            severity=AlertSeverity.WARNING,
            message="warning",
            rolling_ic_30=0.005,
            icir=0.2,
            ic_trend="DECAYING",
            needs_retraining=False,
        )
        report = DegradationReport(
            checked_at=datetime.now(),
            n_agents_checked=1,
            alerts=[alert],
            all_agents_ic_summary=[],
        )
        assert report.has_warning
        assert not report.has_critical
        assert report.overall_status == "warning"

    def test_no_alerts_clean(self):
        report = DegradationReport(
            checked_at=datetime.now(),
            n_agents_checked=3,
            alerts=[],
            all_agents_ic_summary=[],
        )
        assert report.overall_status == "clean"
