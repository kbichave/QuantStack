"""Unit tests for kill switch auto-recovery, escalation, and sizing ramp-back."""

import json
import logging
import sys
from datetime import datetime, timedelta
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# Stub out psycopg and its submodules before any quantstack imports trigger
# the broker_factory -> paper_broker -> db import chain.
_PSYCOPG_MODULES = [
    "psycopg", "psycopg.rows", "psycopg.types", "psycopg.types.json",
    "psycopg_pool",
]
for _mod_name in _PSYCOPG_MODULES:
    if _mod_name not in sys.modules:
        _stub = ModuleType(_mod_name)
        # Provide common attributes that db.py expects
        if _mod_name == "psycopg":
            _stub.Connection = MagicMock
        elif _mod_name == "psycopg.types.json":
            _stub.set_json_loads = MagicMock()
        elif _mod_name == "psycopg_pool":
            _stub.ConnectionPool = MagicMock
        sys.modules[_mod_name] = _stub

from quantstack.execution.kill_switch import KillSwitch, KillSwitchStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kill_switch(tmp_path):
    """Create a fresh KillSwitch with a temp sentinel file."""
    ks = KillSwitch()
    ks.SENTINEL_FILE = tmp_path / "KILL_SWITCH_ACTIVE"
    ks._status = KillSwitchStatus()
    return ks


@pytest.fixture
def mock_db():
    """Patch db_conn and return a mock connection."""
    mock_conn = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    with patch(
        "quantstack.execution.kill_switch_recovery.db_conn",
        return_value=mock_ctx,
    ) as p:
        yield mock_conn


# ---------------------------------------------------------------------------
# AutoRecoveryManager tests
# ---------------------------------------------------------------------------

class TestAutoRecoveryManager:
    """Tests for the AutoRecoveryManager class."""

    def test_does_nothing_when_kill_switch_inactive(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        mgr = AutoRecoveryManager(kill_switch)
        mgr.check()
        # reset should never be called
        assert not kill_switch.is_active()

    def test_does_nothing_for_drawdown_trigger(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        kill_switch.trigger(reason="Auto-trigger: 3-day rolling drawdown 8.00% >= threshold 6.00%")
        mgr = AutoRecoveryManager(kill_switch)
        # Advance past 15 min window
        kill_switch._status.triggered_at = datetime.now() - timedelta(minutes=20)
        mgr.check()
        assert kill_switch.is_active()  # Should NOT have reset

    def test_does_nothing_for_drift_trigger(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        kill_switch.trigger(reason="Auto-trigger: model drift detected on 5/8 active strategies")
        mgr = AutoRecoveryManager(kill_switch)
        kill_switch._status.triggered_at = datetime.now() - timedelta(minutes=20)
        mgr.check()
        assert kill_switch.is_active()

    def test_does_nothing_for_spy_halt_trigger(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        kill_switch.trigger(reason="Auto-trigger: market-wide circuit breaker detected (SPY halted)")
        mgr = AutoRecoveryManager(kill_switch)
        kill_switch._status.triggered_at = datetime.now() - timedelta(minutes=20)
        mgr.check()
        assert kill_switch.is_active()

    def test_investigates_at_5_min(self, kill_switch, mock_db, caplog):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        kill_switch.trigger(reason="Auto-trigger: 3 consecutive broker API failures (last: timeout)")
        kill_switch._status.triggered_at = datetime.now() - timedelta(minutes=7)
        mgr = AutoRecoveryManager(kill_switch)

        with caplog.at_level(logging.INFO):
            mgr.check()

        assert any("investigat" in r.message.lower() for r in caplog.records)
        assert kill_switch.is_active()  # Not reset yet (< 15 min)

    def test_auto_resets_at_15_min_when_broker_responsive(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        kill_switch.trigger(reason="Auto-trigger: 3 consecutive broker API failures (last: timeout)")
        kill_switch._status.triggered_at = datetime.now() - timedelta(minutes=16)

        # Mock daily reset count = 0
        mock_db.execute.return_value.fetchone.return_value = None

        mgr = AutoRecoveryManager(kill_switch)
        mgr._check_broker_health = MagicMock(return_value=True)
        mgr.check()

        assert not kill_switch.is_active()

    def test_no_reset_when_broker_still_down(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        kill_switch.trigger(reason="Auto-trigger: 3 consecutive broker API failures (last: timeout)")
        kill_switch._status.triggered_at = datetime.now() - timedelta(minutes=16)

        mgr = AutoRecoveryManager(kill_switch)
        mgr._check_broker_health = MagicMock(return_value=False)
        mgr.check()

        assert kill_switch.is_active()  # Not reset

    def test_sets_sizing_scalar_on_auto_reset(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        kill_switch.trigger(reason="Auto-trigger: 3 consecutive broker API failures (last: timeout)")
        kill_switch._status.triggered_at = datetime.now() - timedelta(minutes=16)
        mock_db.execute.return_value.fetchone.return_value = None

        mgr = AutoRecoveryManager(kill_switch)
        mgr._check_broker_health = MagicMock(return_value=True)
        mgr.check()

        # Verify system_state write for sizing scalar
        calls = [str(c) for c in mock_db.execute.call_args_list]
        assert any("kill_switch_sizing_scalar" in c for c in calls)

    def test_respects_max_auto_resets_per_day(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        kill_switch.trigger(reason="Auto-trigger: 3 consecutive broker API failures (last: timeout)")
        kill_switch._status.triggered_at = datetime.now() - timedelta(minutes=16)

        # Mock: already 2 resets today
        mock_db.execute.return_value.fetchone.return_value = ("2",)

        mgr = AutoRecoveryManager(kill_switch)
        mgr._check_broker_health = MagicMock(return_value=True)
        mgr.check()

        assert kill_switch.is_active()  # Should NOT reset (at cap)

    def test_backs_off_on_immediate_retrigger(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import AutoRecoveryManager

        kill_switch.trigger(reason="Auto-trigger: 3 consecutive broker API failures (last: timeout)")
        kill_switch._status.triggered_at = datetime.now() - timedelta(minutes=16)
        mock_db.execute.return_value.fetchone.return_value = None

        mgr = AutoRecoveryManager(kill_switch)
        mgr._check_broker_health = MagicMock(return_value=True)
        mgr._last_reset_at = datetime.now() - timedelta(minutes=3)  # Reset 3 min ago
        mgr.check()

        assert kill_switch.is_active()  # Should NOT reset (too soon after last)


# ---------------------------------------------------------------------------
# Sizing ramp-back tests
# ---------------------------------------------------------------------------

class TestSizingRampBack:
    """Tests for the sizing ramp-back mechanism."""

    def test_sizing_scalar_starts_at_half(self, mock_db):
        from quantstack.execution.kill_switch_recovery import SizingRampBack

        ramp = SizingRampBack()
        ramp.initialize_after_reset()

        calls = [str(c) for c in mock_db.execute.call_args_list]
        assert any("0.5" in c and "kill_switch_sizing_scalar" in c for c in calls)

    def test_ramp_sequence(self, mock_db):
        from quantstack.execution.kill_switch_recovery import SizingRampBack

        ramp = SizingRampBack()

        # Simulate 3 successful cycles
        mock_db.execute.return_value.fetchone.side_effect = [
            ("0",),  # successful_cycles = 0 -> set to 1, scalar 0.75
        ]
        ramp.record_successful_cycle()
        calls = [str(c) for c in mock_db.execute.call_args_list]
        assert any("0.75" in c for c in calls)

    def test_ramp_resets_on_retrigger(self, mock_db):
        from quantstack.execution.kill_switch_recovery import SizingRampBack

        ramp = SizingRampBack()
        ramp.reset_on_retrigger()

        calls = [str(c) for c in mock_db.execute.call_args_list]
        assert any("0.5" in c and "kill_switch_sizing_scalar" in c for c in calls)


# ---------------------------------------------------------------------------
# KillSwitchEscalationManager tests
# ---------------------------------------------------------------------------

class TestKillSwitchEscalationManager:
    """Tests for the escalation notification manager."""

    def test_sends_discord_at_tier_1(self, kill_switch, mock_db, caplog):
        from quantstack.execution.kill_switch_recovery import KillSwitchEscalationManager

        kill_switch.trigger(reason="Test trigger")
        mgr = KillSwitchEscalationManager(kill_switch)

        # Mock: no prior escalation tier
        mock_db.execute.return_value.fetchone.return_value = None

        with patch.object(mgr, "_send_discord") as mock_discord:
            with caplog.at_level(logging.INFO):
                mgr.check()
            mock_discord.assert_called_once()

    def test_escalates_at_4_hours(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import KillSwitchEscalationManager

        kill_switch.trigger(reason="Test trigger")
        kill_switch._status.triggered_at = datetime.now() - timedelta(hours=5)
        mgr = KillSwitchEscalationManager(kill_switch)

        # Mock: already at tier 1
        mock_db.execute.return_value.fetchone.return_value = ("1",)

        with patch.object(mgr, "_send_discord") as mock_discord:
            mgr.check()
            mock_discord.assert_called_once()
            # Should be tier 2 (enhanced) message
            call_args = mock_discord.call_args
            assert call_args is not None

    def test_no_duplicate_notifications(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import KillSwitchEscalationManager

        kill_switch.trigger(reason="Test trigger")
        mgr = KillSwitchEscalationManager(kill_switch)

        # Mock: already at tier 1, not enough time for tier 2
        mock_db.execute.return_value.fetchone.return_value = ("1",)
        kill_switch._status.triggered_at = datetime.now() - timedelta(hours=1)

        with patch.object(mgr, "_send_discord") as mock_discord:
            mgr.check()
            mock_discord.assert_not_called()

    def test_resets_tier_when_cleared(self, kill_switch, mock_db):
        from quantstack.execution.kill_switch_recovery import KillSwitchEscalationManager

        # Kill switch is NOT active
        mgr = KillSwitchEscalationManager(kill_switch)
        mgr.check()

        calls = [str(c) for c in mock_db.execute.call_args_list]
        assert any("DELETE" in c and "last_escalation_tier" in c for c in calls)


# ---------------------------------------------------------------------------
# reset() signature tests
# ---------------------------------------------------------------------------

class TestResetSignature:
    """Tests for the enhanced reset() method."""

    def test_reset_with_reason_logs_audit(self, kill_switch):
        from loguru import logger as _loguru
        messages = []
        sink_id = _loguru.add(lambda m: messages.append(m.record["message"]))
        try:
            kill_switch.trigger(reason="Test")
            kill_switch.reset(reset_by="auto_recovery", reason="Broker responsive after 15min")
        finally:
            _loguru.remove(sink_id)
        assert not kill_switch.is_active()
        assert any("Broker responsive" in m for m in messages)

    def test_reset_without_reason_warns(self, kill_switch):
        from loguru import logger as _loguru
        messages = []
        sink_id = _loguru.add(lambda m: messages.append(m.record["message"]))
        try:
            kill_switch.trigger(reason="Test")
            kill_switch.reset(reset_by="manual")
        finally:
            _loguru.remove(sink_id)
        assert not kill_switch.is_active()
        assert any("without reason" in m.lower() for m in messages)
