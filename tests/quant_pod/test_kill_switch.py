# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for KillSwitch.

Tests run with a temporary sentinel file so they never touch the real
~/.quant_pod/KILL_SWITCH_ACTIVE path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from quantstack.execution.kill_switch import KillSwitch


@pytest.fixture
def sentinel_path(tmp_path: Path) -> Path:
    return tmp_path / "KILL_SWITCH_ACTIVE"


@pytest.fixture
def ks(sentinel_path: Path, monkeypatch) -> KillSwitch:
    """Fresh KillSwitch using a temp sentinel file."""
    monkeypatch.setattr(KillSwitch, "SENTINEL_FILE", sentinel_path)
    return KillSwitch()


class TestKillSwitchInitialState:
    def test_not_active_on_fresh_init(self, ks):
        assert ks.is_active() is False

    def test_status_active_false(self, ks):
        assert ks.status().active is False

    def test_guard_does_not_raise_when_inactive(self, ks):
        ks.guard()  # should not raise


class TestTrigger:
    def test_trigger_sets_active(self, ks):
        ks.trigger("test reason")
        assert ks.is_active() is True

    def test_trigger_writes_sentinel_file(self, ks, sentinel_path):
        ks.trigger("sentinel test")
        assert sentinel_path.exists()

    def test_trigger_stores_reason(self, ks):
        ks.trigger("loss limit breached")
        assert ks.status().reason == "loss limit breached"

    def test_trigger_sets_triggered_at(self, ks):
        from datetime import datetime

        ks.trigger("timing test")
        assert ks.status().triggered_at is not None
        assert isinstance(ks.status().triggered_at, datetime)

    def test_guard_raises_when_active(self, ks):
        ks.trigger("danger")
        with pytest.raises(RuntimeError, match="ACTIVE"):
            ks.guard()


class TestReset:
    def test_reset_clears_active(self, ks):
        ks.trigger("temp")
        ks.reset("ops")
        assert ks.is_active() is False

    def test_reset_removes_sentinel_file(self, ks, sentinel_path):
        ks.trigger("temp")
        assert sentinel_path.exists()
        ks.reset("ops")
        assert not sentinel_path.exists()

    def test_reset_when_file_already_gone_is_safe(self, ks, sentinel_path):
        """Atomic delete must not raise even if sentinel was already removed."""
        ks.trigger("temp")
        sentinel_path.unlink()  # Simulate race: someone else deleted it
        ks.reset("ops")  # Should not raise FileNotFoundError
        assert ks.is_active() is False

    def test_reset_records_reset_by(self, ks):
        ks.trigger("temp")
        ks.reset("on-call-engineer")
        assert ks.status().reset_by == "on-call-engineer"


class TestSentinelPersistence:
    def test_new_instance_loads_active_sentinel(self, sentinel_path, monkeypatch):
        """Simulates a process restart — new KillSwitch must see prior trigger."""
        monkeypatch.setattr(KillSwitch, "SENTINEL_FILE", sentinel_path)
        ks1 = KillSwitch()
        ks1.trigger("pre-crash trigger")

        # Second instance — same sentinel file, simulates restart
        ks2 = KillSwitch()
        assert ks2.is_active() is True

    def test_new_instance_not_active_when_no_sentinel(self, sentinel_path, monkeypatch):
        monkeypatch.setattr(KillSwitch, "SENTINEL_FILE", sentinel_path)
        ks = KillSwitch()
        assert ks.is_active() is False

    def test_sentinel_file_content_contains_reason(self, ks, sentinel_path):
        ks.trigger("data feed timeout")
        content = sentinel_path.read_text()
        assert "data feed timeout" in content


class TestPositionCloserRegistration:
    def test_position_closer_called_on_trigger(self, ks):
        called = []
        ks.register_position_closer(lambda: called.append(True))
        ks.trigger("test")
        assert called == [True]

    def test_position_closer_exception_does_not_prevent_trigger(self, ks):
        def bad_closer():
            raise RuntimeError("closer failed")

        ks.register_position_closer(bad_closer)
        ks.trigger("test")  # should not raise even though closer failed
        assert ks.is_active() is True
