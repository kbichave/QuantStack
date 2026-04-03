"""Tests for Section 07: Self-Healing System."""

import signal
import tempfile
import time
import threading

import pytest


class TestHeartbeat:
    """Tests for heartbeat write and health check."""

    def test_write_heartbeat_creates_file(self, tmp_path, monkeypatch):
        from quantstack.health.heartbeat import write_heartbeat, HEARTBEAT_DIR
        monkeypatch.setattr("quantstack.health.heartbeat.HEARTBEAT_DIR", tmp_path)
        write_heartbeat("trading")
        hb_file = tmp_path / "trading-heartbeat"
        assert hb_file.is_file()
        ts = float(hb_file.read_text().strip())
        assert abs(ts - time.time()) < 2

    def test_write_heartbeat_updates_existing(self, tmp_path, monkeypatch):
        from quantstack.health.heartbeat import write_heartbeat
        monkeypatch.setattr("quantstack.health.heartbeat.HEARTBEAT_DIR", tmp_path)
        write_heartbeat("trading")
        ts1 = float((tmp_path / "trading-heartbeat").read_text().strip())
        time.sleep(0.05)
        write_heartbeat("trading")
        ts2 = float((tmp_path / "trading-heartbeat").read_text().strip())
        assert ts2 > ts1

    def test_check_health_returns_true_when_fresh(self, tmp_path, monkeypatch):
        from quantstack.health.heartbeat import write_heartbeat, check_health
        monkeypatch.setattr("quantstack.health.heartbeat.HEARTBEAT_DIR", tmp_path)
        write_heartbeat("trading")
        assert check_health("trading", max_age_seconds=60) is True

    def test_check_health_returns_false_when_stale(self, tmp_path, monkeypatch):
        from quantstack.health.heartbeat import check_health
        monkeypatch.setattr("quantstack.health.heartbeat.HEARTBEAT_DIR", tmp_path)
        hb_file = tmp_path / "trading-heartbeat"
        hb_file.write_text(str(time.time() - 300))
        assert check_health("trading", max_age_seconds=60) is False

    def test_check_health_returns_false_when_missing(self, tmp_path, monkeypatch):
        from quantstack.health.heartbeat import check_health
        monkeypatch.setattr("quantstack.health.heartbeat.HEARTBEAT_DIR", tmp_path)
        assert check_health("nonexistent", max_age_seconds=60) is False

    def test_check_health_default_max_age(self, tmp_path, monkeypatch):
        from quantstack.health.heartbeat import write_heartbeat, check_health, DEFAULT_MAX_AGE
        monkeypatch.setattr("quantstack.health.heartbeat.HEARTBEAT_DIR", tmp_path)
        write_heartbeat("trading")
        assert check_health("trading") is True
        assert DEFAULT_MAX_AGE["trading"] == 120
        assert DEFAULT_MAX_AGE["research"] == 600
        assert DEFAULT_MAX_AGE["supervisor"] == 360


class TestAgentWatchdog:
    """Tests for stuck-agent watchdog timer."""

    def test_watchdog_triggers_callback_after_timeout(self):
        from quantstack.health.watchdog import AgentWatchdog
        triggered = threading.Event()
        wd = AgentWatchdog(timeout_seconds=0.1, on_timeout=triggered.set)
        wd.start_cycle()
        assert triggered.wait(timeout=1.0), "Watchdog should have triggered"

    def test_watchdog_end_cycle_cancels_timer(self):
        from quantstack.health.watchdog import AgentWatchdog
        triggered = threading.Event()
        wd = AgentWatchdog(timeout_seconds=0.2, on_timeout=triggered.set)
        wd.start_cycle()
        wd.end_cycle()
        time.sleep(0.4)
        assert not triggered.is_set(), "Watchdog should NOT have triggered"

    def test_watchdog_does_not_trigger_if_cycle_completes(self):
        from quantstack.health.watchdog import AgentWatchdog
        triggered = threading.Event()
        wd = AgentWatchdog(timeout_seconds=1.0, on_timeout=triggered.set)
        wd.start_cycle()
        time.sleep(0.05)
        wd.end_cycle()
        assert not triggered.is_set()

    def test_watchdog_resets_on_new_cycle(self):
        from quantstack.health.watchdog import AgentWatchdog
        triggered = threading.Event()
        wd = AgentWatchdog(timeout_seconds=0.5, on_timeout=triggered.set)
        wd.start_cycle()
        wd.end_cycle()
        wd.start_cycle()
        wd.end_cycle()
        time.sleep(0.7)
        assert not triggered.is_set()


class TestGracefulShutdown:
    """Tests for SIGTERM/SIGINT signal handler."""

    def test_should_stop_starts_false(self):
        from quantstack.health.shutdown import GracefulShutdown
        gs = GracefulShutdown()
        assert gs.should_stop is False

    def test_sigterm_sets_should_stop(self):
        from quantstack.health.shutdown import GracefulShutdown
        import os
        gs = GracefulShutdown()
        gs.install()
        os.kill(os.getpid(), signal.SIGTERM)
        assert gs.should_stop is True

    def test_sigint_sets_should_stop(self):
        from quantstack.health.shutdown import GracefulShutdown
        import os
        gs = GracefulShutdown()
        gs.install()
        os.kill(os.getpid(), signal.SIGINT)
        assert gs.should_stop is True

    def test_shutdown_calls_cleanup_callbacks(self):
        from quantstack.health.shutdown import GracefulShutdown
        import os
        called = []
        gs = GracefulShutdown()
        gs.register_cleanup(lambda: called.append("a"))
        gs.register_cleanup(lambda: called.append("b"))
        gs.install()
        os.kill(os.getpid(), signal.SIGTERM)
        assert called == ["a", "b"]


class TestResilientRetry:
    """Tests for exponential backoff retry wrapper."""

    def test_retries_on_connection_error(self):
        from quantstack.health.retry import resilient_call
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("down")
            return "ok"

        result = resilient_call(flaky, max_retries=5, base_delay=0.01)
        assert result == "ok"
        assert len(attempts) == 3

    def test_fails_after_max_retries(self):
        from quantstack.health.retry import resilient_call

        def always_fail():
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            resilient_call(always_fail, max_retries=2, base_delay=0.01)

    def test_no_retry_on_value_error(self):
        from quantstack.health.retry import resilient_call
        attempts = []

        def bad_input():
            attempts.append(1)
            raise ValueError("bad")

        with pytest.raises(ValueError):
            resilient_call(bad_input, max_retries=5, base_delay=0.01)
        assert len(attempts) == 1

    def test_db_reconnect_retries_on_operational_error(self):
        from quantstack.health.retry import db_reconnect_wrapper
        attempts = []

        @db_reconnect_wrapper
        def db_op():
            attempts.append(1)
            if len(attempts) < 3:
                raise Exception("OperationalError")
            return "connected"

        result = db_op()
        assert result == "connected"
        assert len(attempts) == 3

    def test_db_reconnect_succeeds_after_transient_failure(self):
        from quantstack.health.retry import db_reconnect_wrapper
        attempts = []

        @db_reconnect_wrapper
        def db_op():
            attempts.append(1)
            if len(attempts) == 1:
                raise Exception("OperationalError")
            return "ok"

        assert db_op() == "ok"
        assert len(attempts) == 2
