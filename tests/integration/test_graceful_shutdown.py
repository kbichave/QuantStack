"""Graceful shutdown tests: verify GracefulShutdown handler behavior."""

import signal
from unittest.mock import MagicMock

from quantstack.health.shutdown import GracefulShutdown


class TestGracefulShutdown:
    def test_should_stop_is_false_initially(self):
        shutdown = GracefulShutdown()
        assert shutdown.should_stop is False

    def test_should_stop_set_on_signal(self):
        shutdown = GracefulShutdown()
        # Simulate SIGTERM without actually registering signal handlers
        shutdown._handle_signal(signal.SIGTERM, None)
        assert shutdown.should_stop is True

    def test_cleanup_callbacks_run_on_signal(self):
        shutdown = GracefulShutdown()
        callback = MagicMock()
        shutdown.register_cleanup(callback)
        shutdown._handle_signal(signal.SIGTERM, None)
        callback.assert_called_once()

    def test_multiple_cleanup_callbacks(self):
        shutdown = GracefulShutdown()
        cb1 = MagicMock()
        cb2 = MagicMock()
        shutdown.register_cleanup(cb1)
        shutdown.register_cleanup(cb2)
        shutdown._handle_signal(signal.SIGINT, None)
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_cleanup_callback_exception_does_not_crash(self):
        shutdown = GracefulShutdown()
        bad_cb = MagicMock(side_effect=RuntimeError("cleanup failed"))
        good_cb = MagicMock()
        shutdown.register_cleanup(bad_cb)
        shutdown.register_cleanup(good_cb)
        # Should not raise
        shutdown._handle_signal(signal.SIGTERM, None)
        assert shutdown.should_stop is True
        good_cb.assert_called_once()
