"""Graceful shutdown handler for SIGTERM/SIGINT."""

import logging
import signal
import threading
from typing import Callable

logger = logging.getLogger(__name__)

_CLEANUP_TIMEOUT = 10  # seconds per callback


class GracefulShutdown:
    """Signal handler for graceful process termination.

    Usage:
        shutdown = GracefulShutdown()
        shutdown.register_cleanup(langfuse_flush)
        shutdown.install()

        while not shutdown.should_stop:
            run_one_cycle()
    """

    def __init__(self) -> None:
        self._should_stop = False
        self._callbacks: list[Callable[[], None]] = []

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup function to run on shutdown signal."""
        self._callbacks.append(callback)

    def install(self) -> None:
        """Register signal handlers for SIGTERM and SIGINT."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def install_async(self, loop) -> None:
        """Register signal handlers using asyncio's event loop (preferred in async context)."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_async_signal, sig)

    def _handle_async_signal(self, signum: int) -> None:
        """Handle signal in async context (no frame argument)."""
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating graceful shutdown", sig_name)
        self._should_stop = True
        self._run_cleanup()

    def _handle_signal(self, signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating graceful shutdown", sig_name)
        self._should_stop = True
        self._run_cleanup()

    def _run_cleanup(self) -> None:
        for callback in self._callbacks:
            try:
                t = threading.Thread(target=callback, daemon=True)
                t.start()
                t.join(timeout=_CLEANUP_TIMEOUT)
                if t.is_alive():
                    logger.warning(
                        "Cleanup callback %s timed out after %ds",
                        callback.__name__, _CLEANUP_TIMEOUT,
                    )
            except Exception:
                logger.exception("Cleanup callback %s failed", callback.__name__)
