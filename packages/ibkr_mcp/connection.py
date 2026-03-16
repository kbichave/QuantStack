"""
IBKRConnectionManager — singleton for the shared IB Gateway socket.

Why a singleton
---------------
``ib_insync.IB()`` represents a persistent TCP socket to IB Gateway.
Opening multiple ``IB()`` instances from the same process (different client IDs)
is possible but wastes sockets and confuses Gateway routing.

``IBKRConnectionManager`` is the single owner of one ``IB()`` instance.
Both ``IBKRDataAdapter`` (data fetching) and ``IBKRBrokerClient`` (trading)
use ``IBKRConnectionManager.get_instance()`` and never create their own ``IB()``.

Thread safety
-------------
``get_instance()`` uses double-checked locking.
``connect()`` / ``disconnect()`` are guarded by ``_lock``.

Failure behaviour
-----------------
``connect()`` retries 3 times with 5-second backoff before raising
``BrokerConnectionError``.  The ibkr_mcp server catches this and
logs a warning but does NOT crash — tools return degraded responses.
"""

from __future__ import annotations

import threading
import time

from loguru import logger

try:
    import ib_insync as ib

    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False


class IBKRConnectionManager:
    """Singleton gateway lifecycle manager."""

    _instance: IBKRConnectionManager | None = None
    _class_lock = threading.Lock()

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4001,
        client_id: int = 1,
        timeout: int = 30,
    ) -> None:
        if not _IB_AVAILABLE:
            raise ImportError("ib_insync is required. Run: uv pip install -e '.[ibkr]'")
        self._host = host
        self._port = port
        self._client_id = client_id
        self._timeout = timeout
        self._ib: ib.IB | None = None
        self._lock = threading.Lock()

    # ── Singleton factory ─────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        host: str = "127.0.0.1",
        port: int = 4001,
        client_id: int = 1,
        timeout: int = 30,
    ) -> IBKRConnectionManager:
        """Return the singleton, creating it on first call."""
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        host=host,
                        port=port,
                        client_id=client_id,
                        timeout=timeout,
                    )
        return cls._instance

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self, retries: int = 3, backoff_s: float = 5.0) -> None:
        """Connect to IB Gateway.  Retries ``retries`` times with ``backoff_s`` delay.

        Raises:
            ConnectionError: If all retries fail.
        """
        from quantcore.execution.broker import BrokerConnectionError

        with self._lock:
            if self._ib and self._ib.isConnected():
                return

            self._ib = ib.IB()
            for attempt in range(1, retries + 1):
                try:
                    self._ib.connect(
                        host=self._host,
                        port=self._port,
                        clientId=self._client_id,
                        timeout=self._timeout,
                    )
                    logger.info(
                        f"[IBKR] Connected to {self._host}:{self._port} "
                        f"(clientId={self._client_id})"
                    )
                    return
                except Exception as exc:
                    logger.warning(f"[IBKR] Connection attempt {attempt}/{retries} failed: {exc}")
                    if attempt < retries:
                        time.sleep(backoff_s)

            raise BrokerConnectionError(
                f"IB Gateway not reachable at {self._host}:{self._port} after {retries} attempts"
            )

    def disconnect(self) -> None:
        with self._lock:
            if self._ib and self._ib.isConnected():
                self._ib.disconnect()
                logger.info("[IBKR] Disconnected")
            self._ib = None

    def is_connected(self) -> bool:
        with self._lock:
            return bool(self._ib and self._ib.isConnected())

    @property
    def ib(self) -> ib.IB:
        """Return the connected IB instance.  Auto-reconnects if needed."""
        with self._lock:
            if self._ib is None or not self._ib.isConnected():
                self.connect()
            return self._ib
