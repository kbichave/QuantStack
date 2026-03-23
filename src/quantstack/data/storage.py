"""
DuckDB-based data storage for multi-timeframe market data.

Provides efficient storage and retrieval with partitioning by symbol and timeframe.

## Connection model

DuckDB allows only ONE write connection per file across OS processes.  The MCP
server is the canonical write owner.  If a live process already holds the write
lock, ``DataStore(read_only=True)`` opens a read-only connection so multiple
consumers can coexist without conflict.

Lock conflict handling in ``_connect_with_lock_guard``:
- Stale lock (owning process is dead): retries for up to ``_STALE_LOCK_RETRY_SECS``
  until the OS releases the file lock.
- Live conflict (owning process is alive): raises ``RuntimeError`` immediately
  with the PID and an exact ``kill <PID>`` command for the operator.
"""

import contextlib
import os
import threading

import duckdb
from loguru import logger

from quantstack.config.settings import get_settings
from quantstack.data._fundamentals_schema import FundamentalsSchemaMixin
from quantstack.data._ohlcv import OHLCVMixin
from quantstack.data._options_news import OptionsNewsMixin
from quantstack.data._schema import SchemaMixin
from quantstack.shared.duckdb_lock import (
    connect_with_lock_guard as _connect_with_lock_guard,
)


class DataStore(SchemaMixin, OHLCVMixin, OptionsNewsMixin, FundamentalsSchemaMixin):
    """
    DuckDB-based storage for OHLCV market data.

    Features:
    - Efficient columnar storage
    - Partitioning by symbol and timeframe
    - Fast analytical queries
    - Support for multi-timeframe data

    ## Connection model

    By default, connections are **short-lived**: each operation opens a connection,
    executes, and closes it.  This releases the DuckDB file lock between calls,
    allowing other processes (scripts, backtests, read-only consumers) to access
    the same database concurrently.

    Pass ``persistent=True`` to hold a single connection for the lifetime of the
    DataStore instance (legacy behavior, needed only for bulk-write workloads
    where per-operation overhead matters).

    Pass ``read_only=True`` to open read-only connections (no lock competition).
    """

    def __init__(
        self,
        db_path: str | None = None,
        read_only: bool = False,
        persistent: bool = False,
    ):
        """
        Initialize the data store.

        Args:
            db_path: Path to database file (uses settings if not provided).
            read_only: Open in read-only mode (no lock competition with write owner).
            persistent: Hold a single connection for the instance lifetime.
                        Default False: connections are opened/closed per operation.
        """
        settings = get_settings()
        self.db_path = db_path or settings.database_path
        self.read_only = read_only
        self._persistent = persistent

        # Ensure directory exists (not needed for read-only, but harmless)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        self._conn: duckdb.DuckDBPyConnection | None = None
        self._conn_lock = threading.Lock()
        if not read_only:
            self._init_schema()

    def _open_connection(self) -> duckdb.DuckDBPyConnection:
        """Open a fresh DuckDB connection."""
        return _connect_with_lock_guard(self.db_path, read_only=self.read_only)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection.

        In persistent mode, caches the connection for the instance lifetime.
        In short-lived mode (default), returns the cached connection created
        by the current ``_use_conn()`` context or falls back to opening one
        (backward compat for callers that access ``.conn`` directly).
        """
        if self._conn is None:
            with self._conn_lock:
                if self._conn is None:
                    self._conn = self._open_connection()
        return self._conn

    def _use_conn(self):
        """Context manager that provides a connection and releases it after use.

        In persistent mode, just yields the cached connection.
        In short-lived mode, opens a fresh connection, yields it, and closes it.
        This releases the DuckDB file lock between operations.
        """
        @contextlib.contextmanager
        def _short_lived():
            conn = self._open_connection()
            try:
                yield conn
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        @contextlib.contextmanager
        def _persistent():
            yield self.conn

        if self._persistent:
            return _persistent()
        return _short_lived()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
