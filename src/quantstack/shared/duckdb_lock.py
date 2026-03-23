# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
DuckDB connection lock guard — shared between quantcore and quant_pod.

Handles two lock-conflict scenarios:
  1. **Stale lock** (owning process is dead): retries for up to
     ``STALE_LOCK_RETRY_SECS`` until the OS releases the file lock.
  2. **Live conflict** (owning process is alive): raises ``RuntimeError``
     immediately with the PID and an actionable ``kill`` command.
"""

from __future__ import annotations

import gc
import os
import re
import time
from pathlib import Path

import duckdb
from loguru import logger

STALE_LOCK_RETRY_SECS: float = 10
STALE_LOCK_POLL_INTERVAL: float = 0.5

_LOCK_PID_RE = re.compile(r"\(PID\s+(\d+)\)")


def pid_from_lock_error(exc: Exception) -> int | None:
    """Extract the PID from a DuckDB lock IOException message, or ``None``."""
    m = _LOCK_PID_RE.search(str(exc))
    return int(m.group(1)) if m else None


def pid_is_alive(pid: int) -> bool:
    """Return True if a process with *pid* is running on this machine."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but owned by another user — still alive.
        return True


def connect_with_lock_guard(
    db_path: str,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, handling stale-lock vs live-conflict gracefully.

    Args:
        db_path: File path or ``":memory:"``.
        read_only: If True, opens in read-only mode (no lock competition).

    Returns:
        An open DuckDB connection.

    Raises:
        RuntimeError: If the lock is held by a live process, or if a stale lock
                      does not clear within ``STALE_LOCK_RETRY_SECS``.
    """
    if db_path == ":memory:":
        return duckdb.connect(db_path)

    if read_only:
        return duckdb.connect(db_path, read_only=True)

    deadline = time.monotonic() + STALE_LOCK_RETRY_SECS
    last_exc: Exception | None = None

    while True:
        try:
            return duckdb.connect(db_path)
        except duckdb.IOException as exc:
            last_exc = exc
            msg = str(exc)
            if "Conflicting lock" not in msg and "lock" not in msg.lower():
                raise  # unrelated I/O error

            pid = pid_from_lock_error(exc)
            if not pid:
                raise  # can't determine owner

            if pid_is_alive(pid):
                if pid == os.getpid():
                    # Self-lock: previous connection in this process wasn't
                    # fully released.  Force-close via GC and retry once.
                    gc.collect()
                    time.sleep(0.2)
                    try:
                        return duckdb.connect(db_path)
                    except duckdb.IOException:
                        pass  # fall through to the RuntimeError below
                raise RuntimeError(
                    f"DuckDB write lock held by live process PID {pid}.\n"
                    f"  → Kill it:   kill {pid}\n"
                    f"  → Or check:  ps -p {pid}\n"
                    f"  DB path: {db_path}"
                ) from exc

            # Stale lock — owning process is dead.  Clean up WAL if present.
            wal_path = Path(f"{db_path}.wal")
            if wal_path.exists():
                logger.warning(
                    f"[duckdb_lock] Removing stale WAL {wal_path} " f"(dead PID {pid})"
                )
                try:
                    wal_path.unlink()
                except OSError as wal_err:
                    logger.warning(f"[duckdb_lock] Could not remove WAL: {wal_err}")

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Stale lock on {db_path} (dead PID {pid}) did not clear "
                    f"after {STALE_LOCK_RETRY_SECS}s. "
                    f"Try: rm -f '{db_path}.wal' and restart."
                ) from last_exc

            logger.warning(
                f"[duckdb_lock] Stale lock (dead PID {pid}), "
                f"retrying in {STALE_LOCK_POLL_INTERVAL}s…"
            )
            time.sleep(STALE_LOCK_POLL_INTERVAL)
