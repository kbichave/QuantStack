# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for DuckDB lock-guard logic.

Lock-guard implementation lives in ``shared.duckdb_lock`` and is used by
``quant_pod.db`` via an aliased import.  Tests that exercise the guard
mock ``duckdb.connect`` and ``shared.duckdb_lock.pid_is_alive`` so they
run instantly without spawning real competing processes.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from quantstack.db import (
    open_db_readonly,
    reset_connection,
    reset_connection_readonly,
)
from quantstack.shared.duckdb_lock import (
    connect_with_lock_guard,
    pid_is_alive,
)

# ---------------------------------------------------------------------------
# pid_is_alive
# ---------------------------------------------------------------------------


class TestIsProcessAlive:
    def test_current_process_alive(self):
        """The current process must always be alive."""
        import os

        assert pid_is_alive(os.getpid()) is True

    def test_dead_pid_returns_false(self):
        """After a process exits, its PID should not be alive."""
        proc = subprocess.Popen(["true"])
        proc.wait()
        assert not pid_is_alive(proc.pid)

    def test_permission_error_treated_as_alive(self):
        """PermissionError from os.kill means the process exists (owned by another user)."""
        import os

        with patch.object(os, "kill", side_effect=PermissionError):
            assert pid_is_alive(99999) is True


# ---------------------------------------------------------------------------
# connect_with_lock_guard
# ---------------------------------------------------------------------------


def _lock_exc(pid: int) -> duckdb.IOException:
    """Build a realistic DuckDB lock IOException for a given PID."""
    return duckdb.IOException(
        f"IO Error: Could not set lock on file '/tmp/test.duckdb': "
        f"Conflicting lock is held in /usr/bin/python3 (PID {pid}) by user testuser."
    )


class TestConnectWithLockGuard:
    def test_opens_clean_db(self, tmp_path):
        """No lock conflict → returns a valid DuckDB connection."""
        path = str(tmp_path / "test.duckdb")
        conn = connect_with_lock_guard(path)
        assert conn is not None
        conn.close()

    def test_stale_lock_retries_and_succeeds(self):
        """
        PID is dead (stale lock) → retries until duckdb.connect succeeds.
        Verify that connect is called more than once.
        """
        dead_pid = 99999
        good_conn = MagicMock()
        side_effects = [
            _lock_exc(dead_pid),
            _lock_exc(dead_pid),
            good_conn,  # third attempt succeeds
        ]

        with (
            patch(
                "quantstack.shared.duckdb_lock.duckdb.connect", side_effect=side_effects
            ) as mock_connect,
            patch("quantstack.shared.duckdb_lock.pid_is_alive", return_value=False),
        ):
            result = connect_with_lock_guard("/fake/path.duckdb")

        assert result is good_conn
        assert mock_connect.call_count == 3

    def test_live_conflict_raises_immediately(self):
        """
        PID is alive → raises RuntimeError immediately with the kill command.
        Must NOT retry (call_count == 1).
        """
        live_pid = 12345
        exc = _lock_exc(live_pid)

        with (
            patch("quantstack.shared.duckdb_lock.duckdb.connect", side_effect=exc) as mock_connect,
            patch("quantstack.shared.duckdb_lock.pid_is_alive", return_value=True),
        ):
            with pytest.raises(RuntimeError, match=f"kill {live_pid}"):
                connect_with_lock_guard("/fake/path.duckdb")

        assert mock_connect.call_count == 1

    def test_stale_lock_deadline_exceeded_raises(self):
        """
        PID is dead but lock never clears within the deadline → RuntimeError
        with 'Stale lock' in the message, NOT the 'kill PID' message.
        """
        dead_pid = 77777

        def always_lock(_path, **_kw):
            raise _lock_exc(dead_pid)

        with (
            patch("quantstack.shared.duckdb_lock.duckdb.connect", side_effect=always_lock),
            patch("quantstack.shared.duckdb_lock.pid_is_alive", return_value=False),
            patch(
                "quantstack.shared.duckdb_lock.time.monotonic",
                side_effect=[0.0, 999.0],  # first call sets deadline, second is past it
            ),
        ):
            with pytest.raises(RuntimeError, match="Stale lock"):
                connect_with_lock_guard("/fake/path.duckdb")

    def test_unrelated_ioexception_propagates(self):
        """Non-lock IOException (e.g. 'Disk full') must propagate unchanged."""
        exc = duckdb.IOException("IO Error: No space left on device")

        with patch("quantstack.shared.duckdb_lock.duckdb.connect", side_effect=exc):
            with pytest.raises(duckdb.IOException, match="No space left"):
                connect_with_lock_guard("/fake/path.duckdb")

    def test_no_pid_in_lock_message_propagates(self):
        """
        'Conflicting lock' message without a PID (unexpected DuckDB format)
        should propagate the original exception, not wrap it.
        """
        exc = duckdb.IOException("IO Error: Conflicting lock is held")

        with patch("quantstack.shared.duckdb_lock.duckdb.connect", side_effect=exc):
            with pytest.raises(duckdb.IOException):
                connect_with_lock_guard("/fake/path.duckdb")


# ---------------------------------------------------------------------------
# open_db_readonly
# ---------------------------------------------------------------------------


class TestOpenDbReadonly:
    def teardown_method(self):
        """Reset the read-only singleton after each test."""
        reset_connection_readonly()

    def test_returns_connection_for_missing_file(self, tmp_path):
        """open_db_readonly creates the DB if it doesn't exist (alias for open_db)."""
        path = str(tmp_path / "nonexistent.duckdb")
        conn = open_db_readonly(path)
        assert conn is not None

    def test_succeeds_when_file_exists(self, tmp_path):
        """open_db_readonly opens an existing DB and supports reads and writes."""
        path = str(tmp_path / "test.duckdb")

        # Create the DB with a write connection first
        write_conn = duckdb.connect(path)
        write_conn.execute("CREATE TABLE foo (id INTEGER)")
        write_conn.close()

        # open_db_readonly is an alias for open_db — full read/write access
        conn = open_db_readonly(path)
        assert conn is not None
        result = conn.execute("SELECT COUNT(*) FROM foo").fetchone()
        assert result is not None

    def test_memory_path_returns_writable_connection(self):
        """
        open_db_readonly(':memory:') falls back to the regular write connection
        so test code that calls this function behaves identically to production.
        """
        try:
            conn = open_db_readonly(":memory:")
            # Must support DDL (write connection)
            conn.execute("CREATE TABLE test_rw (x INTEGER)")
            conn.execute("INSERT INTO test_rw VALUES (42)")
            result = conn.execute("SELECT x FROM test_rw").fetchone()
            assert result == (42,)
        finally:
            reset_connection()
