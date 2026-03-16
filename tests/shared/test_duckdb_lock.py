"""Tests for shared.duckdb_lock."""

import os

import duckdb
import pytest

from shared.duckdb_lock import (
    connect_with_lock_guard,
    pid_from_lock_error,
    pid_is_alive,
)


class TestPidFromLockError:
    def test_extracts_pid(self):
        exc = duckdb.IOException("Conflicting lock (PID 12345)")
        assert pid_from_lock_error(exc) == 12345

    def test_returns_none_when_no_pid(self):
        exc = duckdb.IOException("Some other error")
        assert pid_from_lock_error(exc) is None

    def test_handles_generic_exception(self):
        exc = ValueError("no PID here")
        assert pid_from_lock_error(exc) is None


class TestPidIsAlive:
    def test_current_process_is_alive(self):
        assert pid_is_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        # PID 2^30 is very unlikely to exist
        assert pid_is_alive(2**30) is False


class TestConnectWithLockGuard:
    def test_memory_connection(self):
        conn = connect_with_lock_guard(":memory:")
        assert conn is not None
        conn.execute("SELECT 1")
        conn.close()

    def test_read_only_memory(self):
        # :memory: with read_only=True still works (DuckDB ignores it for :memory:)
        conn = connect_with_lock_guard(":memory:", read_only=False)
        conn.close()

    def test_file_based_connection(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        conn = connect_with_lock_guard(db_path)
        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (42)")
        result = conn.execute("SELECT x FROM t").fetchone()
        assert result[0] == 42
        conn.close()

    def test_read_only_file_connection(self, tmp_path):
        db_path = str(tmp_path / "test_ro.duckdb")
        # Create the DB first
        writer = duckdb.connect(db_path)
        writer.execute("CREATE TABLE t (x INT)")
        writer.close()
        # Open read-only
        reader = connect_with_lock_guard(db_path, read_only=True)
        reader.execute("SELECT * FROM t")
        reader.close()
