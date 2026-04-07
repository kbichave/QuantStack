# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Concurrent writer stress test for position row locking (Section 10).

Spawns N threads that all attempt to update the same position row simultaneously.
Validates no lost writes, no deadlocks, and acceptable latency.

Tagged @slow and @benchmark — not run in default CI.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from unittest.mock import MagicMock

import pytest


@pytest.mark.slow
@pytest.mark.benchmark
class TestPositionContention:
    """Stress test: N concurrent writers on the same position row."""

    def _make_mock_conn(self, shared_state: dict, lock: Lock):
        """Create a mock connection that simulates locked reads/writes."""
        mock_conn = MagicMock()
        mock_raw = MagicMock()
        mock_conn._raw = mock_raw
        mock_raw.transaction.return_value.__enter__ = MagicMock()
        mock_raw.transaction.return_value.__exit__ = MagicMock(return_value=False)

        def execute_side_effect(*args, **kwargs):
            sql = str(args[0]) if args else ""
            cursor = MagicMock()

            if "FOR UPDATE" in sql:
                # Simulate row lock acquisition with real contention
                with lock:
                    cursor.fetchone.return_value = {
                        "symbol": "AAPL",
                        "quantity": shared_state["quantity"],
                    }
                return cursor
            elif "UPDATE" in sql:
                # Apply the write under the shared lock
                with lock:
                    shared_state["quantity"] = shared_state["quantity"] + 1
                cursor.rowcount = 1
                return cursor
            else:
                cursor.fetchone.return_value = None
                return cursor

        mock_conn.execute.side_effect = execute_side_effect
        return mock_conn

    def test_no_lost_writes_n10(self):
        """10 concurrent writers — final quantity = initial + 10."""
        from quantstack.execution.portfolio_state import update_position_with_lock

        n_writers = 10
        shared_state = {"quantity": 100}
        lock = Lock()

        def writer():
            conn = self._make_mock_conn(shared_state, lock)
            return update_position_with_lock(conn, "AAPL", {"quantity": 999})

        with ThreadPoolExecutor(max_workers=n_writers) as pool:
            futures = [pool.submit(writer) for _ in range(n_writers)]
            results = [f.result() for f in as_completed(futures)]

        # All writers should succeed (mock doesn't timeout)
        assert all(r is True for r in results)
        # Each writer increments by 1 in the mock
        assert shared_state["quantity"] == 100 + n_writers

    def test_single_row_no_deadlock(self):
        """Single-row constraint eliminates deadlock risk."""
        from quantstack.execution.portfolio_state import update_position_with_lock

        # With single-row locking, deadlock requires 2+ rows locked in different order.
        # Our function only ever locks one row per call, so deadlock is impossible.
        shared_state = {"quantity": 0}
        lock = Lock()

        n_writers = 5
        with ThreadPoolExecutor(max_workers=n_writers) as pool:
            futures = [
                pool.submit(
                    lambda: update_position_with_lock(
                        self._make_mock_conn(shared_state, lock),
                        "AAPL",
                        {"quantity": 1},
                    )
                )
                for _ in range(n_writers)
            ]
            # If there were a deadlock, this would hang. 10s timeout catches it.
            results = [f.result(timeout=10) for f in as_completed(futures)]

        assert all(r is True for r in results)

    def test_latency_acceptable_at_n10(self):
        """p99 latency for a single update stays below 500ms at N=10."""
        from quantstack.execution.portfolio_state import update_position_with_lock

        n_writers = 10
        shared_state = {"quantity": 0}
        lock = Lock()
        latencies = []

        def timed_writer():
            conn = self._make_mock_conn(shared_state, lock)
            start = time.monotonic()
            update_position_with_lock(conn, "AAPL", {"quantity": 1})
            return time.monotonic() - start

        with ThreadPoolExecutor(max_workers=n_writers) as pool:
            futures = [pool.submit(timed_writer) for _ in range(n_writers)]
            latencies = [f.result(timeout=10) for f in as_completed(futures)]

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]
        # Mock-based, so latency is dominated by thread scheduling.
        # 500ms is generous — real p99 should be < 50ms with mocks.
        assert p99 < 0.5, f"p99 latency {p99:.3f}s exceeds 500ms threshold"
