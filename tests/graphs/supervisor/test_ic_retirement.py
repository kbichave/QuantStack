# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the IC-to-Retirement sweep.

Uses a mock PgConnection that pattern-matches on query substrings to return
appropriate results for each SQL call in run_ic_retirement_sweep().
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call

import pytest

from quantstack.autonomous.ic_retirement import (
    _IC_LOOKBACK_DAYS,
    _ICIR_RETIREMENT_THRESHOLD,
    _MIN_FORWARD_TEST_DAYS,
    run_ic_retirement_sweep,
)


def _make_conn(
    strategies: list[tuple[str, str, datetime]],
    ic_rows_by_strategy: dict[str, list[tuple[float]]],
) -> MagicMock:
    """Build a mock PgConnection.

    Args:
        strategies: list of (strategy_id, status, updated_at) tuples.
            Only forward_testing strategies are returned by the sweep query.
        ic_rows_by_strategy: mapping from strategy_id to list of (icir_21d,) tuples
            returned by the signal_ic query.

    The mock dispatches on query substrings to return the right data.
    It also records UPDATE and INSERT calls for assertion.
    """
    conn = MagicMock()
    updates: list[tuple[str, list]] = []
    inserts: list[tuple[str, list]] = []

    # Track which strategies have been retired so status queries reflect it.
    retired_set: set[str] = set()

    def mock_execute(query: str, params=None):
        q = query.strip().lower()
        result = MagicMock()

        # Query: fetch forward_testing strategies
        if "from strategies" in q and "forward_testing" in q and "select" in q:
            ft = [
                (sid, updated_at)
                for sid, status, updated_at in strategies
                if status == "forward_testing"
            ]
            result.fetchall.return_value = ft
            return result

        # Query: signal_ic lookup
        if "from signal_ic" in q or "signal_ic" in q and "select" in q:
            # Extract strategy_id from params
            strategy_id = params[0] if params else None
            rows = ic_rows_by_strategy.get(strategy_id, [])
            result.fetchall.return_value = rows
            return result

        # UPDATE: retire strategy
        if "update strategies" in q and "retired" in q:
            strategy_id = params[0] if params else None
            if strategy_id:
                retired_set.add(strategy_id)
                updates.append((query, params))
            result.fetchall.return_value = []
            result.fetchone.return_value = None
            return result

        # INSERT: loop_events
        if "insert into loop_events" in q:
            inserts.append((query, params))
            result.fetchall.return_value = []
            result.fetchone.return_value = None
            return result

        # Fallback
        result.fetchall.return_value = []
        result.fetchone.return_value = None
        return result

    conn.execute = mock_execute
    conn.commit = MagicMock()
    conn._updates = updates
    conn._inserts = inserts
    conn._retired_set = retired_set
    return conn


class TestICRetirementSweep:
    """Tests for run_ic_retirement_sweep()."""

    def test_29_days_not_retired(self):
        """Strategy in forward_testing for 29 days with all ICIR < 0.3 is NOT retired."""
        now = datetime.now(timezone.utc)
        updated_at = now - timedelta(days=29)

        conn = _make_conn(
            strategies=[("strat_29d", "forward_testing", updated_at)],
            ic_rows_by_strategy={
                "strat_29d": [(0.1,), (0.15,), (0.2,)],
            },
        )

        retired = run_ic_retirement_sweep(conn)

        assert retired == []
        assert len(conn._updates) == 0
        assert len(conn._inserts) == 0

    def test_31_days_all_below_retired(self):
        """Strategy in forward_testing for 31 days with all ICIR < 0.3 IS retired."""
        now = datetime.now(timezone.utc)
        updated_at = now - timedelta(days=31)

        conn = _make_conn(
            strategies=[("strat_31d", "forward_testing", updated_at)],
            ic_rows_by_strategy={
                "strat_31d": [(0.1,), (0.15,), (0.2,), (0.25,), (0.29,)],
            },
        )

        retired = run_ic_retirement_sweep(conn)

        assert retired == ["strat_31d"]
        assert "strat_31d" in conn._retired_set
        # Verify lifecycle event was logged
        assert len(conn._inserts) == 1
        insert_params = conn._inserts[0][1]
        assert insert_params[1] == "strategy_retired"
        assert insert_params[2] == "ic_retirement_sweep"

    def test_35_days_mixed_ic_not_retired(self):
        """Strategy in forward_testing for 35 days with mixed IC (some >= 0.3) is NOT retired."""
        now = datetime.now(timezone.utc)
        updated_at = now - timedelta(days=35)

        conn = _make_conn(
            strategies=[("strat_mixed", "forward_testing", updated_at)],
            ic_rows_by_strategy={
                "strat_mixed": [(0.1,), (0.35,), (0.2,), (0.15,), (0.28,)],
            },
        )

        retired = run_ic_retirement_sweep(conn)

        assert retired == []
        assert len(conn._updates) == 0

    def test_31_days_sparse_signals_retired(self):
        """Strategy in forward_testing for 31 days with 5 sparse rows all < 0.3 IS retired."""
        now = datetime.now(timezone.utc)
        updated_at = now - timedelta(days=31)

        conn = _make_conn(
            strategies=[("strat_sparse", "forward_testing", updated_at)],
            ic_rows_by_strategy={
                "strat_sparse": [(0.05,), (0.10,), (0.20,), (0.15,), (0.25,)],
            },
        )

        retired = run_ic_retirement_sweep(conn)

        assert retired == ["strat_sparse"]
        assert "strat_sparse" in conn._retired_set

    def test_live_strategy_not_touched(self):
        """A live strategy with bad IC is NOT touched by the sweep."""
        now = datetime.now(timezone.utc)
        updated_at = now - timedelta(days=60)

        conn = _make_conn(
            strategies=[("strat_live", "live", updated_at)],
            ic_rows_by_strategy={
                "strat_live": [(0.05,), (0.10,)],
            },
        )

        retired = run_ic_retirement_sweep(conn)

        assert retired == []
        assert len(conn._updates) == 0

    def test_no_signal_ic_rows_not_retired(self):
        """Strategy with no signal_ic rows is NOT retired (no data to judge)."""
        now = datetime.now(timezone.utc)
        updated_at = now - timedelta(days=45)

        conn = _make_conn(
            strategies=[("strat_no_ic", "forward_testing", updated_at)],
            ic_rows_by_strategy={
                "strat_no_ic": [],
            },
        )

        retired = run_ic_retirement_sweep(conn)

        assert retired == []
        assert len(conn._updates) == 0

    def test_no_forward_testing_strategies(self):
        """When there are no forward_testing strategies, sweep returns empty and no commit."""
        conn = _make_conn(strategies=[], ic_rows_by_strategy={})

        retired = run_ic_retirement_sweep(conn)

        assert retired == []
        # No changes to persist, so commit is not called.
        conn.commit.assert_not_called()

    def test_multiple_strategies_mixed_outcomes(self):
        """Multiple strategies: some retired, some kept."""
        now = datetime.now(timezone.utc)

        conn = _make_conn(
            strategies=[
                ("should_retire", "forward_testing", now - timedelta(days=40)),
                ("should_keep", "forward_testing", now - timedelta(days=40)),
                ("too_new", "forward_testing", now - timedelta(days=10)),
            ],
            ic_rows_by_strategy={
                "should_retire": [(0.1,), (0.2,), (0.15,)],
                "should_keep": [(0.1,), (0.5,), (0.15,)],
                "too_new": [(0.05,)],
            },
        )

        retired = run_ic_retirement_sweep(conn)

        assert retired == ["should_retire"]
        assert "should_retire" in conn._retired_set
        assert "should_keep" not in conn._retired_set
        assert "too_new" not in conn._retired_set

    def test_commit_called(self):
        """Sweep calls conn.commit() to persist changes."""
        now = datetime.now(timezone.utc)

        conn = _make_conn(
            strategies=[("strat_x", "forward_testing", now - timedelta(days=31))],
            ic_rows_by_strategy={"strat_x": [(0.1,)]},
        )

        run_ic_retirement_sweep(conn)
        conn.commit.assert_called()
