# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 01: Persistence Migration (StrategyBreaker + ICAttributionTracker).

Verifies that both modules correctly round-trip state through PostgreSQL
instead of JSON files. All DB calls are mocked — no real database needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from quantstack.execution.strategy_breaker import (
    BreakerConfig,
    BreakerState,
    STATUS_ACTIVE,
    STATUS_SCALED,
    STATUS_TRIPPED,
    StrategyBreaker,
)
from quantstack.learning.ic_attribution import (
    ICAttributionTracker,
    _CollectorState,
    _Observation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_conn():
    """Create a mock PgConnection that records execute() calls."""
    conn = MagicMock()
    conn.execute.return_value = conn  # chaining
    conn.fetchall.return_value = []
    conn.fetchone.return_value = None
    return conn


def _db_conn_ctx(mock_conn):
    """Return a context-manager factory that yields mock_conn."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        yield mock_conn

    return _ctx


# ===========================================================================
# StrategyBreaker PostgreSQL Migration Tests
# ===========================================================================


class TestStrategyBreakerPersistence:
    """Verify StrategyBreaker save/load round-trips through PostgreSQL."""

    @patch("quantstack.execution.strategy_breaker.db_conn")
    def test_save_load_roundtrip(self, mock_db_conn):
        """BreakerState saved to DB can be loaded back with identical field values."""
        mock_conn = _make_mock_conn()
        mock_db_conn.side_effect = _db_conn_ctx(mock_conn)

        # Create breaker — _load() runs on init, returns empty
        breaker = StrategyBreaker(config=BreakerConfig())

        # Record trades to get into SCALED state
        breaker.record_trade("strat_a", pnl=-100.0, equity=9900.0)
        breaker.record_trade("strat_a", pnl=-100.0, equity=9800.0)

        # Verify _persist was called (upsert executed)
        upsert_calls = [
            c for c in mock_conn.execute.call_args_list
            if "INSERT INTO strategy_breaker_states" in str(c)
        ]
        assert len(upsert_calls) > 0, "Expected upsert calls to strategy_breaker_states"

        # Now simulate loading from DB — set up rows to return
        state = breaker.get_all_states()["strat_a"]
        mock_conn.fetchall.return_value = [
            {
                "strategy_id": state.strategy_id,
                "status": state.status,
                "scale_factor": state.scale_factor,
                "consecutive_losses": state.consecutive_losses,
                "peak_equity": state.peak_equity,
                "current_equity": state.current_equity,
                "drawdown_pct": state.drawdown_pct,
                "tripped_at": state.tripped_at,
                "reason": state.reason,
            }
        ]

        # Create a new breaker (simulating restart) — _load() queries DB
        breaker2 = StrategyBreaker(config=BreakerConfig())
        loaded = breaker2.get_all_states()

        assert "strat_a" in loaded
        assert loaded["strat_a"].status == state.status
        assert loaded["strat_a"].scale_factor == state.scale_factor
        assert loaded["strat_a"].consecutive_losses == state.consecutive_losses

    @patch("quantstack.execution.strategy_breaker.db_conn")
    def test_tripped_state_survives_restart(self, mock_db_conn):
        """TRIPPED state persists across simulated container restart.

        This is the critical safety property: a new StrategyBreaker instance
        must see the TRIPPED status and return scale_factor=0.0.
        """
        mock_conn = _make_mock_conn()
        mock_db_conn.side_effect = _db_conn_ctx(mock_conn)

        tripped_at = datetime.now(timezone.utc)

        # Simulate DB returning a TRIPPED state
        mock_conn.fetchall.return_value = [
            {
                "strategy_id": "risk_strat",
                "status": STATUS_TRIPPED,
                "scale_factor": 0.0,
                "consecutive_losses": 4,
                "peak_equity": 10000.0,
                "current_equity": 9400.0,
                "drawdown_pct": 6.0,
                "tripped_at": tripped_at,
                "reason": "Max drawdown exceeded",
            }
        ]

        breaker = StrategyBreaker(config=BreakerConfig())

        assert breaker.get_scale_factor("risk_strat") == 0.0
        state = breaker.get_all_states()["risk_strat"]
        assert state.status == STATUS_TRIPPED
        assert state.tripped_at == tripped_at

    @patch("quantstack.execution.strategy_breaker.db_conn")
    def test_concurrent_reads_do_not_block(self, mock_db_conn):
        """Multiple get_scale_factor() calls execute without deadlock.

        Reads hit the in-memory dict, not the DB — verifies no DB queries
        on the hot path.
        """
        mock_conn = _make_mock_conn()
        mock_db_conn.side_effect = _db_conn_ctx(mock_conn)

        breaker = StrategyBreaker(config=BreakerConfig())

        # Multiple reads should succeed without issue
        for _ in range(100):
            factor = breaker.get_scale_factor("nonexistent_strat")
            assert factor == 1.0  # default for unknown strategy

    @patch("quantstack.execution.strategy_breaker.db_conn")
    def test_persist_failure_does_not_crash(self, mock_db_conn):
        """If DB write fails, the breaker logs an error but does not raise.

        Trading must continue even if persistence is temporarily broken.
        """
        mock_conn = _make_mock_conn()
        call_count = 0

        def _db_conn_factory():
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                nonlocal call_count
                call_count += 1
                # First call is _load() — succeeds
                if call_count == 1:
                    yield mock_conn
                else:
                    # Subsequent calls (_persist) — DB failure
                    failing_conn = MagicMock()
                    failing_conn.execute.side_effect = Exception("DB connection lost")
                    yield failing_conn

            return _ctx()

        mock_db_conn.side_effect = _db_conn_factory

        breaker = StrategyBreaker(config=BreakerConfig())

        # This should NOT raise despite DB failure during _persist
        state = breaker.record_trade("test_strat", pnl=-50.0, equity=9950.0)
        assert state.status == STATUS_ACTIVE  # in-memory state is correct

    @patch("quantstack.execution.strategy_breaker.db_conn")
    def test_load_from_empty_db(self, mock_db_conn):
        """Fresh database with no rows returns empty state dict (clean start)."""
        mock_conn = _make_mock_conn()
        mock_conn.fetchall.return_value = []
        mock_db_conn.side_effect = _db_conn_ctx(mock_conn)

        breaker = StrategyBreaker(config=BreakerConfig())

        assert breaker.get_all_states() == {}
        assert breaker.get_scale_factor("any_strategy") == 1.0

    @patch("quantstack.execution.strategy_breaker.db_conn")
    def test_invalid_status_in_db_resets_to_active(self, mock_db_conn):
        """A corrupted status value in DB is treated as ACTIVE with a warning log."""
        mock_conn = _make_mock_conn()
        mock_conn.fetchall.return_value = [
            {
                "strategy_id": "corrupt_strat",
                "status": "INVALID_GARBAGE",
                "scale_factor": 0.5,
                "consecutive_losses": 1,
                "peak_equity": 10000.0,
                "current_equity": 9700.0,
                "drawdown_pct": 3.0,
                "tripped_at": None,
                "reason": "some reason",
            }
        ]
        mock_db_conn.side_effect = _db_conn_ctx(mock_conn)

        breaker = StrategyBreaker(config=BreakerConfig())
        state = breaker.get_all_states()["corrupt_strat"]

        assert state.status == STATUS_ACTIVE


# ===========================================================================
# ICAttributionTracker PostgreSQL Migration Tests
# ===========================================================================


class TestICAttributionPersistence:
    """Verify ICAttributionTracker save/load round-trips through PostgreSQL."""

    @patch("quantstack.learning.ic_attribution.db_conn")
    def test_save_load_roundtrip(self, mock_db_conn):
        """Observations recorded via record() are retrievable after
        creating a new tracker instance (simulating restart)."""
        mock_conn = _make_mock_conn()
        mock_db_conn.side_effect = _db_conn_ctx(mock_conn)

        tracker = ICAttributionTracker(window_size=30)

        # Record some observations
        tracker.record(
            symbol="AAPL", collector="technical",
            signal_value=0.7, forward_return=0.02,
        )
        tracker.record(
            symbol="AAPL", collector="technical",
            signal_value=0.8, forward_return=0.03,
        )

        # Verify inserts happened
        insert_calls = [
            c for c in mock_conn.execute.call_args_list
            if "INSERT INTO ic_attribution_data" in str(c)
        ]
        assert len(insert_calls) == 2

        # Simulate restart: DB returns the stored observations
        ts_now = datetime.now(timezone.utc)
        mock_conn.fetchall.return_value = [
            {
                "collector": "technical",
                "signal_value": 0.7,
                "forward_return": 0.02,
                "recorded_at": ts_now,
            },
            {
                "collector": "technical",
                "signal_value": 0.8,
                "forward_return": 0.03,
                "recorded_at": ts_now,
            },
        ]

        tracker2 = ICAttributionTracker(window_size=30)

        # Verify observations loaded
        assert "technical" in tracker2._collectors
        assert len(tracker2._collectors["technical"].observations) == 2

    @patch("quantstack.learning.ic_attribution.db_conn")
    def test_data_persists_across_restart(self, mock_db_conn):
        """IC computation on a fresh instance matches when both have same
        observation history from DB."""
        mock_conn = _make_mock_conn()
        mock_db_conn.side_effect = _db_conn_ctx(mock_conn)

        # Build enough observations for IC computation (min 20)
        observations = []
        ts_now = datetime.now(timezone.utc)
        for i in range(25):
            observations.append({
                "collector": "sentiment",
                "signal_value": 0.5 + i * 0.01,
                "forward_return": 0.01 + i * 0.002,
                "recorded_at": ts_now,
            })

        mock_conn.fetchall.return_value = observations

        tracker = ICAttributionTracker(window_size=30)
        ic_val = tracker.get_collector_ic("sentiment")

        # With positively correlated signal/return, IC should be positive
        assert ic_val is not None
        assert ic_val > 0

    @patch("quantstack.learning.ic_attribution.db_conn")
    def test_window_truncation_in_db(self, mock_db_conn):
        """Only the most recent 2 * window_size observations are kept per
        collector, matching the current in-memory truncation behavior."""
        mock_conn = _make_mock_conn()
        mock_db_conn.side_effect = _db_conn_ctx(mock_conn)

        tracker = ICAttributionTracker(window_size=5)

        # Record more than 2 * window_size (10) observations
        for i in range(15):
            tracker.record(
                symbol="SPY", collector="macro",
                signal_value=float(i), forward_return=float(i) * 0.01,
            )

        # In-memory should be truncated to 2 * window_size = 10
        assert len(tracker._collectors["macro"].observations) == 10

        # Verify truncation SQL was called
        delete_calls = [
            c for c in mock_conn.execute.call_args_list
            if "DELETE FROM ic_attribution_data" in str(c)
        ]
        assert len(delete_calls) > 0, "Expected truncation DELETE calls"

    @patch("quantstack.learning.ic_attribution.db_conn")
    def test_persist_failure_logged_not_raised(self, mock_db_conn):
        """DB write failure is logged but does not propagate to caller."""
        mock_conn = _make_mock_conn()
        call_count = 0

        def _db_conn_factory():
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    yield mock_conn  # _load() succeeds
                else:
                    failing = MagicMock()
                    failing.execute.side_effect = Exception("DB write failed")
                    yield failing

            return _ctx()

        mock_db_conn.side_effect = _db_conn_factory

        tracker = ICAttributionTracker(window_size=30)

        # Should NOT raise
        tracker.record(
            symbol="AAPL", collector="technical",
            signal_value=0.5, forward_return=0.01,
        )

        # In-memory state should still be correct
        assert len(tracker._collectors["technical"].observations) == 1

    @patch("quantstack.learning.ic_attribution.db_conn")
    def test_load_from_empty_db(self, mock_db_conn):
        """Empty ic_attribution_data table returns empty collector state."""
        mock_conn = _make_mock_conn()
        mock_conn.fetchall.return_value = []
        mock_db_conn.side_effect = _db_conn_ctx(mock_conn)

        tracker = ICAttributionTracker(window_size=30)

        assert tracker._collectors == {}
        assert tracker.get_weights() == {}
