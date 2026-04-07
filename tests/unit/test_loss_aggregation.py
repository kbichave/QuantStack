"""Tests for loss aggregation — groups classified losses, ranks, queues research tasks."""

import asyncio
import json
from contextlib import contextmanager
from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

from quantstack.learning.loss_aggregation import run_loss_aggregation


def _make_mock_conn(fetchall_return=None):
    """Build a mock db_conn context manager yielding a connection with fetchall/execute."""
    conn = MagicMock()
    conn.fetchall.return_value = fetchall_return if fetchall_return is not None else []
    conn.execute.return_value = None

    @contextmanager
    def mock_db_conn():
        yield conn

    return mock_db_conn, conn


def _loss_row(failure_mode, strategy_id, symbol, pnl, pnl_pct=None):
    return {
        "failure_mode": failure_mode,
        "strategy_id": strategy_id,
        "symbol": symbol,
        "pnl": pnl,
        "pnl_pct": pnl_pct or round(pnl / 10000, 4),
    }


class TestLossAggregation:
    """Loss aggregation test suite."""

    @patch("quantstack.learning.loss_aggregation.db_conn")
    def test_groups_losses_by_failure_mode_strategy_symbol(self, mock_db):
        rows = [
            _loss_row("stop_loss", "mom_1", "AAPL", -100),
            _loss_row("stop_loss", "mom_1", "AAPL", -200),
            _loss_row("stop_loss", "mom_1", "AAPL", -50),
            _loss_row("late_entry", "rev_2", "TSLA", -300),
            _loss_row("late_entry", "rev_2", "TSLA", -150),
        ]
        conn = MagicMock()
        conn.fetchall.return_value = rows
        call_count = 0

        @contextmanager
        def mock_ctx():
            nonlocal call_count
            call_count += 1
            yield conn

        mock_db.side_effect = mock_ctx
        result = asyncio.run(run_loss_aggregation())

        assert result["groups_found"] == 2
        # Verify per-group aggregates in top_patterns or via groups_found
        patterns = result["top_patterns"]
        by_key = {(p["failure_mode"], p["strategy_id"], p["symbol"]): p for p in patterns}
        assert by_key[("late_entry", "rev_2", "TSLA")]["cumulative_pnl"] == -450
        assert by_key[("late_entry", "rev_2", "TSLA")]["trade_count"] == 2
        assert by_key[("stop_loss", "mom_1", "AAPL")]["cumulative_pnl"] == -350
        assert by_key[("stop_loss", "mom_1", "AAPL")]["trade_count"] == 3

    @patch("quantstack.learning.loss_aggregation.db_conn")
    def test_top_3_patterns_ranked_by_abs_pnl(self, mock_db):
        rows = [
            _loss_row("a", "s1", "X", -100),
            _loss_row("b", "s2", "Y", -500),
            _loss_row("c", "s3", "Z", -300),
            _loss_row("d", "s4", "W", -50),
            _loss_row("e", "s5", "V", -900),
        ]
        conn = MagicMock()
        conn.fetchall.return_value = rows

        @contextmanager
        def mock_ctx():
            yield conn

        mock_db.side_effect = mock_ctx
        result = asyncio.run(run_loss_aggregation())

        top = result["top_patterns"]
        assert len(top) == 3
        # Ordered by descending abs(cumulative_pnl): 900, 500, 300
        assert top[0]["cumulative_pnl"] == -900
        assert top[1]["cumulative_pnl"] == -500
        assert top[2]["cumulative_pnl"] == -300

    @patch("quantstack.learning.loss_aggregation.db_conn")
    def test_auto_generates_research_tasks(self, mock_db):
        rows = [
            _loss_row("stop_loss", "s1", "AAPL", -200),
            _loss_row("late_entry", "s2", "TSLA", -400),
            _loss_row("regime_miss", "s3", "NVDA", -600),
        ]
        conn = MagicMock()
        conn.fetchall.return_value = rows

        @contextmanager
        def mock_ctx():
            yield conn

        mock_db.side_effect = mock_ctx
        result = asyncio.run(run_loss_aggregation())

        assert result["tasks_created"] == 3
        # Find INSERT INTO research_queue calls
        rq_calls = [
            c for c in conn.execute.call_args_list
            if "research_queue" in str(c)
        ]
        assert len(rq_calls) == 3
        # task_type should be the failure_mode, not 'bug_fix'
        for c in rq_calls:
            args = c[0]  # positional args
            params = args[1]  # second positional: parameter list
            task_type = params[0]
            assert task_type != "bug_fix"
            assert task_type in ("stop_loss", "late_entry", "regime_miss")

    @patch("quantstack.learning.loss_aggregation.db_conn")
    def test_aggregation_stored_in_table(self, mock_db):
        rows = [
            _loss_row("stop_loss", "s1", "AAPL", -200),
        ]
        conn = MagicMock()
        conn.fetchall.return_value = rows

        @contextmanager
        def mock_ctx():
            yield conn

        mock_db.side_effect = mock_ctx
        result = asyncio.run(run_loss_aggregation())

        la_calls = [
            c for c in conn.execute.call_args_list
            if "loss_aggregation" in str(c)
        ]
        assert len(la_calls) >= 1
        sql = la_calls[0][0][0]
        params = la_calls[0][0][1]
        assert "INSERT INTO loss_aggregation" in sql
        assert params[0] == date.today()  # date
        assert params[1] == "stop_loss"   # failure_mode
        assert params[4] == 1             # trade_count
        assert params[5] == -200          # cumulative_pnl

    @patch("quantstack.learning.loss_aggregation.db_conn")
    def test_empty_losses_graceful(self, mock_db):
        conn = MagicMock()
        conn.fetchall.return_value = []

        @contextmanager
        def mock_ctx():
            yield conn

        mock_db.side_effect = mock_ctx
        result = asyncio.run(run_loss_aggregation())

        assert result == {"groups_found": 0, "tasks_created": 0, "top_patterns": []}
        # No INSERT calls should have been made
        conn.execute.assert_not_called()

    @patch("quantstack.learning.loss_aggregation.db_conn")
    def test_unclassified_losses_included(self, mock_db):
        rows = [
            _loss_row(None, "s1", "AAPL", -100),
            {"failure_mode": None, "strategy_id": "s1", "symbol": "AAPL", "pnl": -50, "pnl_pct": -0.005},
        ]
        conn = MagicMock()
        conn.fetchall.return_value = rows

        @contextmanager
        def mock_ctx():
            yield conn

        mock_db.side_effect = mock_ctx
        result = asyncio.run(run_loss_aggregation())

        assert result["groups_found"] == 1
        assert result["top_patterns"][0]["failure_mode"] == "unclassified"
        assert result["top_patterns"][0]["cumulative_pnl"] == -150

    @patch("quantstack.learning.loss_aggregation.db_conn")
    def test_rerun_upserts_no_duplicates(self, mock_db):
        rows = [_loss_row("stop_loss", "s1", "AAPL", -200)]
        conn = MagicMock()
        conn.fetchall.return_value = rows

        @contextmanager
        def mock_ctx():
            yield conn

        mock_db.side_effect = mock_ctx
        asyncio.run(run_loss_aggregation())

        la_calls = [
            c for c in conn.execute.call_args_list
            if "loss_aggregation" in str(c)
        ]
        assert len(la_calls) >= 1
        sql = la_calls[0][0][0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql

    @patch("quantstack.learning.loss_aggregation.db_conn")
    def test_priority_scales_with_loss_magnitude(self, mock_db):
        rows = [
            _loss_row("a", "s1", "X", -500),
            _loss_row("b", "s2", "Y", -1200),
            _loss_row("c", "s3", "Z", -50),
        ]
        conn = MagicMock()
        conn.fetchall.return_value = rows

        @contextmanager
        def mock_ctx():
            yield conn

        mock_db.side_effect = mock_ctx
        result = asyncio.run(run_loss_aggregation())

        top = result["top_patterns"]
        by_fm = {p["failure_mode"]: p for p in top}
        # $500 loss -> min(9, int(500/100)) = min(9,5) = 5
        assert by_fm["a"]["priority"] == 5
        # $1200 loss -> min(9, int(1200/100)) = min(9,12) = 9
        assert by_fm["b"]["priority"] == 9
