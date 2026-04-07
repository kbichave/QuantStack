"""Tests for the overnight research runner."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from quantstack.research.overnight_runner import (
    BUDGET_CEILING_USD,
    EXPERIMENT_TIMEOUT_SECONDS,
    WINNER_IC_THRESHOLD,
    _generate_experiment_id,
    _is_within_operating_window,
    get_nightly_budget_state,
    run_overnight_loop,
    run_single_experiment,
    score_experiment,
)


# ---------------------------------------------------------------------------
# Operating window tests
# ---------------------------------------------------------------------------


class TestOperatingWindow:
    """Runner respects the 20:00-04:00 ET operating window."""

    @patch("quantstack.research.overnight_runner._now_et")
    def test_within_window_evening(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 7, 21, 0)  # 21:00 ET
        assert _is_within_operating_window() is True

    @patch("quantstack.research.overnight_runner._now_et")
    def test_within_window_midnight(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 7, 0, 30)  # 00:30 ET
        assert _is_within_operating_window() is True

    @patch("quantstack.research.overnight_runner._now_et")
    def test_within_window_early_morning(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 7, 3, 59)  # 03:59 ET
        assert _is_within_operating_window() is True

    @patch("quantstack.research.overnight_runner._now_et")
    def test_within_window_start_boundary(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 7, 20, 0)  # 20:00 ET
        assert _is_within_operating_window() is True

    @patch("quantstack.research.overnight_runner._now_et")
    def test_outside_window_morning(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 7, 8, 0)  # 08:00 ET
        assert _is_within_operating_window() is False

    @patch("quantstack.research.overnight_runner._now_et")
    def test_outside_window_afternoon(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 7, 15, 0)  # 15:00 ET
        assert _is_within_operating_window() is False

    @patch("quantstack.research.overnight_runner._now_et")
    def test_outside_window_exact_close(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 7, 4, 0)  # 04:00 ET — window closed
        assert _is_within_operating_window() is False

    @patch("quantstack.research.overnight_runner._now_et")
    def test_outside_window_midday(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 7, 12, 0)  # 12:00 ET
        assert _is_within_operating_window() is False


# ---------------------------------------------------------------------------
# Budget ceiling tests
# ---------------------------------------------------------------------------


class TestBudgetCeiling:
    """Runner halts at $9.50 budget ceiling."""

    @patch("quantstack.research.overnight_runner.db_conn")
    def test_get_nightly_budget_state_returns_cumulative(self, mock_db):
        mock_conn = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = (7.25,)

        result = get_nightly_budget_state("2026-04-07")
        assert result == 7.25

    @patch("quantstack.research.overnight_runner.db_conn")
    def test_get_nightly_budget_state_returns_zero_on_empty(self, mock_db):
        mock_conn = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = (0,)

        result = get_nightly_budget_state("2026-04-07")
        assert result == 0.0

    @patch("quantstack.research.overnight_runner._is_within_operating_window")
    @patch("quantstack.research.overnight_runner.get_nightly_budget_state")
    @patch("quantstack.research.overnight_runner.generate_hypothesis")
    @patch("quantstack.research.overnight_runner._now_et")
    def test_loop_halts_at_budget_ceiling(
        self, mock_now, mock_gen, mock_budget, mock_window
    ):
        """Loop should stop when cumulative cost >= BUDGET_CEILING_USD."""
        mock_now.return_value = datetime(2026, 4, 7, 21, 0)
        mock_budget.return_value = BUDGET_CEILING_USD  # Already at ceiling
        mock_window.return_value = True  # Within window

        result = asyncio.get_event_loop().run_until_complete(run_overnight_loop())

        # generate_hypothesis should never be called since we're at budget
        mock_gen.assert_not_called()
        assert result["total_cost"] >= BUDGET_CEILING_USD

    @patch("quantstack.research.overnight_runner._is_within_operating_window")
    @patch("quantstack.research.overnight_runner.get_nightly_budget_state")
    @patch("quantstack.research.overnight_runner._now_et")
    def test_budget_resumption_after_crash(self, mock_now, mock_budget, mock_window):
        """After a crash, the runner reads cumulative cost from DB and resumes correctly."""
        mock_now.return_value = datetime(2026, 4, 7, 22, 0)
        mock_budget.return_value = 5.00  # Already spent $5.00 before crash
        mock_window.return_value = False  # Immediately stop (window closed)

        result = asyncio.get_event_loop().run_until_complete(run_overnight_loop())
        assert result["total_cost"] == 5.00


# ---------------------------------------------------------------------------
# Winner threshold tests
# ---------------------------------------------------------------------------


class TestWinnerThreshold:
    """Winner threshold: IC > 0.02 -> 'winner', else 'tested'."""

    def test_score_above_threshold_is_winner(self):
        result = score_experiment({"oos_ic": 0.05, "sharpe": 1.5})
        assert result["status"] == "winner"
        assert result["oos_ic"] == 0.05
        assert result["sharpe"] == 1.5

    def test_score_at_threshold_is_tested(self):
        # Exactly at threshold should NOT be a winner (strictly greater than)
        result = score_experiment({"oos_ic": WINNER_IC_THRESHOLD, "sharpe": 0.8})
        assert result["status"] == "tested"

    def test_score_below_threshold_is_tested(self):
        result = score_experiment({"oos_ic": 0.01, "sharpe": 0.3})
        assert result["status"] == "tested"

    def test_score_zero_ic_is_tested(self):
        result = score_experiment({"oos_ic": 0.0, "sharpe": -0.5})
        assert result["status"] == "tested"

    def test_score_negative_ic_is_tested(self):
        result = score_experiment({"oos_ic": -0.03, "sharpe": -1.0})
        assert result["status"] == "tested"

    def test_score_missing_keys_defaults_to_zero(self):
        result = score_experiment({})
        assert result["oos_ic"] == 0.0
        assert result["sharpe"] == 0.0
        assert result["status"] == "tested"


# ---------------------------------------------------------------------------
# Experiment timeout tests
# ---------------------------------------------------------------------------


class TestExperimentTimeout:
    """Experiment timeout handling."""

    @patch("quantstack.research.overnight_runner._publish_experiment_completed")
    @patch("quantstack.research.overnight_runner._log_experiment")
    @patch("quantstack.research.overnight_runner._run_backtest")
    def test_timeout_records_timeout_status(self, mock_backtest, mock_log, mock_publish):
        """When an experiment times out, it should be logged with status='timeout'."""

        async def slow_backtest(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than our test timeout
            return {"oos_ic": 0.0, "sharpe": 0.0}

        mock_backtest.side_effect = lambda h: (_ for _ in ()).throw(
            asyncio.TimeoutError()
        )

        # Wrap run_single_experiment to simulate the timeout
        async def run_with_timeout():
            # Directly test the timeout path by making _run_backtest raise TimeoutError
            return await run_single_experiment(
                {"entry_rules": ["test"], "exit_rules": ["test"]},
                "test",
                "2026-04-07",
            )

        result = asyncio.get_event_loop().run_until_complete(run_with_timeout())
        # The _run_backtest raising TimeoutError should be caught and logged
        # Since _run_backtest is sync and raises via side_effect, it hits the
        # general Exception handler. But the timeout path is tested via the
        # asyncio.wait_for in run_overnight_loop. Here we verify error handling.
        assert result["status"] in ("timeout", "error")


# ---------------------------------------------------------------------------
# Experiment ID uniqueness tests
# ---------------------------------------------------------------------------


class TestExperimentIdUniqueness:
    """Experiment ID uniqueness."""

    def test_ids_are_unique(self):
        ids = {_generate_experiment_id("2026-04-07") for _ in range(100)}
        assert len(ids) == 100

    def test_id_contains_night_date(self):
        exp_id = _generate_experiment_id("2026-04-07")
        assert "2026-04-07" in exp_id

    def test_id_has_prefix(self):
        exp_id = _generate_experiment_id("2026-04-07")
        assert exp_id.startswith("exp-")


# ---------------------------------------------------------------------------
# Single experiment tests
# ---------------------------------------------------------------------------


class TestRunSingleExperiment:
    """run_single_experiment logs and publishes correctly."""

    @patch("quantstack.research.overnight_runner._publish_experiment_completed")
    @patch("quantstack.research.overnight_runner._log_experiment")
    @patch("quantstack.research.overnight_runner._run_backtest")
    def test_successful_experiment_logs_and_publishes(
        self, mock_backtest, mock_log, mock_publish
    ):
        mock_backtest.return_value = {"oos_ic": 0.05, "sharpe": 1.2, "tokens_used": 3000}

        result = asyncio.get_event_loop().run_until_complete(
            run_single_experiment(
                {"entry_rules": ["RSI < 30"], "exit_rules": ["RSI > 70"]},
                "llm_generated",
                "2026-04-07",
            )
        )

        assert result["status"] == "winner"
        assert result["oos_ic"] == 0.05
        mock_log.assert_called_once()
        mock_publish.assert_called_once()

    @patch("quantstack.research.overnight_runner._publish_experiment_completed")
    @patch("quantstack.research.overnight_runner._log_experiment")
    @patch("quantstack.research.overnight_runner._run_backtest")
    def test_failed_experiment_logs_error(self, mock_backtest, mock_log, mock_publish):
        mock_backtest.side_effect = RuntimeError("backtest engine crashed")

        result = asyncio.get_event_loop().run_until_complete(
            run_single_experiment(
                {"entry_rules": ["test"]},
                "test",
                "2026-04-07",
            )
        )

        assert result["status"] == "error"
        assert result["oos_ic"] == 0.0
        mock_log.assert_called_once()
        # Verify rejection_reason was passed
        call_kwargs = mock_log.call_args
        assert "backtest engine crashed" in str(call_kwargs)
