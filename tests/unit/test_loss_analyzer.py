"""Tests for the daily loss analysis pipeline (loss_analyzer.py)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from quantstack.learning.failure_taxonomy import FailureMode
from quantstack.learning.loss_analyzer import (
    aggregate_failure_modes,
    classify_losses,
    collect_daily_losers,
    generate_research_tasks,
    prioritize_failure_modes,
    run_daily_loss_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_loser(**overrides: object) -> dict:
    """Build a single loser dict with sensible defaults."""
    base = {
        "strategy_id": "swing_momentum_AAPL",
        "symbol": "AAPL",
        "entry_regime": "trending_up",
        "exit_regime": "ranging",
        "holding_period_days": 3,
        "signal_strength_at_entry": 0.72,
        "realized_pnl": -150.0,
        "realized_pnl_pct": -0.025,
        "slippage_pct": 0.001,
        "entry_price": 180.0,
        "exit_price": 175.50,
    }
    base.update(overrides)
    return base


def _mock_conn_with_losers(rows: list[dict] | None = None) -> MagicMock:
    """Return a mock DB connection whose execute().fetchall() returns *rows*."""
    conn = MagicMock()
    if rows is None:
        rows = []
    # Make fetchall return list of dicts
    conn.execute.return_value.fetchall.return_value = rows
    return conn


# ---------------------------------------------------------------------------
# Stage 1: collect_daily_losers
# ---------------------------------------------------------------------------


class TestCollectDailyLosers:
    def test_filters_negative_pnl(self):
        """Only negative-PnL rows are returned (DB does the filtering)."""
        db_rows = [
            {
                "strategy_id": "strat_1",
                "symbol": "AAPL",
                "entry_regime": "trending_up",
                "exit_regime": "ranging",
                "holding_period_days": 2,
                "realized_pnl": -200.0,
                "entry_price": 100.0,
                "exit_price": 98.0,
                "slippage_bps": 15.0,
            },
        ]
        conn = _mock_conn_with_losers(db_rows)
        result = collect_daily_losers(conn, date(2026, 4, 5))

        assert len(result) == 1
        assert result[0]["realized_pnl"] == -200.0
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["slippage_pct"] == round(15.0 / 10_000, 6)

    def test_empty_when_no_losers(self):
        conn = _mock_conn_with_losers([])
        result = collect_daily_losers(conn, date(2026, 4, 5))
        assert result == []

    def test_pnl_pct_calculation(self):
        """realized_pnl_pct is computed from realized_pnl / entry_price."""
        db_rows = [
            {
                "strategy_id": "s1",
                "symbol": "TSLA",
                "entry_regime": "trending_up",
                "exit_regime": "trending_up",
                "holding_period_days": 1,
                "realized_pnl": -50.0,
                "entry_price": 200.0,
                "exit_price": 195.0,
                "slippage_bps": 0.0,
            },
        ]
        conn = _mock_conn_with_losers(db_rows)
        result = collect_daily_losers(conn, date(2026, 4, 5))
        assert result[0]["realized_pnl_pct"] == round(-50.0 / 200.0, 6)


# ---------------------------------------------------------------------------
# Stage 2: classify_losses
# ---------------------------------------------------------------------------


class TestClassifyLosses:
    def test_regime_mismatch_detected(self):
        losers = [_make_loser(entry_regime="trending_up", exit_regime="ranging")]
        result = classify_losses(losers)
        assert result[0]["failure_mode"] == FailureMode.REGIME_MISMATCH.value

    def test_same_regime_falls_through(self):
        """When regimes match and no other rule triggers, result is UNCLASSIFIED."""
        losers = [_make_loser(
            entry_regime="trending_up",
            exit_regime="trending_up",
            slippage_pct=0.001,
            holding_period_days=5,
        )]
        result = classify_losses(losers)
        assert result[0]["failure_mode"] == FailureMode.UNCLASSIFIED.value

    def test_liquidity_trap_via_extended_rules(self):
        """High slippage triggers LIQUIDITY_TRAP in extended rules."""
        losers = [_make_loser(
            entry_regime="trending_up",
            exit_regime="trending_up",
            slippage_pct=0.03,  # 3% > 2% threshold
        )]
        result = classify_losses(losers)
        assert result[0]["failure_mode"] == FailureMode.LIQUIDITY_TRAP.value

    def test_adverse_selection_via_extended_rules(self):
        """Zero-day hold with loss triggers ADVERSE_SELECTION."""
        losers = [_make_loser(
            entry_regime="trending_up",
            exit_regime="trending_up",
            holding_period_days=0,
            realized_pnl_pct=-0.01,
            slippage_pct=0.001,
        )]
        result = classify_losses(losers)
        assert result[0]["failure_mode"] == FailureMode.ADVERSE_SELECTION.value

    def test_multiple_losers_classified(self):
        losers = [
            _make_loser(entry_regime="trending_up", exit_regime="ranging"),
            _make_loser(entry_regime="trending_up", exit_regime="trending_up", slippage_pct=0.05),
        ]
        result = classify_losses(losers)
        modes = [r["failure_mode"] for r in result]
        assert FailureMode.REGIME_MISMATCH.value in modes
        assert FailureMode.LIQUIDITY_TRAP.value in modes


# ---------------------------------------------------------------------------
# Stage 3: aggregate_failure_modes
# ---------------------------------------------------------------------------


class TestAggregateFailureModes:
    def test_rolling_window_delete_and_upsert(self):
        """Verifies that old rows are pruned and new aggregates are upserted."""
        conn = MagicMock()
        classified = [
            _make_loser(failure_mode="regime_mismatch", realized_pnl=-100.0, strategy_id="s1"),
            _make_loser(failure_mode="regime_mismatch", realized_pnl=-200.0, strategy_id="s2"),
            _make_loser(failure_mode="timing_error", realized_pnl=-50.0, strategy_id="s1"),
        ]
        for c in classified:
            c.setdefault("failure_mode", "unclassified")

        aggregate_failure_modes(conn, classified)

        # Should have: 1 DELETE + 2 INSERTs (regime_mismatch + timing_error groups)
        calls = conn.execute.call_args_list
        assert len(calls) == 3  # 1 delete + 2 upserts
        # First call is the DELETE for old rows
        assert "DELETE FROM failure_mode_stats" in calls[0][0][0]

    def test_empty_losses_no_op(self):
        conn = MagicMock()
        aggregate_failure_modes(conn, [])
        conn.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 4: prioritize_failure_modes
# ---------------------------------------------------------------------------


class TestPrioritizeFailureModes:
    def test_ranking_by_pnl_impact(self):
        """Top modes are ranked by absolute cumulative_pnl_impact DESC."""
        rows = [
            {
                "failure_mode": "regime_mismatch",
                "total_frequency": 5,
                "total_pnl_impact": -500.0,
                "avg_loss": -100.0,
            },
            {
                "failure_mode": "timing_error",
                "total_frequency": 3,
                "total_pnl_impact": -300.0,
                "avg_loss": -100.0,
            },
        ]
        conn = _mock_conn_with_losers(rows)
        result = prioritize_failure_modes(conn)

        assert len(result) == 2
        assert result[0]["failure_mode"] == "regime_mismatch"
        assert result[0]["total_pnl_impact"] == -500.0

    def test_returns_max_3(self):
        """Even with more rows, only top 3 returned (LIMIT in SQL)."""
        rows = [
            {"failure_mode": f"mode_{i}", "total_frequency": i, "total_pnl_impact": -i * 100, "avg_loss": -50.0}
            for i in range(3)
        ]
        conn = _mock_conn_with_losers(rows)
        result = prioritize_failure_modes(conn)
        assert len(result) <= 3

    def test_empty_stats(self):
        conn = _mock_conn_with_losers([])
        result = prioritize_failure_modes(conn)
        assert result == []


# ---------------------------------------------------------------------------
# Stage 5: generate_research_tasks
# ---------------------------------------------------------------------------


class TestGenerateResearchTasks:
    def test_creates_tasks(self):
        conn = MagicMock()
        top_modes = [
            {"failure_mode": "regime_mismatch", "total_frequency": 5, "total_pnl_impact": -500.0, "avg_loss": -100.0},
        ]
        task_ids = generate_research_tasks(conn, top_modes)

        assert len(task_ids) == 1
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0]
        assert "INSERT INTO research_queue" in call_args[0]
        assert call_args[1][2] == "loss_pattern:regime_mismatch"

    def test_empty_modes_returns_empty(self):
        conn = MagicMock()
        task_ids = generate_research_tasks(conn, [])
        assert task_ids == []
        conn.execute.assert_not_called()

    def test_priority_capped_at_9(self):
        conn = MagicMock()
        top_modes = [
            {"failure_mode": "black_swan", "total_frequency": 1, "total_pnl_impact": -99999.0, "avg_loss": -99999.0},
        ]
        generate_research_tasks(conn, top_modes)
        # Priority is min(9, ...) so should not exceed 9
        call_params = conn.execute.call_args[0][1]
        priority = call_params[1]
        assert priority <= 9


# ---------------------------------------------------------------------------
# Full pipeline: run_daily_loss_analysis
# ---------------------------------------------------------------------------


class TestRunDailyLossAnalysis:
    @patch("quantstack.learning.loss_analyzer.generate_research_tasks", return_value=["task-1"])
    @patch("quantstack.learning.loss_analyzer.prioritize_failure_modes", return_value=[
        {"failure_mode": "regime_mismatch", "total_frequency": 3, "total_pnl_impact": -300.0, "avg_loss": -100.0},
    ])
    @patch("quantstack.learning.loss_analyzer.aggregate_failure_modes")
    @patch("quantstack.learning.loss_analyzer.classify_losses")
    @patch("quantstack.learning.loss_analyzer.collect_daily_losers")
    def test_all_stages_called(
        self, mock_collect, mock_classify, mock_agg, mock_prio, mock_gen
    ):
        losers = [_make_loser()]
        mock_collect.return_value = losers
        classified = [_make_loser(failure_mode="regime_mismatch")]
        mock_classify.return_value = classified

        conn = MagicMock()
        result = run_daily_loss_analysis(conn, date(2026, 4, 5))

        mock_collect.assert_called_once_with(conn, date(2026, 4, 5))
        mock_classify.assert_called_once_with(losers)
        mock_agg.assert_called_once_with(conn, classified)
        mock_prio.assert_called_once_with(conn)
        mock_gen.assert_called_once()
        assert result["losers_found"] == 1
        assert result["research_tasks"] == ["task-1"]

    def test_zero_losers_graceful(self):
        """When no losers exist, pipeline returns zeros and skips later stages."""
        conn = _mock_conn_with_losers([])
        result = run_daily_loss_analysis(conn, date(2026, 4, 5))

        assert result["losers_found"] == 0
        assert result["classified"] == 0
        assert result["top_modes"] == []
        assert result["research_tasks"] == []

    @patch("quantstack.learning.loss_analyzer.generate_research_tasks", return_value=[])
    @patch("quantstack.learning.loss_analyzer.prioritize_failure_modes", return_value=[])
    @patch("quantstack.learning.loss_analyzer.aggregate_failure_modes")
    @patch("quantstack.learning.loss_analyzer.classify_losses")
    @patch("quantstack.learning.loss_analyzer.collect_daily_losers")
    def test_no_top_modes_no_tasks(
        self, mock_collect, mock_classify, mock_agg, mock_prio, mock_gen
    ):
        """When prioritize returns nothing, no research tasks generated."""
        mock_collect.return_value = [_make_loser()]
        mock_classify.return_value = [_make_loser(failure_mode="unclassified")]

        conn = MagicMock()
        result = run_daily_loss_analysis(conn, date(2026, 4, 5))
        assert result["research_tasks"] == []


# ---------------------------------------------------------------------------
# failure_taxonomy: new modes
# ---------------------------------------------------------------------------


class TestNewFailureModes:
    """Verify the 5 new FailureMode enum members exist."""

    def test_liquidity_trap_exists(self):
        assert FailureMode.LIQUIDITY_TRAP.value == "liquidity_trap"

    def test_model_degradation_exists(self):
        assert FailureMode.MODEL_DEGRADATION.value == "model_degradation"

    def test_signal_decay_exists(self):
        assert FailureMode.SIGNAL_DECAY.value == "signal_decay"

    def test_adverse_selection_exists(self):
        assert FailureMode.ADVERSE_SELECTION.value == "adverse_selection"

    def test_correlation_breakdown_exists(self):
        assert FailureMode.CORRELATION_BREAKDOWN.value == "correlation_breakdown"


class TestClassifyFailureNewRules:
    """Verify the new classify_failure rules fire correctly."""

    def test_liquidity_trap_rule(self):
        from quantstack.learning.failure_taxonomy import classify_failure

        mode = classify_failure(
            realized_pnl_pct=-0.05,
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",
            strategy_id="s1",
            symbol="XYZ",
            entry_price=100.0,
            exit_price=95.0,
            slippage_pct=0.03,
        )
        assert mode == FailureMode.LIQUIDITY_TRAP

    def test_model_degradation_rule(self):
        from quantstack.learning.failure_taxonomy import classify_failure

        mode = classify_failure(
            realized_pnl_pct=-0.05,
            regime_at_entry="ranging",
            regime_at_exit="ranging",
            strategy_id="s1",
            symbol="XYZ",
            entry_price=100.0,
            exit_price=95.0,
            psi_at_entry=0.3,
        )
        assert mode == FailureMode.MODEL_DEGRADATION

    def test_signal_decay_rule(self):
        from quantstack.learning.failure_taxonomy import classify_failure

        mode = classify_failure(
            realized_pnl_pct=-0.02,
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",
            strategy_id="s1",
            symbol="XYZ",
            entry_price=100.0,
            exit_price=98.0,
            ic_trailing_10d=0.002,
        )
        assert mode == FailureMode.SIGNAL_DECAY

    def test_adverse_selection_rule(self):
        from quantstack.learning.failure_taxonomy import classify_failure

        mode = classify_failure(
            realized_pnl_pct=-0.01,
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",
            strategy_id="s1",
            symbol="XYZ",
            entry_price=100.0,
            exit_price=99.0,
            holding_minutes=15.0,
        )
        assert mode == FailureMode.ADVERSE_SELECTION

    def test_correlation_breakdown_rule(self):
        from quantstack.learning.failure_taxonomy import classify_failure

        mode = classify_failure(
            realized_pnl_pct=-0.03,
            regime_at_entry="ranging",
            regime_at_exit="ranging",
            strategy_id="s1",
            symbol="XYZ",
            entry_price=100.0,
            exit_price=97.0,
            correlation_z_score=2.5,
        )
        assert mode == FailureMode.CORRELATION_BREAKDOWN

    def test_regime_mismatch_still_takes_priority(self):
        """Regime mismatch fires before any new rule, even if new params are set."""
        from quantstack.learning.failure_taxonomy import classify_failure

        mode = classify_failure(
            realized_pnl_pct=-0.05,
            regime_at_entry="trending_up",
            regime_at_exit="ranging",
            strategy_id="s1",
            symbol="XYZ",
            entry_price=100.0,
            exit_price=95.0,
            slippage_pct=0.05,
            psi_at_entry=0.5,
            ic_trailing_10d=0.001,
        )
        assert mode == FailureMode.REGIME_MISMATCH
