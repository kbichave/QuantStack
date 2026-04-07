"""Tests for the morning validator."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, call, patch

import pytest

from quantstack.core.backtesting.patience import PatienceConfig, WindowResult
from quantstack.research.morning_validator import (
    _fetch_winners,
    _update_experiment_status,
    register_draft_strategy,
    run_morning_validation,
    validate_winner,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _make_experiment(
    experiment_id: str = "exp-2026-04-07-abc12345",
    oos_ic: float = 0.05,
    sharpe: float = 1.5,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "hypothesis": {
            "entry_rules": ["RSI < 30"],
            "exit_rules": ["RSI > 70"],
            "parameters": {"lookback": 14},
        },
        "hypothesis_source": "llm_generated",
        "oos_ic": oos_ic,
        "sharpe": sharpe,
    }


def _make_window_results(passed_count: int) -> list[WindowResult]:
    """Create window results with a given number of passing windows."""
    windows = [
        ("full", passed_count >= 1),
        ("recent", passed_count >= 2),
        ("stressed", passed_count >= 3),
    ]
    return [
        WindowResult(
            window_name=name,
            passed=passed,
            sharpe=1.2 if passed else -0.5,
            max_drawdown=0.08 if passed else 0.25,
            ic=0.04 if passed else 0.005,
        )
        for name, passed in windows
    ]


# ---------------------------------------------------------------------------
# Processes all winners
# ---------------------------------------------------------------------------


class TestProcessAllWinners:
    """run_morning_validation processes all winners."""

    @patch("quantstack.research.morning_validator._update_experiment_status")
    @patch("quantstack.research.morning_validator.register_draft_strategy")
    @patch("quantstack.research.morning_validator.validate_winner")
    @patch("quantstack.research.morning_validator._fetch_winners")
    def test_processes_every_winner(
        self, mock_fetch, mock_validate, mock_register, mock_update
    ):
        winners = [_make_experiment(f"exp-{i}") for i in range(5)]
        mock_fetch.return_value = winners
        mock_validate.return_value = {
            "status": "draft",
            "provisional": False,
            "window_results": _make_window_results(3),
            "rejection_reason": None,
        }
        mock_register.return_value = "strat-exp-0"

        result = asyncio.get_event_loop().run_until_complete(
            run_morning_validation("2026-04-07")
        )

        assert result["total_winners"] == 5
        assert mock_validate.call_count == 5
        assert mock_register.call_count == 5

    @patch("quantstack.research.morning_validator._update_experiment_status")
    @patch("quantstack.research.morning_validator.register_draft_strategy")
    @patch("quantstack.research.morning_validator.validate_winner")
    @patch("quantstack.research.morning_validator._fetch_winners")
    def test_mixed_draft_and_rejected(
        self, mock_fetch, mock_validate, mock_register, mock_update
    ):
        winners = [_make_experiment(f"exp-{i}") for i in range(3)]
        mock_fetch.return_value = winners

        # First passes, second fails, third passes
        mock_validate.side_effect = [
            {"status": "draft", "provisional": False, "window_results": _make_window_results(3), "rejection_reason": None},
            {"status": "rejected", "provisional": False, "window_results": _make_window_results(1), "rejection_reason": "Failed patience windows: recent, stressed"},
            {"status": "draft", "provisional": True, "window_results": _make_window_results(2), "rejection_reason": None},
        ]
        mock_register.return_value = "strat-id"

        result = asyncio.get_event_loop().run_until_complete(
            run_morning_validation("2026-04-07")
        )

        assert result["drafted"] == 2
        assert result["rejected"] == 1
        assert mock_register.call_count == 2


# ---------------------------------------------------------------------------
# 3-window patience protocol
# ---------------------------------------------------------------------------


class TestPatienceProtocol:
    """Uses 3-window patience protocol for validation."""

    @patch("quantstack.research.morning_validator._run_patience_windows")
    def test_all_three_pass_gives_draft_non_provisional(self, mock_windows):
        mock_windows.return_value = _make_window_results(3)

        result = validate_winner(_make_experiment())

        assert result["status"] == "draft"
        assert result["provisional"] is False
        assert result["rejection_reason"] is None

    @patch("quantstack.research.morning_validator._run_patience_windows")
    def test_two_of_three_pass_gives_draft_provisional(self, mock_windows):
        mock_windows.return_value = _make_window_results(2)

        result = validate_winner(_make_experiment())

        assert result["status"] == "draft"
        assert result["provisional"] is True
        assert result["rejection_reason"] is None

    @patch("quantstack.research.morning_validator._run_patience_windows")
    def test_one_of_three_pass_gives_rejected(self, mock_windows):
        mock_windows.return_value = _make_window_results(1)

        result = validate_winner(_make_experiment())

        assert result["status"] == "rejected"
        assert result["provisional"] is False
        assert "recent" in result["rejection_reason"]
        assert "stressed" in result["rejection_reason"]

    @patch("quantstack.research.morning_validator._run_patience_windows")
    def test_zero_pass_gives_rejected(self, mock_windows):
        mock_windows.return_value = _make_window_results(0)

        result = validate_winner(_make_experiment())

        assert result["status"] == "rejected"
        assert "full" in result["rejection_reason"]
        assert "recent" in result["rejection_reason"]
        assert "stressed" in result["rejection_reason"]


# ---------------------------------------------------------------------------
# Draft strategy registration
# ---------------------------------------------------------------------------


class TestRegisterDraftStrategy:
    """Registers passing winners as draft strategies."""

    @patch("quantstack.research.morning_validator.db_conn")
    def test_registers_draft_strategy(self, mock_db):
        mock_conn = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        experiment = _make_experiment()
        validation = {"status": "draft", "provisional": False}

        strategy_id = register_draft_strategy(experiment, validation)

        assert strategy_id is not None
        assert strategy_id.startswith("strat-")
        mock_conn.execute.assert_called_once()

    @patch("quantstack.research.morning_validator.db_conn")
    def test_does_not_register_rejected(self, mock_db):
        experiment = _make_experiment()
        validation = {"status": "rejected", "provisional": False}

        strategy_id = register_draft_strategy(experiment, validation)

        assert strategy_id is None
        mock_db.assert_not_called()

    @patch("quantstack.research.morning_validator.db_conn")
    def test_registers_provisional_flag(self, mock_db):
        mock_conn = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        experiment = _make_experiment()
        validation = {"status": "draft", "provisional": True}

        strategy_id = register_draft_strategy(experiment, validation)

        assert strategy_id is not None
        # Verify provisional=True was passed to the DB
        insert_args = mock_conn.execute.call_args[0][1]
        # provisional is the 7th positional param (index 6):
        # strategy_id, name, status, hypothesis, oos_ic, sharpe, provisional, ...
        assert insert_args[6] is True


# ---------------------------------------------------------------------------
# Rejection logging
# ---------------------------------------------------------------------------


class TestRejectionLogging:
    """Logs rejection reasons for failures."""

    @patch("quantstack.research.morning_validator._update_experiment_status")
    @patch("quantstack.research.morning_validator.register_draft_strategy")
    @patch("quantstack.research.morning_validator.validate_winner")
    @patch("quantstack.research.morning_validator._fetch_winners")
    def test_rejected_experiment_gets_reason_logged(
        self, mock_fetch, mock_validate, mock_register, mock_update
    ):
        mock_fetch.return_value = [_make_experiment("exp-rejected-1")]
        mock_validate.return_value = {
            "status": "rejected",
            "provisional": False,
            "window_results": _make_window_results(0),
            "rejection_reason": "Failed patience windows: full, recent, stressed",
        }

        asyncio.get_event_loop().run_until_complete(
            run_morning_validation("2026-04-07")
        )

        mock_update.assert_called_with(
            "exp-rejected-1",
            "rejected",
            "Failed patience windows: full, recent, stressed",
        )

    @patch("quantstack.research.morning_validator._update_experiment_status")
    @patch("quantstack.research.morning_validator.register_draft_strategy")
    @patch("quantstack.research.morning_validator.validate_winner")
    @patch("quantstack.research.morning_validator._fetch_winners")
    def test_validation_exception_logs_error_reason(
        self, mock_fetch, mock_validate, mock_register, mock_update
    ):
        mock_fetch.return_value = [_make_experiment("exp-error-1")]
        mock_validate.side_effect = RuntimeError("backtest engine unavailable")

        result = asyncio.get_event_loop().run_until_complete(
            run_morning_validation("2026-04-07")
        )

        assert result["rejected"] == 1
        mock_update.assert_called_once()
        update_args = mock_update.call_args[0]
        assert update_args[0] == "exp-error-1"
        assert update_args[1] == "rejected"
        assert "backtest engine unavailable" in update_args[2]


# ---------------------------------------------------------------------------
# Zero winners
# ---------------------------------------------------------------------------


class TestZeroWinners:
    """Handles zero winners gracefully."""

    @patch("quantstack.research.morning_validator._fetch_winners")
    def test_zero_winners_returns_empty_summary(self, mock_fetch):
        mock_fetch.return_value = []

        result = asyncio.get_event_loop().run_until_complete(
            run_morning_validation("2026-04-07")
        )

        assert result["total_winners"] == 0
        assert result["drafted"] == 0
        assert result["rejected"] == 0
        assert result["strategies_registered"] == []
