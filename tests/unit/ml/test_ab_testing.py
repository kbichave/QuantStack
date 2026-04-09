"""Tests for A/B testing framework."""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantstack.ml.ab_testing import ABTestManager, ABTestResult


class TestABTestManager:
    """Test A/B swap and rollback logic without DB dependency."""

    def test_swap_after_2_consecutive_wins(self):
        """Challenger that wins 2 consecutive weeks triggers swap."""
        manager = ABTestManager()

        # Mock _compute_weekly_ics to return controlled values
        # Most recent week first: challenger wins weeks 0 and 1
        with patch.object(manager, "_compute_weekly_ics") as mock_ics:
            mock_ics.side_effect = lambda model_id, n_weeks=4: (
                [0.05, 0.04, 0.06, 0.03]  # champion
                if "v1" in model_id
                else [0.08, 0.07, 0.02, 0.01]  # challenger wins 2 recent
            )

            # Mock DB interactions
            champion_mock = MagicMock()
            champion_mock.model_id = "SPY_v1"
            champion_mock.backtest_ic = 0.05

            with patch("quantstack.ml.ab_testing.query_champion", return_value=champion_mock):
                with patch("quantstack.ml.ab_testing.db_conn") as mock_db:
                    mock_conn = MagicMock()
                    mock_conn.fetchall.return_value = [{"model_id": "SPY_v2"}]
                    mock_db.return_value.__enter__ = lambda s: mock_conn
                    mock_db.return_value.__exit__ = MagicMock(return_value=False)

                    result = manager.evaluate_weekly("SPY")

        assert result.should_swap is True
        assert result.consecutive_wins >= 2

    def test_no_swap_after_1_win(self):
        """Challenger that wins 1 week then loses does not trigger swap."""
        manager = ABTestManager()

        with patch.object(manager, "_compute_weekly_ics") as mock_ics:
            mock_ics.side_effect = lambda model_id, n_weeks=4: (
                [0.05, 0.06, 0.04, 0.03]  # champion wins week 1
                if "v1" in model_id
                else [0.08, 0.03, 0.02, 0.01]  # challenger wins only week 0
            )

            champion_mock = MagicMock()
            champion_mock.model_id = "SPY_v1"

            with patch("quantstack.ml.ab_testing.query_champion", return_value=champion_mock):
                with patch("quantstack.ml.ab_testing.db_conn") as mock_db:
                    mock_conn = MagicMock()
                    mock_conn.fetchall.return_value = [{"model_id": "SPY_v2"}]
                    mock_db.return_value.__enter__ = lambda s: mock_conn
                    mock_db.return_value.__exit__ = MagicMock(return_value=False)

                    result = manager.evaluate_weekly("SPY")

        assert result.should_swap is False
        assert result.consecutive_wins == 1

    def test_no_champion_returns_no_action(self):
        """No champion model returns early with no_champion reason."""
        manager = ABTestManager()

        with patch("quantstack.ml.ab_testing.query_champion", return_value=None):
            result = manager.evaluate_weekly("SPY")

        assert result.should_swap is False
        assert result.reason == "no_champion"

    def test_rollback_on_50pct_ic_drop(self):
        """Champion IC dropping below 50% of backtest triggers rollback."""
        manager = ABTestManager()

        champion_mock = MagicMock()
        champion_mock.model_id = "SPY_v1"
        champion_mock.backtest_ic = 0.10

        # Mock DB to return predictions with anti-correlated realized returns
        # Predictions go 0→0.95, realized go 0.95→0 — guarantees negative IC
        rows = [
            {"prediction": float(i) / 20, "realized_return": float(19 - i) / 20}
            for i in range(20)
        ]

        with patch("quantstack.ml.ab_testing.query_champion", return_value=champion_mock):
            with patch("quantstack.ml.ab_testing.db_conn") as mock_db:
                mock_conn = MagicMock()
                mock_conn.fetchall.return_value = rows
                mock_db.return_value.__enter__ = lambda s: mock_conn
                mock_db.return_value.__exit__ = MagicMock(return_value=False)

                should_rollback, reason = manager.check_rollback("SPY")

        # Anti-correlated: IC ≈ -1.0, well below 0.05 (50% of 0.10)
        assert should_rollback is True
        assert "rolling IC" in reason

    def test_no_rollback_normal_ic(self):
        """Healthy IC does not trigger rollback."""
        manager = ABTestManager()

        champion_mock = MagicMock()
        champion_mock.model_id = "SPY_v1"
        champion_mock.backtest_ic = 0.10

        # Mock perfectly correlated predictions
        rows = [
            {"prediction": float(i) / 20, "realized_return": float(i) / 20}
            for i in range(20)
        ]

        with patch("quantstack.ml.ab_testing.query_champion", return_value=champion_mock):
            with patch("quantstack.ml.ab_testing.db_conn") as mock_db:
                mock_conn = MagicMock()
                mock_conn.fetchall.return_value = rows
                mock_db.return_value.__enter__ = lambda s: mock_conn
                mock_db.return_value.__exit__ = MagicMock(return_value=False)

                should_rollback, reason = manager.check_rollback("SPY")

        assert should_rollback is False
        assert "healthy" in reason

    def test_ab_test_result_dataclass(self):
        """ABTestResult fields have correct defaults."""
        result = ABTestResult(strategy_id="AAPL", champion_id="v1", challenger_id="v2")
        assert result.should_swap is False
        assert result.should_rollback is False
        assert result.consecutive_wins == 0
        assert result.champion_weekly_ics == []
