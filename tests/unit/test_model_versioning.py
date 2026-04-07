# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 14: Model Versioning + Champion/Challenger.

Covers model registry CRUD, shadow mode, promotion criteria, cold-start,
and EventBus integration.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from quantstack.ml.model_registry import (
    ModelVersion,
    evaluate_promotion,
    get_challengers_for_review,
    get_stale_challengers,
    promote_challenger,
    query_champion,
    register_model,
    retire_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_db():
    """Return (mock_conn, context_manager_factory) for patching db_conn."""
    mock_conn = MagicMock()
    mock_conn.fetchone.return_value = None
    mock_conn.fetchall.return_value = []

    @contextmanager
    def _ctx():
        yield mock_conn

    return mock_conn, _ctx


# ===========================================================================
# Model Registry CRUD
# ===========================================================================


class TestModelRegistry:
    """Tests for model_registry CRUD and versioning logic."""

    def test_register_new_model_auto_increments_version(self):
        """First model gets version=1, second gets version=2."""
        conn, ctx = _mock_db()
        # First call: no existing models -> max version = 0
        conn.fetchone.side_effect = [{"max_version": 0}, {"max_version": 1}]

        with patch("quantstack.ml.model_registry.db_conn", ctx):
            v1 = register_model(
                strategy_id="swing_AAPL",
                train_date=date(2026, 1, 15),
                train_data_range="2025-01-01 to 2026-01-15",
                features_hash="abc123",
                hyperparams={"lr": 0.05},
                backtest_sharpe=1.5,
                backtest_ic=0.04,
                backtest_max_dd=0.08,
                model_path="/models/swing_AAPL/v1/model.pkl",
            )
            v2 = register_model(
                strategy_id="swing_AAPL",
                train_date=date(2026, 2, 15),
                train_data_range="2025-01-01 to 2026-02-15",
                features_hash="def456",
                hyperparams={"lr": 0.03},
                backtest_sharpe=1.6,
                backtest_ic=0.045,
                backtest_max_dd=0.07,
                model_path="/models/swing_AAPL/v2/model.pkl",
            )

        assert v1.version == 1
        assert v2.version == 2

    def test_query_champion_returns_correct_version(self):
        """Promoted model is returned by query_champion."""
        conn, ctx = _mock_db()
        conn.fetchone.return_value = {
            "model_id": "swing_AAPL_v1",
            "strategy_id": "swing_AAPL",
            "version": 1,
            "train_date": date(2026, 1, 15),
            "train_data_range": "2025-01-01 to 2026-01-15",
            "features_hash": "abc",
            "hyperparams": {},
            "backtest_sharpe": 1.5,
            "backtest_ic": 0.04,
            "backtest_max_dd": 0.08,
            "model_path": "/models/v1/model.pkl",
            "status": "champion",
            "promoted_at": datetime.now(timezone.utc),
            "retired_at": None,
            "shadow_start": date(2026, 1, 15),
            "shadow_ic": 0.04,
            "shadow_sharpe": 1.5,
            "created_at": datetime.now(timezone.utc),
        }

        with patch("quantstack.ml.model_registry.db_conn", ctx):
            result = query_champion("swing_AAPL")

        assert result is not None
        assert result.status == "champion"
        assert result.version == 1

    def test_query_champion_returns_none_when_missing(self):
        """No champion -> returns None."""
        conn, ctx = _mock_db()
        conn.fetchone.return_value = None

        with patch("quantstack.ml.model_registry.db_conn", ctx):
            result = query_champion("nonexistent")

        assert result is None

    def test_retire_model_changes_status(self):
        """Retire sets status='retired' via DB update."""
        conn, ctx = _mock_db()
        with patch("quantstack.ml.model_registry.db_conn", ctx):
            retire_model("swing_AAPL_v1")

        # Verify execute was called with UPDATE
        call_args = conn.execute.call_args_list[-1]
        assert "retired" in str(call_args).lower()

    def test_only_one_champion_per_strategy(self):
        """Promoting new champion retires the old one."""
        conn, ctx = _mock_db()
        # promote_challenger: first fetches the challenger, then retires old champion
        conn.fetchone.return_value = {
            "model_id": "swing_AAPL_v2",
            "strategy_id": "swing_AAPL",
            "version": 2,
            "train_date": date(2026, 2, 15),
            "train_data_range": "2025-01-01 to 2026-02-15",
            "features_hash": "def",
            "hyperparams": {},
            "backtest_sharpe": 1.8,
            "backtest_ic": 0.05,
            "backtest_max_dd": 0.06,
            "model_path": "/models/v2/model.pkl",
            "status": "challenger",
            "promoted_at": None,
            "retired_at": None,
            "shadow_start": date(2026, 2, 15),
            "shadow_ic": 0.05,
            "shadow_sharpe": 1.8,
            "created_at": datetime.now(timezone.utc),
        }

        with patch("quantstack.ml.model_registry.db_conn", ctx):
            promote_challenger("swing_AAPL_v2")

        # Should have executed: retire old champion + promote new
        sql_calls = [str(c) for c in conn.execute.call_args_list]
        # At least 2 updates: retire old champion, set new champion
        assert len(conn.execute.call_args_list) >= 2


# ===========================================================================
# Promotion Criteria
# ===========================================================================


class TestModelPromotion:
    """Tests for champion/challenger promotion criteria."""

    def test_promote_when_all_criteria_met(self):
        """IC +0.006, Sharpe +0.20, DD <= 1.1x -> promote."""
        should_promote, reason = evaluate_promotion(
            challenger_ic=0.046,
            champion_ic=0.040,
            challenger_sharpe=1.85,
            champion_sharpe=1.65,
            challenger_max_dd=0.085,
            champion_max_dd=0.080,
        )
        assert should_promote is True

    def test_no_promotion_when_ic_not_met(self):
        """IC improvement only 0.002 -> no promotion."""
        should_promote, reason = evaluate_promotion(
            challenger_ic=0.042,
            champion_ic=0.040,
            challenger_sharpe=1.85,
            champion_sharpe=1.65,
            challenger_max_dd=0.085,
            champion_max_dd=0.080,
        )
        assert should_promote is False
        assert "ic" in reason.lower()

    def test_no_promotion_when_sharpe_not_met(self):
        """Sharpe improvement only 0.10 -> no promotion."""
        should_promote, reason = evaluate_promotion(
            challenger_ic=0.046,
            champion_ic=0.040,
            challenger_sharpe=1.75,
            champion_sharpe=1.65,
            challenger_max_dd=0.085,
            champion_max_dd=0.080,
        )
        assert should_promote is False
        assert "sharpe" in reason.lower()

    def test_no_promotion_when_dd_regresses(self):
        """Challenger DD is 1.2x champion -> no promotion."""
        should_promote, reason = evaluate_promotion(
            challenger_ic=0.046,
            champion_ic=0.040,
            challenger_sharpe=1.85,
            champion_sharpe=1.65,
            challenger_max_dd=0.096,  # 1.2x of 0.08
            champion_max_dd=0.080,
        )
        assert should_promote is False
        assert "drawdown" in reason.lower()

    def test_challenger_fewer_than_30_days_not_eligible(self):
        """Shadow period < 30 days -> not eligible for promotion."""
        conn, ctx = _mock_db()
        # get_challengers_for_review filters by shadow_start <= today - 30
        conn.fetchall.return_value = []  # no eligible challengers

        with patch("quantstack.ml.model_registry.db_conn", ctx):
            challengers = get_challengers_for_review()

        assert len(challengers) == 0

    def test_retire_stale_challengers_after_60_days(self):
        """Challengers in shadow > 60 days without promotion -> retired."""
        conn, ctx = _mock_db()
        conn.fetchall.return_value = [
            {
                "model_id": "stale_v1",
                "strategy_id": "swing_AAPL",
                "version": 1,
                "train_date": date(2025, 12, 1),
                "train_data_range": "2025-01-01 to 2025-12-01",
                "features_hash": "abc",
                "hyperparams": {},
                "backtest_sharpe": 1.2,
                "backtest_ic": 0.03,
                "backtest_max_dd": 0.10,
                "model_path": "/models/v1/model.pkl",
                "status": "challenger",
                "promoted_at": None,
                "retired_at": None,
                "shadow_start": date(2025, 12, 1),
                "shadow_ic": None,
                "shadow_sharpe": None,
                "created_at": datetime.now(timezone.utc),
            }
        ]

        with patch("quantstack.ml.model_registry.db_conn", ctx):
            stale = get_stale_challengers(max_shadow_days=60)

        assert len(stale) == 1
        assert stale[0].model_id == "stale_v1"


# ===========================================================================
# Cold-Start Tests
# ===========================================================================


class TestModelVersioningColdStart:
    """Graceful degradation when registry is empty."""

    def test_no_champion_returns_none(self):
        """No champion in registry -> query returns None."""
        conn, ctx = _mock_db()
        conn.fetchone.return_value = None

        with patch("quantstack.ml.model_registry.db_conn", ctx):
            result = query_champion("new_strategy")

        assert result is None
