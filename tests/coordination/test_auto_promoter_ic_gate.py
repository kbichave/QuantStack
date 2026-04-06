"""Tests for IC gate in auto-promoter (section-04).

Uses a mock PgConnection to avoid requiring a live database. The tests focus
on the IC gate logic: threshold checks, grandfathering, and None handling.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from quantstack.coordination.auto_promoter import (
    AutoPromoter,
    PromotionCriteria,
    PromotionDecision,
)


def _make_promoter(
    icir_value: float | None = 0.35,
    grandfathered_until: datetime | None = None,
    forward_test_days: int = 30,
    trade_count: int = 20,
    win_rate: float = 0.55,
    live_count: int = 3,
    bt_sharpe: float = 1.5,
) -> tuple[AutoPromoter, str]:
    """Build a mock-backed AutoPromoter with configurable IC gate inputs.

    Returns (promoter, strategy_id).
    """
    strategy_id = "test_ic_gate_001"
    now = datetime.now(timezone.utc)
    updated_at = now - timedelta(days=forward_test_days)

    conn = MagicMock()

    def mock_execute(query, params=None):
        q = query.strip().lower()
        result = MagicMock()

        # _get_grandfathered_until
        if "ic_gate_grandfathered_until" in q and "select" in q:
            result.fetchone.return_value = (grandfathered_until,) if grandfathered_until is not None else (None,)
            return result

        # _get_icir
        if "icir_21d" in q and "signal_ic" in q:
            result.fetchone.return_value = (icir_value,) if icir_value is not None else None
            return result

        # _get_forward_test_outcomes
        if "strategy_outcomes" in q:
            # Generate synthetic outcomes that pass all metric gates.
            # Small wins and smaller losses → low drawdown, decent win rate.
            outcomes = []
            wins = int(trade_count * win_rate)
            for i in range(wins):
                outcomes.append((0.015, "win", updated_at + timedelta(days=i), updated_at + timedelta(days=i + 1)))
            for i in range(trade_count - wins):
                outcomes.append((-0.005, "loss", updated_at + timedelta(days=wins + i), updated_at + timedelta(days=wins + i + 1)))
            result.fetchall.return_value = outcomes
            return result

        # _count_live_strategies
        if "count" in q and "status = 'live'" in q:
            result.fetchone.return_value = (live_count,)
            return result

        # Fallback
        result.fetchone.return_value = None
        result.fetchall.return_value = []
        return result

    conn.execute = mock_execute

    bt_summary = json.dumps({"sharpe_ratio": bt_sharpe})
    promoter = AutoPromoter(conn=conn, criteria=PromotionCriteria())
    return promoter, strategy_id, updated_at, bt_summary


class TestICGate:
    """Tests for the IC gate in _evaluate_one."""

    def test_icir_below_threshold_holds(self):
        """Strategy with ICIR 0.28 < 0.3 → hold with reason containing 'ICIR below'."""
        promoter, sid, updated_at, bt = _make_promoter(icir_value=0.28)
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "hold"
        assert "ICIR below" in decision.reason
        assert "0.280" in decision.reason
        assert decision.criteria_results.get("icir_gate") is False

    def test_no_ic_history_holds(self):
        """No rows in signal_ic → hold with 'Insufficient IC history'."""
        promoter, sid, updated_at, bt = _make_promoter(icir_value=None)
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "hold"
        assert "Insufficient IC history" in decision.reason
        assert decision.criteria_results.get("icir_gate") is False

    def test_icir_above_threshold_promotes(self):
        """ICIR 0.35 >= 0.3 with all other gates passing → promote."""
        promoter, sid, updated_at, bt = _make_promoter(icir_value=0.35)
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "promote"
        assert decision.criteria_results.get("icir_gate") is True
        assert decision.evidence.get("icir_21d") == 0.35

    def test_icir_barely_above_promotes(self):
        """ICIR 0.31 (barely above 0.3) → promote."""
        promoter, sid, updated_at, bt = _make_promoter(icir_value=0.31)
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "promote"
        assert decision.criteria_results.get("icir_gate") is True

    def test_grandfathered_skips_ic_gate(self):
        """Grandfathered strategy (future date) → IC gate skipped."""
        future = datetime.now(timezone.utc) + timedelta(days=30)
        promoter, sid, updated_at, bt = _make_promoter(icir_value=0.10, grandfathered_until=future)
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        # IC gate should be None (not checked), and promotion should succeed
        # if other gates pass
        assert decision.criteria_results.get("icir_gate") is None
        assert decision.decision == "promote"

    def test_grandfathered_past_sunset_applies_gate(self):
        """Grandfathered strategy past sunset → IC gate applies normally."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        promoter, sid, updated_at, bt = _make_promoter(icir_value=0.10, grandfathered_until=past)
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "hold"
        assert "ICIR below" in decision.reason
        assert decision.criteria_results.get("icir_gate") is False

    def test_criteria_results_true_when_passes(self):
        """icir_gate is True when ICIR passes."""
        promoter, sid, updated_at, bt = _make_promoter(icir_value=0.5)
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.criteria_results["icir_gate"] is True

    def test_criteria_results_none_when_grandfathered(self):
        """icir_gate is None when grandfathered."""
        future = datetime.now(timezone.utc) + timedelta(days=30)
        promoter, sid, updated_at, bt = _make_promoter(icir_value=None, grandfathered_until=future)
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.criteria_results.get("icir_gate") is None
