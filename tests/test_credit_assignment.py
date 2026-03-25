# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for step-level credit assignment."""

from __future__ import annotations

import pytest

from quantstack.db import pg_conn, run_migrations
from quantstack.optimization.credit_assignment import CreditAssigner, StepCredit


@pytest.fixture
def conn():
    with pg_conn() as c:
        run_migrations(c)
        yield c


@pytest.fixture
def assigner(conn):
    return CreditAssigner(conn)


def _make_context(**overrides) -> dict:
    defaults = dict(
        trade_id=1,
        realized_pnl_pct=-2.5,
        regime_at_entry="trending_up",
        regime_at_exit="trending_up",
        strategy_id="regime_momentum_v1",
        conviction=0.65,
        position_size="half",
        debate_verdict="",
        signals_present=True,
        strategy_regime_affinity=0.8,
    )
    defaults.update(overrides)
    return defaults


class TestHeuristicCredit:
    def test_regime_shift_negative_credit(self, assigner):
        ctx = _make_context(regime_at_entry="trending_up", regime_at_exit="ranging")
        credits = assigner.assign_heuristic(ctx)
        regime_credit = next(c for c in credits if c.step_type == "regime")
        assert regime_credit.credit_score == -0.5
        assert "shifted" in regime_credit.evidence.lower()

    def test_regime_stable_loss_zero_credit(self, assigner):
        ctx = _make_context(regime_at_entry="trending_up", regime_at_exit="trending_up")
        credits = assigner.assign_heuristic(ctx)
        regime_credit = next(c for c in credits if c.step_type == "regime")
        assert regime_credit.credit_score == 0.0

    def test_winning_trade_positive_credits(self, assigner):
        ctx = _make_context(realized_pnl_pct=3.0)
        credits = assigner.assign_heuristic(ctx)
        assert all(c.credit_score >= 0 for c in credits)

    def test_low_affinity_strategy_negative(self, assigner):
        ctx = _make_context(strategy_regime_affinity=0.3)
        credits = assigner.assign_heuristic(ctx)
        strat_credit = next(c for c in credits if c.step_type == "strategy_selection")
        assert strat_credit.credit_score == -0.5

    def test_oversized_position_negative(self, assigner):
        ctx = _make_context(conviction=0.45, position_size="half")
        credits = assigner.assign_heuristic(ctx)
        sizing_credit = next(c for c in credits if c.step_type == "sizing")
        assert sizing_credit.credit_score == -0.3

    def test_missing_signals_negative(self, assigner):
        ctx = _make_context(signals_present=False)
        credits = assigner.assign_heuristic(ctx)
        signal_credit = next(c for c in credits if c.step_type == "signal")
        assert signal_credit.credit_score == -0.4

    def test_debate_passed_but_big_loss(self, assigner):
        ctx = _make_context(realized_pnl_pct=-4.0, debate_verdict="pass")
        credits = assigner.assign_heuristic(ctx)
        debate_credit = next(c for c in credits if c.step_type == "debate")
        assert debate_credit.credit_score == -0.3

    def test_all_step_types_present(self, assigner):
        ctx = _make_context()
        credits = assigner.assign_heuristic(ctx)
        step_types = {c.step_type for c in credits}
        assert step_types == {"signal", "regime", "strategy_selection", "sizing", "debate"}

    def test_persists_to_db(self, assigner, conn):
        ctx = _make_context()
        assigner.assign_heuristic(ctx)
        count = conn.execute("SELECT COUNT(*) FROM step_credits").fetchone()[0]
        assert count == 5  # one per step type


class TestWorstStep:
    def test_returns_lowest_credit(self):
        credits = [
            StepCredit("signal", "", 0.1, "heuristic", ""),
            StepCredit("regime", "", -0.5, "heuristic", ""),
            StepCredit("sizing", "", -0.2, "heuristic", ""),
        ]
        worst = CreditAssigner.get_worst_step(credits)
        assert worst.step_type == "regime"
        assert worst.credit_score == -0.5

    def test_empty_returns_none(self):
        assert CreditAssigner.get_worst_step([]) is None


class TestStatistical:
    def test_heuristic_records_credits(self, assigner):
        """assign_heuristic returns step credits for a losing trade."""
        credits = assigner.assign_heuristic(
            _make_context(trade_id=1, realized_pnl_pct=-2.0)
        )
        assert len(credits) > 0
        assert all(hasattr(c, "step_type") for c in credits)
