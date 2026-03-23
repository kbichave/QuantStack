# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ShadowEvaluator.

All tests use in-memory DuckDB — no file I/O, no external services.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import duckdb
import numpy as np
import pytest
from quantstack.rl.shadow_mode import ShadowEvaluator


def _make_store() -> MagicMock:
    conn = duckdb.connect(":memory:")
    store = MagicMock()
    store.conn = conn
    return store


def _make_evaluator():
    return ShadowEvaluator(_make_store())


class TestShadowEvaluatorInit:
    def test_creates_table(self):
        store = _make_store()
        ShadowEvaluator(store)
        tables = store.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_type='BASE TABLE'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "rl_shadow_decisions" in table_names

    def test_idempotent_init(self):
        store = _make_store()
        ShadowEvaluator(store)
        ShadowEvaluator(store)  # should not raise


class TestRecordDecision:
    def test_returns_decision_id(self):
        ev = _make_evaluator()
        decision_id = ev.record_decision(
            tool_name="rl_position_size",
            rl_recommendation={"scale": 0.7},
            crew_action="buy",
            crew_confidence=0.8,
            symbol="SPY",
        )
        assert isinstance(decision_id, str)
        assert len(decision_id) > 0

    def test_decision_stored_in_db(self):
        store = _make_store()
        ev = ShadowEvaluator(store)
        decision_id = ev.record_decision(
            tool_name="rl_position_size",
            rl_recommendation={"scale": 0.6},
        )
        row = store.conn.execute(
            "SELECT decision_id FROM rl_shadow_decisions WHERE decision_id = ?",
            [decision_id],
        ).fetchone()
        assert row is not None
        assert row[0] == decision_id

    def test_multiple_decisions(self):
        store = _make_store()
        ev = ShadowEvaluator(store)
        for i in range(5):
            ev.record_decision("rl_position_size", {"scale": i * 0.1})
        count = store.conn.execute(
            "SELECT COUNT(*) FROM rl_shadow_decisions"
        ).fetchone()[0]
        assert count == 5


class TestRecordOutcome:
    def test_updates_pnl(self):
        store = _make_store()
        ev = ShadowEvaluator(store)
        did = ev.record_decision("rl_position_size", {"scale": 0.5})
        ev.record_outcome(did, pnl=250.0, slippage_bps=3.2)
        row = store.conn.execute(
            "SELECT pnl, slippage_bps FROM rl_shadow_decisions WHERE decision_id = ?",
            [did],
        ).fetchone()
        assert row[0] == pytest.approx(250.0)
        assert row[1] == pytest.approx(3.2)

    def test_outcome_for_missing_id_does_not_raise(self):
        ev = _make_evaluator()
        # Should silently ignore unknown id
        ev.record_outcome("nonexistent-id", pnl=100.0)


class TestEvaluateShadowPeriod:
    def _fill_decisions(self, ev, tool_name, n=70, pnl_func=None):
        """Helper: insert n decisions with outcomes."""
        if pnl_func is None:

            def pnl_func(i):
                return 100.0 if i % 2 == 0 else -50.0

        for i in range(n):
            did = ev.record_decision(
                tool_name=tool_name,
                rl_recommendation={"scale": 0.7 if i % 2 == 0 else 0.3},
                crew_action="buy",
            )
            ev.record_outcome(did, pnl=pnl_func(i), slippage_bps=3.0)

    def test_insufficient_observations(self):
        store = _make_store()
        ev = ShadowEvaluator(store)
        # Only 5 decisions — below 63 minimum
        for _i in range(5):
            did = ev.record_decision("rl_position_size", {"scale": 0.5})
            ev.record_outcome(did, pnl=100.0)
        result = ev.evaluate_shadow_period("sizing", min_observations=63)
        assert result.ready_for_promotion is False
        assert result.n_observations == 5

    def test_sufficient_observations_returns_metrics(self):
        store = _make_store()
        ev = ShadowEvaluator(store)
        self._fill_decisions(ev, "rl_position_size", n=70)
        result = ev.evaluate_shadow_period("sizing", min_observations=63)
        assert result.n_observations == 70
        assert result.rl_simulated_sharpe is not None
        assert result.rl_simulated_win_rate is not None
        assert result.rl_simulated_max_drawdown is not None

    def test_get_observation_count(self):
        store = _make_store()
        ev = ShadowEvaluator(store)
        for _ in range(10):
            ev.record_decision("rl_position_size", {"scale": 0.5})
        assert ev.get_observation_count("sizing") == 10

    def test_get_observation_count_zero_for_unknown(self):
        ev = _make_evaluator()
        assert ev.get_observation_count("unknown_type") == 0

    def test_tool_map_routing(self):
        """Decisions for rl_execution_strategy are picked up under agent_type='execution'."""
        store = _make_store()
        ev = ShadowEvaluator(store)
        for _ in range(5):
            ev.record_decision("rl_execution_strategy", {"strategy": "BALANCED"})
        assert ev.get_observation_count("execution") == 5
        assert ev.get_observation_count("sizing") == 0

    def test_evaluation_metrics_are_finite(self):
        store = _make_store()
        ev = ShadowEvaluator(store)
        np.random.seed(42)
        for _i in range(70):
            did = ev.record_decision(
                "rl_position_size", {"scale": float(np.random.uniform(0.3, 0.9))}
            )
            ev.record_outcome(did, pnl=float(np.random.randn() * 200), slippage_bps=3.0)
        result = ev.evaluate_shadow_period("sizing", min_observations=63)
        if result.rl_simulated_sharpe is not None:
            assert np.isfinite(result.rl_simulated_sharpe)
        if result.rl_simulated_max_drawdown is not None:
            assert 0.0 <= result.rl_simulated_max_drawdown
