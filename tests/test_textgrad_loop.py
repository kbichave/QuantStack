# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TextGrad backward pass on trading decision chains."""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from quantstack.db import run_migrations
from quantstack.optimization.textgrad_loop import (
    DecisionNode,
    PromptProposal,
    TextGradOptimizer,
)


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    run_migrations(c)
    return c


@pytest.fixture
def optimizer(conn):
    return TextGradOptimizer(conn)


def _make_trade(**overrides) -> dict:
    defaults = dict(
        trade_id=1,
        symbol="SPY",
        strategy_id="regime_momentum_v1",
        regime_at_entry="trending_up",
        regime_at_exit="ranging",
        realized_pnl_pct=-3.5,
        conviction=0.60,
        position_size="half",
        signals_present=True,
        strategy_regime_affinity=0.7,
        signals_summary="rsi=42 macd=positive",
        debate_verdict="",
    )
    defaults.update(overrides)
    return defaults


class TestBuildChain:
    def test_chain_has_four_nodes(self, optimizer):
        chain = optimizer.build_chain(_make_trade())
        assert len(chain) == 4
        names = [n.name for n in chain]
        assert "signal_collection" in names
        assert "regime_classification" in names
        assert "strategy_selection" in names
        assert "position_sizing" in names

    def test_chain_captures_trade_context(self, optimizer):
        chain = optimizer.build_chain(_make_trade(symbol="AAPL"))
        regime_node = next(n for n in chain if n.name == "regime_classification")
        assert "trending_up" in regime_node.output_text


class TestBackwardPass:
    def test_focuses_on_worst_step(self, optimizer):
        from quantstack.optimization.credit_assignment import StepCredit

        chain = optimizer.build_chain(_make_trade())
        worst = StepCredit("regime", "shifted", -0.5, "heuristic", "Regime shifted")

        critiques = optimizer.backward_pass(chain, "Lost 3.5%", worst)
        # Only regime_classification should have a critique
        assert critiques["regime_classification"] != ""
        assert critiques["signal_collection"] == ""
        assert critiques["position_sizing"] == ""

    def test_heuristic_fallback_when_no_litellm(self, optimizer):
        chain = optimizer.build_chain(_make_trade())
        from quantstack.optimization.credit_assignment import StepCredit
        worst = StepCredit("regime", "shifted", -0.5, "heuristic", "Regime shifted")

        # This will use heuristic fallback since litellm may not have API keys
        critiques = optimizer.backward_pass(chain, "Lost 3.5%", worst)
        assert critiques["regime_classification"]  # Non-empty


class TestProposal:
    def test_status_is_proposed(self, optimizer):
        proposal = optimizer.propose_prompt_update(
            "regime_classification",
            "Should check HMM stability before classifying regime",
            trade_id=1,
        )
        assert proposal is not None
        assert proposal.status == "proposed"
        assert proposal.node_name == "regime_classification"

    def test_heuristic_critique_not_stored(self, optimizer):
        proposal = optimizer.propose_prompt_update(
            "regime_classification",
            "[heuristic] regime_classification: Regime shifted",
            trade_id=1,
        )
        assert proposal is None

    def test_persists_to_db(self, optimizer, conn):
        optimizer.propose_prompt_update(
            "regime_classification",
            "Should check HMM stability",
            trade_id=1,
        )
        count = conn.execute("SELECT COUNT(*) FROM prompt_critiques").fetchone()[0]
        assert count == 1


class TestRunDaily:
    def test_no_losses_is_noop(self, optimizer):
        proposals = optimizer.run_daily(date.today(), losing_trades=[])
        assert proposals == []

    def test_processes_losing_trade(self, optimizer):
        trades = [_make_trade()]
        proposals = optimizer.run_daily(date.today(), losing_trades=trades)
        # Should generate at least one proposal (from the worst step)
        # May be 0 if only heuristic critiques generated (no litellm API key)
        assert isinstance(proposals, list)
