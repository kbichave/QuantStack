# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ReflexionMemory — structured episodic memory with root-cause classification."""

from __future__ import annotations

import pytest

from quantstack.autonomous.reflection import TradeReflection
from quantstack.db import pg_conn, run_migrations
from quantstack.optimization.reflexion_memory import (
    MAX_EPISODES,
    ReflexionEpisode,
    ReflexionMemory,
    RootCause,
)


@pytest.fixture
def conn():
    """PostgreSQL connection with all migrations applied."""
    with pg_conn() as c:
        run_migrations(c)
        yield c


@pytest.fixture
def mem(conn):
    return ReflexionMemory(conn)


def _make_reflection(**overrides) -> TradeReflection:
    defaults = dict(
        symbol="SPY",
        strategy_id="regime_momentum_v1",
        action="buy",
        entry_price=450.0,
        exit_price=445.0,
        realized_pnl_pct=-2.5,
        holding_days=3,
        regime_at_entry="trending_up",
        regime_at_exit="trending_up",
        conviction_at_entry=0.65,
        signals_at_entry="rsi=42 macd=positive adx=28",
    )
    defaults.update(overrides)
    return TradeReflection(**defaults)


# ---------------------------------------------------------------------------
# Root cause classification
# ---------------------------------------------------------------------------

class TestClassifyRootCause:
    def test_regime_shift(self):
        ref = _make_reflection(regime_at_entry="trending_up", regime_at_exit="ranging")
        assert ReflexionMemory.classify_root_cause(ref) == RootCause.REGIME_SHIFT

    def test_regime_shift_not_triggered_when_exit_unknown(self):
        ref = _make_reflection(regime_at_entry="trending_up", regime_at_exit="unknown")
        assert ReflexionMemory.classify_root_cause(ref) != RootCause.REGIME_SHIFT

    def test_sizing_error(self):
        ref = _make_reflection(
            realized_pnl_pct=-6.0, conviction_at_entry=0.8,
            regime_at_entry="trending_up", regime_at_exit="trending_up",
        )
        assert ReflexionMemory.classify_root_cause(ref) == RootCause.SIZING_ERROR

    def test_data_gap(self):
        ref = _make_reflection(signals_at_entry="", regime_at_exit="trending_up")
        assert ReflexionMemory.classify_root_cause(ref) == RootCause.DATA_GAP

    def test_strategy_mismatch_momentum_ranging(self):
        ref = _make_reflection(
            strategy_id="regime_momentum_v1",
            regime_at_entry="ranging",
            regime_at_exit="ranging",
        )
        assert ReflexionMemory.classify_root_cause(ref) == RootCause.STRATEGY_MISMATCH

    def test_strategy_mismatch_reversion_trending(self):
        ref = _make_reflection(
            strategy_id="mean_reversion_v1",
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",
        )
        assert ReflexionMemory.classify_root_cause(ref) == RootCause.STRATEGY_MISMATCH

    def test_stop_loss_width(self):
        ref = _make_reflection(
            realized_pnl_pct=-7.0, conviction_at_entry=0.4,
            regime_at_entry="trending_up", regime_at_exit="trending_up",
        )
        assert ReflexionMemory.classify_root_cause(ref) == RootCause.STOP_LOSS_WIDTH

    def test_entry_timing(self):
        ref = _make_reflection(
            realized_pnl_pct=-2.0,
            regime_at_entry="trending_up", regime_at_exit="trending_up",
        )
        assert ReflexionMemory.classify_root_cause(ref) == RootCause.ENTRY_TIMING

    def test_unknown_fallback(self):
        ref = _make_reflection(
            signals_at_entry="",
            regime_at_entry="unknown", regime_at_exit="unknown",
        )
        # No regime shift, no signals → DATA_GAP (not unknown, since empty signals)
        assert ReflexionMemory.classify_root_cause(ref) == RootCause.DATA_GAP


# ---------------------------------------------------------------------------
# Record and retrieve roundtrip
# ---------------------------------------------------------------------------

class TestRecordAndRetrieve:
    def test_roundtrip(self, mem):
        ref = _make_reflection(regime_at_entry="trending_up", regime_at_exit="ranging")
        episode = mem.record_episode(ref)

        assert isinstance(episode, ReflexionEpisode)
        assert episode.root_cause == RootCause.REGIME_SHIFT
        assert episode.symbol == "SPY"
        assert episode.pnl_pct == -2.5
        assert "regime" in episode.verbal_reinforcement.lower()

        # Retrieve
        results = mem.get_relevant("trending_up", "regime_momentum_v1", "SPY")
        assert len(results) >= 1
        assert results[0].episode_id == episode.episode_id

    def test_persists_to_db(self, mem, conn):
        ref = _make_reflection()
        mem.record_episode(ref)

        count = conn.execute("SELECT COUNT(*) FROM reflexion_episodes").fetchone()[0]
        assert count == 1

    def test_multiple_episodes(self, mem):
        for i in range(5):
            ref = _make_reflection(symbol=f"SYM{i}", realized_pnl_pct=-(i + 1.5))
            mem.record_episode(ref)

        results = mem.get_relevant("trending_up")
        assert len(results) == 3  # default k=3


# ---------------------------------------------------------------------------
# Retrieval filtering
# ---------------------------------------------------------------------------

class TestGetRelevant:
    def test_filters_by_regime(self, mem):
        mem.record_episode(_make_reflection(regime_at_entry="trending_up", regime_at_exit="ranging"))
        mem.record_episode(_make_reflection(regime_at_entry="ranging", regime_at_exit="ranging"))

        results = mem.get_relevant("ranging")
        assert all(ep.regime == "ranging" for ep in results)

    def test_empty_when_no_episodes(self, mem):
        assert mem.get_relevant("trending_up") == []

    def test_fallback_when_no_exact_match(self, mem):
        mem.record_episode(_make_reflection(regime_at_entry="trending_up", regime_at_exit="ranging"))
        # No "trending_down" + specific strategy match — falls back to regime-only
        results = mem.get_relevant("trending_up", "nonexistent_strategy", "ZZZZZ")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------

class TestInjectIntoPrompt:
    def test_format(self, mem):
        ref = _make_reflection(regime_at_entry="trending_up", regime_at_exit="ranging")
        episode = mem.record_episode(ref)

        prompt = mem.inject_into_prompt("You are a trader.", [episode])
        assert prompt.startswith("## Lessons from Similar Past Trades")
        assert "You are a trader." in prompt
        assert "regime_shift" in prompt

    def test_empty_episodes_returns_base(self, mem):
        assert mem.inject_into_prompt("base", []) == "base"


# ---------------------------------------------------------------------------
# SQL retrieval relevance
# ---------------------------------------------------------------------------

class TestSQLRetrieval:
    def test_filters_by_strategy_and_symbol(self, mem):
        # Record two episodes: one for SPY momentum, one for XLE vol_compress
        mem.record_episode(_make_reflection(
            symbol="SPY", strategy_id="regime_momentum_v1",
            regime_at_entry="trending_up", regime_at_exit="ranging",
        ))
        mem.record_episode(_make_reflection(
            symbol="XLE", strategy_id="vol_compress_xle_v1",
            regime_at_entry="ranging", regime_at_exit="ranging",
            realized_pnl_pct=-3.0,
        ))

        # Querying for SPY momentum should return only SPY episode
        results = mem.get_relevant("trending_up", "regime_momentum_v1", "SPY")
        assert len(results) == 1
        assert results[0].symbol == "SPY"

    def test_orders_by_severity(self, mem):
        mem.record_episode(_make_reflection(
            symbol="A", realized_pnl_pct=-1.5,
            regime_at_entry="trending_up", regime_at_exit="trending_up",
        ))
        mem.record_episode(_make_reflection(
            symbol="B", realized_pnl_pct=-8.0,
            regime_at_entry="trending_up", regime_at_exit="trending_up",
        ))

        results = mem.get_relevant("trending_up", k=2)
        # Most severe loss first
        assert results[0].symbol == "B"
        assert results[1].symbol == "A"
