# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for learning engine modules."""

import pytest
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from quant_pod.knowledge.store import KnowledgeStore
from quant_pod.knowledge.models import (
    TradeRecord,
    TradeDirection,
    StructureType,
    TradeStatus,
)
from quant_pod.learning.skill_tracker import SkillTracker, AgentSkill
from quant_pod.learning.structure_stats import StructureStats
from quant_pod.learning.expectancy_engine import ExpectancyEngine


@pytest.fixture
def store():
    """Create a temporary knowledge store for testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    store = KnowledgeStore(db_path=db_path)
    yield store
    store.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def store_with_trades(store):
    """Create store with sample trades."""
    # Create mix of winning and losing trades
    trades = [
        # Winning trades
        TradeRecord(
            symbol="SPY",
            direction=TradeDirection.LONG,
            structure_type=StructureType.CALL_SPREAD,
            status=TradeStatus.CLOSED,
            pnl=150.0,
            confidence_score=0.7,
            regime_at_entry="TRENDING_UP",
        ),
        TradeRecord(
            symbol="SPY",
            direction=TradeDirection.LONG,
            structure_type=StructureType.CALL_SPREAD,
            status=TradeStatus.CLOSED,
            pnl=200.0,
            confidence_score=0.8,
            regime_at_entry="TRENDING_UP",
        ),
        TradeRecord(
            symbol="QQQ",
            direction=TradeDirection.SHORT,
            structure_type=StructureType.PUT_SPREAD,
            status=TradeStatus.CLOSED,
            pnl=100.0,
            confidence_score=0.6,
            regime_at_entry="TRENDING_DOWN",
        ),
        # Losing trades
        TradeRecord(
            symbol="SPY",
            direction=TradeDirection.LONG,
            structure_type=StructureType.CALL_SPREAD,
            status=TradeStatus.CLOSED,
            pnl=-100.0,
            confidence_score=0.5,
            regime_at_entry="RANGING",
        ),
        TradeRecord(
            symbol="IWM",
            direction=TradeDirection.SHORT,
            structure_type=StructureType.IRON_CONDOR,
            status=TradeStatus.CLOSED,
            pnl=-50.0,
            confidence_score=0.4,
            regime_at_entry="VOLATILE",
        ),
    ]

    for trade in trades:
        store.save_trade(trade)

    return store


class TestSkillTracker:
    """Test skill tracker functionality."""

    def test_update_agent_skill_prediction(self, store):
        """Test updating agent skill with predictions."""
        tracker = SkillTracker(store)

        # Update with correct prediction
        skill = tracker.update_agent_skill("wave_analyst", prediction_correct=True)
        assert skill.prediction_count == 1
        assert skill.prediction_accuracy > 0.5

        # Update with incorrect prediction
        skill = tracker.update_agent_skill("wave_analyst", prediction_correct=False)
        assert skill.prediction_count == 2

    def test_update_agent_skill_signal(self, store):
        """Test updating agent skill with signal P&L."""
        tracker = SkillTracker(store)

        # Update with winning signal
        skill = tracker.update_agent_skill("trade_builder", signal_pnl=100.0)
        assert skill.signal_count == 1
        assert skill.signal_win_rate > 0.5
        assert skill.avg_signal_pnl == 100.0

        # Update with losing signal
        skill = tracker.update_agent_skill("trade_builder", signal_pnl=-50.0)
        assert skill.signal_count == 2
        assert skill.avg_signal_pnl == 25.0  # (100 - 50) / 2

    def test_get_confidence_adjustment(self, store):
        """Test confidence adjustment calculation."""
        tracker = SkillTracker(store)

        # New agent with no data
        adj = tracker.get_confidence_adjustment("new_agent")
        assert adj == 1.0  # No adjustment for new agents

        # Agent with history
        for i in range(15):
            tracker.update_agent_skill("experienced_agent", prediction_correct=True)

        adj = tracker.get_confidence_adjustment("experienced_agent")
        assert adj > 1.0  # High skill should boost confidence


class TestStructureStats:
    """Test structure statistics functionality."""

    def test_record_trade_outcome(self, store_with_trades):
        """Test recording trade outcomes."""
        stats = StructureStats(store_with_trades)

        # Check call spread stats
        call_spread_stats = stats.get_structure_stats(StructureType.CALL_SPREAD)

        assert call_spread_stats.total_trades == 3
        assert call_spread_stats.winning_trades == 2
        assert call_spread_stats.losing_trades == 1

    def test_get_best_structures(self, store_with_trades):
        """Test getting best performing structures."""
        stats = StructureStats(store_with_trades)

        best = stats.get_best_structures(min_trades=1)

        assert len(best) > 0
        # Call spread should be near top with positive expectancy

    def test_structure_recommendation(self, store_with_trades):
        """Test structure recommendation."""
        stats = StructureStats(store_with_trades)

        # Get recommendation for bullish direction
        rec = stats.get_structure_recommendation(direction="LONG")

        assert rec is not None
        assert rec in [
            StructureType.CALL_SPREAD,
            StructureType.LONG_CALL,
            StructureType.PUT_SPREAD,
            StructureType.DIAGONAL,
        ]


class TestExpectancyEngine:
    """Test expectancy calculations."""

    def test_calculate_expectancy(self, store_with_trades):
        """Test expectancy calculation."""
        engine = ExpectancyEngine(store_with_trades)

        result = engine.calculate_expectancy()

        # Should have positive expectancy overall (3 wins, 2 losses)
        assert result.sample_size == 5
        assert result.win_rate == 0.6
        assert result.expectancy > 0  # More wins than losses with higher avg win

    def test_calculate_expectancy_by_structure(self, store_with_trades):
        """Test expectancy by structure type."""
        engine = ExpectancyEngine(store_with_trades)

        result = engine.calculate_expectancy(structure=StructureType.CALL_SPREAD)

        assert result.sample_size == 3
        # Call spread has 2 wins, 1 loss

    def test_kelly_fraction(self, store_with_trades):
        """Test Kelly criterion calculation."""
        engine = ExpectancyEngine(store_with_trades)

        kelly = engine.get_kelly_fraction(kelly_mode="half")

        # With positive expectancy, should recommend some fraction
        assert kelly >= 0
        assert kelly <= 1

    def test_trade_quality_score(self, store_with_trades):
        """Test trade quality scoring."""
        engine = ExpectancyEngine(store_with_trades)

        # Good trade: high win rate, good R/R
        score = engine.get_trade_quality_score(
            expected_win_rate=0.7,
            expected_risk_reward=2.0,
        )

        assert score["quality_score"] > 50
        assert score["expected_ev"] > 0

        # Bad trade: low win rate, poor R/R
        score = engine.get_trade_quality_score(
            expected_win_rate=0.3,
            expected_risk_reward=0.5,
        )

        assert score["quality_score"] < 50
        assert score["recommendation"] in ["AVOID", "STRONG_AVOID"]
