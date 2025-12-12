# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for knowledge store."""

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
    MarketObservation,
    WaveScenario,
    WavePosition,
    RegimeState,
    RegimeType,
    VolatilityRegime,
    TradingSignal,
    TradeLeg,
)


@pytest.fixture
def store():
    """Create a temporary knowledge store for testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    store = KnowledgeStore(db_path=db_path)
    yield store
    store.close()

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


class TestTradeOperations:
    """Test trade journal operations."""

    def test_save_and_get_trade(self, store):
        """Test saving and retrieving a trade."""
        trade = TradeRecord(
            symbol="SPY",
            direction=TradeDirection.LONG,
            structure_type=StructureType.CALL_SPREAD,
            status=TradeStatus.OPEN,
            entry_price=5.00,
            quantity=1,
            confidence_score=0.75,
            legs=[
                TradeLeg(symbol="SPY", action="BUY_TO_OPEN", quantity=1),
            ],
        )

        trade_id = store.save_trade(trade)

        assert trade_id is not None
        assert trade_id > 0

        # Retrieve
        retrieved = store.get_trade(trade_id)

        assert retrieved is not None
        assert retrieved.symbol == "SPY"
        assert retrieved.direction == TradeDirection.LONG
        assert retrieved.entry_price == 5.00

    def test_get_trades_with_filters(self, store):
        """Test querying trades with filters."""
        # Create multiple trades
        for i in range(5):
            trade = TradeRecord(
                symbol="SPY" if i < 3 else "QQQ",
                direction=TradeDirection.LONG,
                structure_type=StructureType.CALL_SPREAD,
                status=TradeStatus.OPEN if i < 2 else TradeStatus.CLOSED,
                pnl=100.0 if i >= 2 else None,
            )
            store.save_trade(trade)

        # Filter by symbol
        spy_trades = store.get_trades(symbol="SPY")
        assert len(spy_trades) == 3

        # Filter by status
        open_trades = store.get_trades(status=TradeStatus.OPEN)
        assert len(open_trades) == 2

    def test_get_open_trades(self, store):
        """Test getting open trades."""
        # Create trades
        for status in [TradeStatus.OPEN, TradeStatus.CLOSED, TradeStatus.OPEN]:
            trade = TradeRecord(
                symbol="SPY",
                direction=TradeDirection.LONG,
                structure_type=StructureType.CALL_SPREAD,
                status=status,
            )
            store.save_trade(trade)

        open_trades = store.get_open_trades()
        assert len(open_trades) == 2


class TestObservationOperations:
    """Test market observation operations."""

    def test_save_and_get_observation(self, store):
        """Test saving and retrieving observations."""
        obs = MarketObservation(
            symbol="SPY",
            observation_type="PRICE_ALERT",
            current_price=450.00,
            alert_message="SPY up 1.5%",
            severity="WARNING",
        )

        obs_id = store.save_observation(obs)

        assert obs_id is not None

        # Retrieve recent
        recent = store.get_recent_observations(symbol="SPY", hours=1)

        assert len(recent) == 1
        assert recent[0].symbol == "SPY"
        assert recent[0].alert_message == "SPY up 1.5%"

    def test_mark_observations_processed(self, store):
        """Test marking observations as processed."""
        # Create observations
        obs_ids = []
        for i in range(3):
            obs = MarketObservation(
                symbol="SPY",
                observation_type="PRICE_ALERT",
                current_price=450.00 + i,
                alert_message=f"Alert {i}",
            )
            obs_ids.append(store.save_observation(obs))

        # Mark first two as processed
        store.mark_observations_processed(obs_ids[:2])

        # Get unprocessed
        unprocessed = store.get_recent_observations(unprocessed_only=True)
        assert len(unprocessed) == 1


class TestWaveScenarioOperations:
    """Test wave scenario operations."""

    def test_save_and_get_wave_scenario(self, store):
        """Test saving and retrieving wave scenarios."""
        scenario = WaveScenario(
            symbol="SPY",
            timeframe="daily",
            wave_position=WavePosition.WAVE_3,
            wave_degree="Primary",
            confidence=0.75,
            invalidation_level=440.00,
            scenario_type="BULLISH",
            description="Wave 3 extension in progress",
            primary_target=480.00,
        )

        scenario_id = store.save_wave_scenario(scenario)

        assert scenario_id is not None

        # Retrieve active
        active = store.get_active_wave_scenarios(symbol="SPY")

        assert len(active) == 1
        assert active[0].wave_position == WavePosition.WAVE_3
        assert active[0].primary_target == 480.00

    def test_invalidate_wave_scenario(self, store):
        """Test invalidating a wave scenario."""
        scenario = WaveScenario(
            symbol="SPY",
            timeframe="daily",
            wave_position=WavePosition.WAVE_5,
            wave_degree="Primary",
            confidence=0.6,
            invalidation_level=445.00,
            scenario_type="BULLISH",
            description="Final wave",
        )

        scenario_id = store.save_wave_scenario(scenario)

        # Invalidate
        store.invalidate_wave_scenario(scenario_id)

        # Should not appear in active
        active = store.get_active_wave_scenarios(symbol="SPY")
        assert len(active) == 0


class TestRegimeStateOperations:
    """Test regime state operations."""

    def test_save_and_get_regime(self, store):
        """Test saving and retrieving regime state."""
        state = RegimeState(
            symbol="SPY",
            timeframe="daily",
            trend_regime=RegimeType.TRENDING_UP,
            volatility_regime=VolatilityRegime.NORMAL,
            atr=5.5,
            atr_percentile=55.0,
            adx=28.0,
            confidence=0.8,
        )

        state_id = store.save_regime_state(state)

        assert state_id is not None

        # Get current
        current = store.get_current_regime("SPY")

        assert current is not None
        assert current.trend_regime == RegimeType.TRENDING_UP
        assert current.volatility_regime == VolatilityRegime.NORMAL


class TestSignalOperations:
    """Test trading signal operations."""

    def test_save_and_get_signal(self, store):
        """Test saving and retrieving signals."""
        signal = TradingSignal(
            symbol="SPY",
            direction=TradeDirection.LONG,
            signal_type="WAVE_TARGET",
            strength=0.8,
            confidence=0.7,
            entry_price=450.00,
            target_price=470.00,
            stop_loss=440.00,
            rationale="Wave 3 target",
            source_agent="wave_analyst",
        )

        signal_id = store.save_signal(signal)

        assert signal_id is not None

        # Get active
        active = store.get_active_signals(symbol="SPY")

        assert len(active) == 1
        assert active[0].target_price == 470.00
