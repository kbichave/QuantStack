# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.strategy.rules module."""

import numpy as np
import pandas as pd
import pytest

from quantcore.config.timeframes import Timeframe
from quantcore.strategy.rules import MeanReversionRules, EntrySignal


class TestEntrySignal:
    """Test EntrySignal dataclass."""

    def test_entry_signal_creation(self):
        """Test creating an entry signal."""
        signal = EntrySignal(
            triggered=True,
            direction="LONG",
            zscore=-2.5,
            reversion_confirmed=True,
            stretch_magnitude=-2.8,
            price=100.0,
            atr=1.5,
        )

        assert signal.triggered is True
        assert signal.direction == "LONG"
        assert signal.zscore == -2.5
        assert signal.reversion_confirmed is True
        assert signal.stretch_magnitude == -2.8
        assert signal.price == 100.0
        assert signal.atr == 1.5

    def test_strength_property(self):
        """Test signal strength calculation."""
        # Small stretch - low strength
        signal = EntrySignal(
            triggered=True,
            direction="LONG",
            zscore=-1.5,
            reversion_confirmed=True,
            stretch_magnitude=-1.5,
            price=100.0,
            atr=1.5,
        )
        assert signal.strength == pytest.approx(0.5, rel=0.1)

        # Large stretch - capped at 1.0
        signal = EntrySignal(
            triggered=True,
            direction="LONG",
            zscore=-4.0,
            reversion_confirmed=True,
            stretch_magnitude=-4.0,
            price=100.0,
            atr=1.5,
        )
        assert signal.strength == 1.0


class TestMeanReversionRules:
    """Test MeanReversionRules class."""

    @pytest.fixture
    def rules(self) -> MeanReversionRules:
        """Create default mean reversion rules."""
        return MeanReversionRules(Timeframe.D1)

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with features."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(50) * 0.5)

        df = pd.DataFrame(
            {
                "open": close + np.random.randn(50) * 0.2,
                "high": close + np.abs(np.random.randn(50) * 0.5),
                "low": close - np.abs(np.random.randn(50) * 0.5),
                "close": close,
                "volume": np.random.randint(10000, 100000, 50),
                "zscore_price": np.random.randn(50) * 1.5,  # Random z-scores
                "atr": np.ones(50) * 1.5,
            },
            index=dates,
        )
        return df

    def test_init_default_params(self, rules):
        """Test initialization with default params."""
        assert rules.timeframe == Timeframe.D1
        assert rules.zscore_threshold > 0
        assert rules.reversion_delta == 0.2
        assert rules.require_price_confirmation is True

    def test_init_custom_params(self):
        """Test initialization with custom params."""
        rules = MeanReversionRules(
            Timeframe.H1,
            zscore_threshold=1.5,
            reversion_delta=0.3,
            require_price_confirmation=False,
        )

        assert rules.timeframe == Timeframe.H1
        assert rules.zscore_threshold == 1.5
        assert rules.reversion_delta == 0.3
        assert rules.require_price_confirmation is False

    def test_check_entry_insufficient_data(self, rules):
        """Test check_entry with insufficient data."""
        df = pd.DataFrame({"close": [100]})
        signal = rules.check_entry(df)

        assert not signal.triggered
        assert signal.direction == "NONE"

    def test_check_entry_missing_zscore(self, rules):
        """Test check_entry with missing zscore_price column."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000] * 5,
            },
            index=dates,
        )

        signal = rules.check_entry(df)
        assert not signal.triggered

    def test_check_entry_long_signal(self, rules):
        """Test long entry signal generation."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")

        # Create oversold setup with reversion
        df = pd.DataFrame(
            {
                "open": [100, 99, 98, 97, 98],
                "high": [101, 100, 99, 98, 99],
                "low": [99, 98, 97, 96, 97],
                "close": [100, 99, 98, 97, 98.5],  # Uptick on last bar
                "volume": [1000] * 5,
                "zscore_price": [0, -1, -2, -2.5, -2.0],  # Oversold then reverting
                "atr": [1.0] * 5,
            },
            index=dates,
        )

        # Use lower threshold to make test work
        rules = MeanReversionRules(Timeframe.D1, zscore_threshold=2.0)
        signal = rules.check_entry(df)

        assert signal.triggered
        assert signal.direction == "LONG"
        assert signal.reversion_confirmed

    def test_check_entry_short_signal(self, rules):
        """Test short entry signal generation."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")

        # Create overbought setup with reversion
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 102],
                "high": [101, 102, 103, 104, 103],
                "low": [99, 100, 101, 102, 101],
                "close": [100, 101, 102, 103, 101.5],  # Downtick on last bar
                "volume": [1000] * 5,
                "zscore_price": [0, 1, 2, 2.5, 2.0],  # Overbought then reverting
                "atr": [1.0] * 5,
            },
            index=dates,
        )

        rules = MeanReversionRules(Timeframe.D1, zscore_threshold=2.0)
        signal = rules.check_entry(df)

        assert signal.triggered
        assert signal.direction == "SHORT"

    def test_check_entry_no_signal(self, rules):
        """Test no signal when conditions not met."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")

        # Z-score in neutral zone
        df = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100],
                "high": [101, 101, 101, 101, 101],
                "low": [99, 99, 99, 99, 99],
                "close": [100, 100, 100, 100, 100],
                "volume": [1000] * 5,
                "zscore_price": [0, 0.5, -0.5, 0.3, -0.2],  # Neutral
                "atr": [1.0] * 5,
            },
            index=dates,
        )

        signal = rules.check_entry(df)
        assert not signal.triggered
        assert signal.direction == "NONE"


class TestExitRules:
    """Test exit rules."""

    @pytest.fixture
    def rules(self) -> MeanReversionRules:
        return MeanReversionRules(Timeframe.D1)

    def test_check_exit_long_take_profit(self, rules):
        """Test long take profit exit."""
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")

        df = pd.DataFrame(
            {
                "open": [100, 102, 105],
                "high": [101, 104, 108],  # High reaches TP
                "low": [99, 101, 104],
                "close": [100, 103, 107],
                "volume": [1000] * 3,
                "zscore_price": [-2, -1, 0],
            },
            index=dates,
        )

        should_exit, reason, price = rules.check_exit(
            df,
            entry_direction="LONG",
            entry_price=100,
            tp_price=106,  # Will be hit by high of 108
            sl_price=95,
        )

        assert should_exit
        assert reason == "TP"
        assert price == 106

    def test_check_exit_long_stop_loss(self, rules):
        """Test long stop loss exit."""
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")

        df = pd.DataFrame(
            {
                "open": [100, 98, 94],
                "high": [101, 99, 95],
                "low": [99, 93, 92],  # Low hits SL
                "close": [100, 94, 93],
                "volume": [1000] * 3,
                "zscore_price": [0, -2, -3],
            },
            index=dates,
        )

        should_exit, reason, price = rules.check_exit(
            df,
            entry_direction="LONG",
            entry_price=100,
            tp_price=110,
            sl_price=94,  # Will be hit by low of 93
        )

        assert should_exit
        assert reason == "SL"
        assert price == 94

    def test_check_exit_short_take_profit(self, rules):
        """Test short take profit exit."""
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")

        df = pd.DataFrame(
            {
                "open": [100, 98, 95],
                "high": [101, 99, 96],
                "low": [99, 94, 92],  # Low reaches TP
                "close": [100, 96, 93],
                "volume": [1000] * 3,
                "zscore_price": [2, 1, 0],
            },
            index=dates,
        )

        should_exit, reason, price = rules.check_exit(
            df,
            entry_direction="SHORT",
            entry_price=100,
            tp_price=94,  # Will be hit by low of 92
            sl_price=105,
        )

        assert should_exit
        assert reason == "TP"
        assert price == 94

    def test_check_exit_no_exit(self, rules):
        """Test no exit when price within bounds."""
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")

        df = pd.DataFrame(
            {
                "open": [100, 100, 100],
                "high": [101, 101, 101],
                "low": [99, 99, 99],
                "close": [100, 100, 100],
                "volume": [1000] * 3,
                "zscore_price": [-2, -1.8, -1.5],  # Not yet at exit threshold
            },
            index=dates,
        )

        should_exit, reason, price = rules.check_exit(
            df,
            entry_direction="LONG",
            entry_price=100,
            tp_price=110,
            sl_price=90,
        )

        assert not should_exit


class TestScanForSignals:
    """Test signal scanning."""

    def test_scan_for_signals(self):
        """Test scanning DataFrame for signals."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(20) * 0.5)

        df = pd.DataFrame(
            {
                "open": close + np.random.randn(20) * 0.2,
                "high": close + np.abs(np.random.randn(20) * 0.5),
                "low": close - np.abs(np.random.randn(20) * 0.5),
                "close": close,
                "volume": np.random.randint(10000, 100000, 20),
                "zscore_price": np.random.randn(20) * 2,
                "atr": np.ones(20) * 1.5,
            },
            index=dates,
        )

        rules = MeanReversionRules(Timeframe.D1)
        result = rules.scan_for_signals(df)

        assert "mr_signal" in result.columns
        assert "mr_direction" in result.columns
        assert "mr_strength" in result.columns
        assert len(result) == len(df)
