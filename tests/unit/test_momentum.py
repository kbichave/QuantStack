# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.features.momentum module."""

import numpy as np
import pandas as pd
import pytest

from quantcore.config.timeframes import Timeframe
from quantcore.features.momentum import MomentumFeatures


class TestMomentumFeatures:
    """Test MomentumFeatures class."""

    @pytest.fixture
    def features(self) -> MomentumFeatures:
        """Create momentum features instance."""
        return MomentumFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        """Create sample OHLCV data with trends."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Create trending price data
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.abs(np.random.randn(100) * 0.5)
        low = close - np.abs(np.random.randn(100) * 0.5)
        open_ = close + np.random.randn(100) * 0.2

        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(10000, 100000, 100),
            },
            index=dates,
        )

    def test_compute_returns_dataframe(self, features, ohlcv):
        """Test compute returns DataFrame."""
        result = features.compute(ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_compute_adds_all_features(self, features, ohlcv):
        """Test all expected features are added."""
        result = features.compute(ohlcv)
        feature_names = features.get_feature_names()

        for name in feature_names:
            assert name in result.columns, f"Missing feature: {name}"

    def test_feature_count(self, features):
        """Test correct number of features."""
        names = features.get_feature_names()
        assert len(names) >= 15  # At least 15 momentum features


class TestRSI:
    """Test RSI calculation."""

    @pytest.fixture
    def features(self) -> MomentumFeatures:
        return MomentumFeatures(Timeframe.D1)

    def test_rsi_range(self, features):
        """Test RSI is in 0-100 range."""
        # Create data with clear up trend
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close = np.linspace(100, 150, 50) + np.random.randn(50) * 0.1
        high = close + 0.5
        low = close - 0.5

        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        result = features.compute(df)
        rsi = result["rsi"].dropna()

        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_rsi_uptrend(self, features):
        """Test RSI is computed for uptrending data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        # Strong uptrend with some noise to avoid constant gains
        np.random.seed(42)
        # Use cumulative returns to create realistic price movements
        returns = 0.01 + np.random.randn(100) * 0.02  # Positive drift
        close = 100 * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.randn(100) * 0.005))
        low = close * (1 - np.abs(np.random.randn(100) * 0.005))

        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": [1000] * 100,
            },
            index=dates,
        )

        result = features.compute(df)

        # Check RSI is computed and in valid range
        assert "rsi" in result.columns
        rsi = result["rsi"].dropna()
        assert len(rsi) > 0, "RSI should have non-NaN values"
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI should be in 0-100 range"

    def test_rsi_downtrend(self, features):
        """Test RSI is low in downtrend."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        # Strong downtrend
        close = np.linspace(200, 100, 50)
        high = close + 0.5
        low = close - 0.5

        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        result = features.compute(df)
        # RSI should be low at end of strong downtrend
        assert result["rsi"].iloc[-1] < 30


class TestStochastic:
    """Test Stochastic Oscillator calculation."""

    @pytest.fixture
    def features(self) -> MomentumFeatures:
        return MomentumFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(np.random.randn(50)),
                "low": close - np.abs(np.random.randn(50)),
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

    def test_stoch_k_range(self, features, ohlcv):
        """Test Stochastic %K is in 0-100 range."""
        result = features.compute(ohlcv)
        stoch_k = result["stoch_k"].dropna()

        assert (stoch_k >= 0).all()
        assert (stoch_k <= 100).all()

    def test_stoch_d_range(self, features, ohlcv):
        """Test Stochastic %D is in 0-100 range."""
        result = features.compute(ohlcv)
        stoch_d = result["stoch_d"].dropna()

        assert (stoch_d >= 0).all()
        assert (stoch_d <= 100).all()

    def test_stoch_cross_values(self, features, ohlcv):
        """Test stoch_cross has valid values."""
        result = features.compute(ohlcv)
        stoch_cross = result["stoch_cross"].dropna()

        # Should only be -1, 0, or 1
        assert set(stoch_cross.unique()).issubset({-1, 0, 1})


class TestMACD:
    """Test MACD calculation."""

    @pytest.fixture
    def features(self) -> MomentumFeatures:
        return MomentumFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

    def test_macd_components_present(self, features, ohlcv):
        """Test MACD components are computed."""
        result = features.compute(ohlcv)

        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_macd_histogram_calculation(self, features, ohlcv):
        """Test MACD histogram is difference of line and signal."""
        result = features.compute(ohlcv)

        expected_histogram = result["macd_line"] - result["macd_signal"]
        pd.testing.assert_series_equal(
            result["macd_histogram"],
            expected_histogram,
            check_names=False,
        )

    def test_macd_cross_values(self, features, ohlcv):
        """Test macd_cross has valid values."""
        result = features.compute(ohlcv)
        macd_cross = result["macd_cross"].dropna()

        # Should only be -1, 0, or 1
        assert set(macd_cross.unique()).issubset({-1, 0, 1})


class TestROC:
    """Test Rate of Change calculation."""

    @pytest.fixture
    def features(self) -> MomentumFeatures:
        return MomentumFeatures(Timeframe.D1)

    def test_roc_calculation(self, features):
        """Test ROC calculation is correct."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        close = pd.Series(np.linspace(100, 110, 20), index=dates)

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000] * 20,
            },
            index=dates,
        )

        result = features.compute(df)

        # ROC should be positive for uptrend
        roc = result["roc"].dropna()
        assert (roc > 0).all()


class TestWilliamsR:
    """Test Williams %R calculation."""

    @pytest.fixture
    def features(self) -> MomentumFeatures:
        return MomentumFeatures(Timeframe.D1)

    def test_williams_r_range(self, features):
        """Test Williams %R is in -100 to 0 range."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(np.random.randn(50)),
                "low": close - np.abs(np.random.randn(50)),
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        result = features.compute(df)
        williams_r = result["williams_r"].dropna()

        assert (williams_r >= -100).all()
        assert (williams_r <= 0).all()


class TestMomentumScore:
    """Test combined momentum score."""

    @pytest.fixture
    def features(self) -> MomentumFeatures:
        return MomentumFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(np.random.randn(100)),
                "low": close - np.abs(np.random.randn(100)),
                "close": close,
                "volume": [1000] * 100,
            },
            index=dates,
        )

    def test_momentum_score_range(self, features, ohlcv):
        """Test momentum score is in -100 to 100 range."""
        result = features.compute(ohlcv)
        score = result["momentum_score"].dropna()

        assert (score >= -100).all()
        assert (score <= 100).all()


class TestRSIZones:
    """Test RSI zone indicators."""

    @pytest.fixture
    def features(self) -> MomentumFeatures:
        return MomentumFeatures(Timeframe.D1)

    def test_rsi_oversold_indicator(self, features):
        """Test RSI oversold indicator."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        # Strong downtrend to get oversold
        close = np.linspace(200, 100, 50)

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        result = features.compute(df)

        # Should have some oversold signals
        assert result["rsi_oversold"].sum() > 0
        # Oversold should be binary
        assert set(result["rsi_oversold"].unique()).issubset({0, 1})

    def test_rsi_overbought_indicator(self, features):
        """Test RSI overbought indicator is binary."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        # Strong uptrend with noise
        np.random.seed(42)
        base = np.linspace(100, 200, 50)
        close = base + np.random.randn(50) * 0.1

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        result = features.compute(df)

        # Overbought should be binary (0 or 1)
        assert set(result["rsi_overbought"].unique()).issubset({0, 1})
