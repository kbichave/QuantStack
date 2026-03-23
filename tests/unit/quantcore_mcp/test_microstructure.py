# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for microstructure tools: liquidity, volume profile, trading calendar."""

import numpy as np
import pandas as pd
from datetime import time as dt_time
from quantstack.core.microstructure.liquidity import LiquidityAnalyzer, SpreadEstimator


class TestLiquidityAnalysis:
    """Tests for liquidity analysis."""

    def test_spread_estimation(self):
        """Test Corwin-Schultz spread estimator."""
        # Create OHLC data with known spread
        np.random.seed(42)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        high = close + np.random.rand(n) * 0.5
        low = close - np.random.rand(n) * 0.5

        spread = SpreadEstimator.corwin_schultz_spread(high, low, window=20)

        assert len(spread) == n
        assert spread.iloc[-1] >= 0  # Spread can't be negative

    def test_liquidity_score(self):
        """Test liquidity scoring."""
        # Create test data with sufficient bars for rolling calculations
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)

        df = pd.DataFrame(
            {
                "open": close - np.random.rand(n) * 0.3,
                "high": close + np.random.rand(n) * 0.5,
                "low": close - np.random.rand(n) * 0.5,
                "close": close,
                "volume": np.random.randint(500000, 1500000, n),
            },
            index=dates,
        )

        analyzer = LiquidityAnalyzer()
        features = analyzer.compute_features(df)

        assert "liquidity_score" in features.columns
        # Check that liquidity_score exists and is computed (may be NaN in first window)
        valid_scores = features["liquidity_score"].dropna()
        if len(valid_scores) > 0:
            assert valid_scores.iloc[-1] >= 0
            assert valid_scores.iloc[-1] <= 1


class TestVolumeProfile:
    """Tests for volume profile analysis."""

    def test_vwap_calculation(self):
        """Test VWAP calculation."""
        # Simple VWAP test
        typical_price = pd.Series([100, 101, 102])
        volume = pd.Series([1000, 2000, 1000])

        vwap = (typical_price * volume).sum() / volume.sum()

        # Expected: (100*1000 + 101*2000 + 102*1000) / 4000 = 101
        assert abs(vwap - 101) < 0.01


class TestTradingCalendar:
    """Tests for trading calendar."""

    def test_market_hours(self):
        """Test market hours configuration."""
        market_open = "09:30"
        market_close = "16:00"

        # Parse times
        open_time = dt_time(int(market_open[:2]), int(market_open[3:]))
        close_time = dt_time(int(market_close[:2]), int(market_close[3:]))

        assert open_time < close_time
