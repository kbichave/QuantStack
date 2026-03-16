# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for QuantAgent feature computation used by MCP tools."""

import numpy as np
import pandas as pd


class TestQuantAgentFeatures:
    """Tests for QuantAgent feature computation used by MCP tools."""

    def test_pattern_features_computation(self):
        """Test pattern feature computation."""
        from quantcore.config.timeframes import Timeframe
        from quantcore.features.quantagents_pattern import QuantAgentsPatternFeatures

        # Create test data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 100),
                "high": np.linspace(101, 112, 100),
                "low": np.linspace(99, 108, 100),
                "close": np.linspace(100, 110, 100),
                "volume": np.random.randint(1000000, 5000000, 100),
            },
            index=dates,
        )

        pattern_calc = QuantAgentsPatternFeatures(Timeframe.D1)
        result = pattern_calc.compute(df)

        assert "qa_pattern_is_pullback" in result.columns
        assert "qa_pattern_is_breakout" in result.columns
        assert "qa_pattern_consolidation" in result.columns

    def test_trend_features_computation(self):
        """Test trend feature computation."""
        from quantcore.config.timeframes import Timeframe
        from quantcore.features.quantagents_trend import QuantAgentsTrendFeatures

        # Create test data (need more bars for trend calculation)
        dates = pd.date_range("2024-01-01", periods=150, freq="D")
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 150, 150),
                "high": np.linspace(101, 152, 150),
                "low": np.linspace(99, 148, 150),
                "close": np.linspace(100, 150, 150),
                "volume": np.random.randint(1000000, 5000000, 150),
            },
            index=dates,
        )

        trend_calc = QuantAgentsTrendFeatures(Timeframe.D1)
        result = trend_calc.compute(df)

        assert "qa_trend_slope_short" in result.columns
        assert "qa_trend_regime" in result.columns
        assert "qa_trend_quality_med" in result.columns
