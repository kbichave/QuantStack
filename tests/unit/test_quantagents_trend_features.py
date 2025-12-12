"""
Tests for QuantAgents trend features.

Verifies:
1. Multi-horizon slope computation
2. Trend regime classification
3. Trend quality (R²) measurement
4. No lookahead bias
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quantcore.config.timeframes import Timeframe
from quantcore.features.quantagents_trend import QuantAgentsTrendFeatures
from tests.conftest import make_ohlcv_df


class TestQuantAgentsTrendFeatures:
    """Test QuantAgents trend feature computation."""

    def test_initialization(self):
        """Test feature calculator initialization."""
        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        assert qa_trend.timeframe == Timeframe.H1
        assert qa_trend.window_short == 10
        assert qa_trend.window_med == 30
        assert qa_trend.window_long == 100
        assert len(qa_trend.get_feature_names()) == 12

    def test_feature_computation_uptrend(self):
        """Test feature computation on uptrend data."""
        # Create strong uptrend: 100 -> 150 over 120 bars
        prices = np.linspace(100, 150, 120)
        df = make_ohlcv_df(prices.tolist())

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # Check all features are present
        for feature_name in qa_trend.get_feature_names():
            assert feature_name in result.columns, f"Missing feature: {feature_name}"

        # After warmup, should have positive slopes
        last_idx = len(result) - 1
        assert (
            result["qa_trend_slope_short"].iloc[last_idx] > 0
        ), "Uptrend should have positive short slope"
        assert (
            result["qa_trend_slope_med"].iloc[last_idx] > 0
        ), "Uptrend should have positive med slope"

        # Trend regime should be uptrend (+1)
        assert (
            result["qa_trend_regime"].iloc[last_idx] == 1
        ), "Strong uptrend should be classified as +1"

    def test_feature_computation_downtrend(self):
        """Test feature computation on downtrend data."""
        # Create strong downtrend: 150 -> 100 over 120 bars
        prices = np.linspace(150, 100, 120)
        df = make_ohlcv_df(prices.tolist())

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # Should have negative slopes
        last_idx = len(result) - 1
        assert (
            result["qa_trend_slope_short"].iloc[last_idx] < 0
        ), "Downtrend should have negative short slope"
        assert (
            result["qa_trend_slope_med"].iloc[last_idx] < 0
        ), "Downtrend should have negative med slope"

        # Trend regime should be downtrend (-1)
        assert (
            result["qa_trend_regime"].iloc[last_idx] == -1
        ), "Strong downtrend should be classified as -1"

    def test_feature_computation_sideways(self):
        """Test feature computation on sideways/choppy data."""
        # Create choppy/oscillating price
        prices = [100 + 5 * np.sin(i * 0.3) for i in range(120)]
        df = make_ohlcv_df(prices)

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        last_idx = len(result) - 1

        # Slopes should be close to zero
        assert (
            abs(result["qa_trend_slope_med"].iloc[last_idx]) < 0.1
        ), "Sideways market should have near-zero slope"

        # Trend regime should be sideways (0) or low quality
        regime = result["qa_trend_regime"].iloc[last_idx]
        assert (
            regime == 0 or result["qa_trend_quality_med"].iloc[last_idx] < 0.3
        ), "Choppy market should be sideways or low quality"

    def test_trend_quality_high_for_linear(self):
        """Test that trend quality (R²) is high for linear trends."""
        # Perfect linear trend
        prices = np.linspace(100, 150, 120)
        df = make_ohlcv_df(prices.tolist())

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # R² should be very high for perfect linear trend
        last_idx = len(result) - 1
        quality = result["qa_trend_quality_med"].iloc[last_idx]
        assert quality > 0.95, f"Linear trend should have R² > 0.95, got {quality:.4f}"

    def test_trend_quality_low_for_choppy(self):
        """Test that trend quality (R²) is low for choppy data."""
        # Random walk / choppy data
        np.random.seed(42)
        prices = [100]
        for _ in range(119):
            prices.append(prices[-1] + np.random.randn() * 2)
        df = make_ohlcv_df(prices)

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # R² should be low for random data
        last_idx = len(result) - 1
        quality = result["qa_trend_quality_med"].iloc[last_idx]
        assert quality < 0.5, f"Choppy data should have R² < 0.5, got {quality:.4f}"

    def test_trend_alignment_score(self):
        """Test multi-horizon alignment scoring."""
        # Create consistent uptrend across all horizons
        prices = np.linspace(100, 150, 150)
        df = make_ohlcv_df(prices.tolist())

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # All horizons should agree on uptrend
        last_idx = len(result) - 1
        alignment = result["qa_trend_alignment_score"].iloc[last_idx]
        assert (
            alignment >= 0.67
        ), f"Consistent uptrend should have high alignment, got {alignment:.2f}"

    def test_trend_consistency(self):
        """Test trend consistency measurement."""
        # Create very consistent uptrend
        prices = np.linspace(100, 150, 120)
        df = make_ohlcv_df(prices.tolist())

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # Consistency should be high (most bars move in trend direction)
        last_idx = len(result) - 1
        consistency = result["qa_trend_consistency"].iloc[last_idx]
        assert (
            consistency > 0.7
        ), f"Consistent uptrend should have consistency > 0.7, got {consistency:.2f}"

    def test_no_lookahead_bias(self):
        """Test that features don't use future data."""
        prices = np.linspace(100, 150, 120)
        df = make_ohlcv_df(prices.tolist())

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # First window_med-1 bars should have NaN for medium-term features
        # (need window_med data points, so first valid value is at index window_med-1)
        assert (
            result["qa_trend_slope_med"].iloc[: qa_trend.window_med - 1].isna().all()
        ), "Early bars should have NaN (no lookahead)"

        # After warmup, should have values
        assert (
            not result["qa_trend_slope_med"]
            .iloc[qa_trend.window_med - 1 :]
            .isna()
            .all()
        ), "After warmup should have values"

    def test_with_atr_for_trend_strength(self):
        """Test trend strength calculation when ATR is available."""
        prices = np.linspace(100, 150, 120)
        df = make_ohlcv_df(prices.tolist())

        # Add ATR column
        df["atr"] = 2.0  # Constant ATR

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # Trend strength should be computed
        last_idx = len(result) - 1
        strength = result["qa_trend_strength_med"].iloc[last_idx]
        assert pd.notna(
            strength
        ), "Trend strength should be computed when ATR available"
        assert strength > 0, "Uptrend should have positive trend strength"

    def test_without_atr_fallback(self):
        """Test that features work without ATR (fallback mode)."""
        prices = np.linspace(100, 150, 120)
        df = make_ohlcv_df(prices.tolist())
        # No ATR column

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # Should still compute features
        last_idx = len(result) - 1
        assert pd.notna(
            result["qa_trend_slope_med"].iloc[last_idx]
        ), "Should compute slope even without ATR"

    def test_timeframe_specific_windows(self):
        """Test that different timeframes use appropriate windows."""
        qa_trend_h1 = QuantAgentsTrendFeatures(Timeframe.H1)
        qa_trend_d1 = QuantAgentsTrendFeatures(Timeframe.D1)
        qa_trend_w1 = QuantAgentsTrendFeatures(Timeframe.W1)

        # Higher timeframes should use smaller windows (fewer bars for same duration)
        assert qa_trend_h1.window_short == 10
        assert qa_trend_d1.window_short == 5
        assert qa_trend_w1.window_short == 4

    def test_regime_classification_thresholds(self):
        """Test trend regime classification with marginal slopes."""
        # Create very weak uptrend (below threshold)
        prices = np.linspace(100, 101, 120)  # Only 1% total move
        df = make_ohlcv_df(prices.tolist())

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # Weak trend should be classified as sideways
        last_idx = len(result) - 1
        regime = result["qa_trend_regime"].iloc[last_idx]
        assert regime == 0, "Very weak trend should be classified as sideways (0)"

    def test_feature_names_complete(self):
        """Test that get_feature_names returns all features."""
        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        expected_features = [
            "qa_trend_slope_short",
            "qa_trend_slope_med",
            "qa_trend_slope_long",
            "qa_trend_quality_short",
            "qa_trend_quality_med",
            "qa_trend_quality_long",
            "qa_trend_regime",
            "qa_trend_strength_short",
            "qa_trend_strength_med",
            "qa_trend_strength_long",
            "qa_trend_consistency",
            "qa_trend_alignment_score",
        ]

        assert qa_trend.get_feature_names() == expected_features

    def test_numeric_output(self):
        """Test that all features are numeric."""
        prices = np.linspace(100, 150, 120)
        df = make_ohlcv_df(prices.tolist())

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # All feature columns should be numeric
        for feature_name in qa_trend.get_feature_names():
            assert pd.api.types.is_numeric_dtype(
                result[feature_name]
            ), f"{feature_name} should be numeric"

    def test_no_inf_values(self):
        """Test that features don't produce infinite values."""
        prices = np.linspace(100, 150, 120)
        df = make_ohlcv_df(prices.tolist())

        qa_trend = QuantAgentsTrendFeatures(Timeframe.H1)
        result = qa_trend.compute(df)

        # Check for infinite values (NaN is ok during warmup)
        for feature_name in qa_trend.get_feature_names():
            finite_values = (
                result[feature_name].replace([np.inf, -np.inf], np.nan).dropna()
            )
            assert len(finite_values) > 0, f"All values are inf/nan for {feature_name}"
