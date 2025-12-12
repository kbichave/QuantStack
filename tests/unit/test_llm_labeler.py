"""
Tests for LLM-based labeling interface.

Verifies:
1. Mock label generation
2. Hybrid label creation
3. Label statistics
4. File loading (when files exist)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from quantcore.labeling.llm_labeler import LLMLabelProvider
from tests.conftest import make_ohlcv_df


class TestLLMLabelProvider:
    """Test LLM label provider functionality."""

    def test_initialization_mock_mode(self):
        """Test initialization in mock mode."""
        provider = LLMLabelProvider(use_mock=True)
        assert provider.use_mock is True
        assert provider.labels_df is None

    def test_initialization_with_nonexistent_file(self):
        """Test initialization with nonexistent file path."""
        provider = LLMLabelProvider(
            label_file_path="/tmp/nonexistent_labels.parquet", use_mock=True
        )
        assert provider.labels_df is None  # File doesn't exist, falls back to mock

    def test_attach_llm_labels_mock_mode(self):
        """Test attaching mock LLM labels to DataFrame."""
        prices = np.linspace(100, 120, 60)
        df = make_ohlcv_df(prices.tolist())

        # Add ATR label
        df["label_long"] = np.random.choice([0, 1], size=len(df))

        provider = LLMLabelProvider(use_mock=True)
        result = provider.attach_llm_labels(df)

        # Check that LLM label columns were added
        assert "label_llm_quality" in result.columns
        assert "label_llm_type" in result.columns

        # Check that quality is in [0, 1]
        quality = result["label_llm_quality"].dropna()
        assert (quality >= 0).all()
        assert (quality <= 1).all()

    def test_mock_labels_with_qa_trend(self):
        """Test that mock labels use QA trend features when available."""
        prices = np.linspace(100, 150, 60)  # Strong uptrend
        df = make_ohlcv_df(prices.tolist())

        # Add features that should improve quality
        df["label_long"] = 1  # Win
        df["qa_trend_regime"] = 1  # Uptrend
        df["qa_pattern_is_pullback"] = 1  # Pullback in uptrend

        provider = LLMLabelProvider(use_mock=True)
        result = provider.attach_llm_labels(df)

        # Quality should be high for ideal setup
        quality = result["label_llm_quality"].iloc[-1]
        assert quality > 0.7, f"Ideal setup should have high quality, got {quality:.2f}"

        # Label type should be "ideal_pullback"
        label_type = result["label_llm_type"].iloc[-1]
        assert (
            label_type == "ideal_pullback"
        ), f"Expected 'ideal_pullback', got '{label_type}'"

    def test_mock_labels_with_poor_setup(self):
        """Test that mock labels downgrade quality for poor setups."""
        prices = [100] * 60  # Flat/choppy
        df = make_ohlcv_df(prices)

        df["label_long"] = 0  # Loss
        df["qa_trend_regime"] = 0  # Sideways
        df["qa_trend_quality_med"] = 0.2  # Low quality trend

        provider = LLMLabelProvider(use_mock=True)
        result = provider.attach_llm_labels(df)

        # Quality should be lower for poor setup
        quality = result["label_llm_quality"].iloc[-1]
        assert quality < 0.5, f"Poor setup should have low quality, got {quality:.2f}"

    def test_create_hybrid_label_basic(self):
        """Test hybrid label creation from ATR + LLM quality."""
        df = pd.DataFrame(
            {
                "label_long": [1, 1, 0, 0, 1],
                "label_llm_quality": [0.9, 0.3, 0.8, 0.2, 0.5],
            },
            index=pd.date_range("2024-01-01", periods=5, freq="1H"),
        )

        provider = LLMLabelProvider(use_mock=True)
        hybrid = provider.create_hybrid_label(df, quality_weight=0.3)

        # Hybrid should be weighted combination
        assert len(hybrid) == 5
        assert (hybrid >= 0).all()
        assert (hybrid <= 1).all()

        # High ATR + high quality -> high hybrid
        assert hybrid.iloc[0] > 0.9, "High ATR + high quality should give high hybrid"

        # High ATR + low quality -> lower hybrid (deemphasized)
        assert (
            0.7 < hybrid.iloc[1] < 0.9
        ), "High ATR + low quality should be deemphasized"

    def test_create_hybrid_label_missing_atr(self):
        """Test hybrid label when ATR label is missing."""
        df = pd.DataFrame(
            {
                "label_llm_quality": [0.9, 0.3, 0.8],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1H"),
        )

        provider = LLMLabelProvider(use_mock=True)
        hybrid = provider.create_hybrid_label(df)

        # Should return all NaN when ATR label missing
        assert hybrid.isna().all()

    def test_create_hybrid_label_missing_quality(self):
        """Test hybrid label when LLM quality is missing."""
        df = pd.DataFrame(
            {
                "label_long": [1, 0, 1],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1H"),
        )

        provider = LLMLabelProvider(use_mock=True)
        hybrid = provider.create_hybrid_label(df)

        # Should fall back to ATR label
        assert (hybrid == df["label_long"]).all()

    def test_create_hybrid_label_different_weights(self):
        """Test that quality weight parameter works correctly."""
        df = pd.DataFrame(
            {
                "label_long": [1, 1],
                "label_llm_quality": [0.0, 1.0],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="1H"),
        )

        provider = LLMLabelProvider(use_mock=True)

        # With weight=0, hybrid should equal ATR label
        hybrid_0 = provider.create_hybrid_label(df, quality_weight=0.0)
        assert (hybrid_0 == 1.0).all()

        # With weight=1, hybrid should equal quality
        hybrid_1 = provider.create_hybrid_label(df, quality_weight=1.0)
        assert hybrid_1.iloc[0] == 0.0
        assert hybrid_1.iloc[1] == 1.0

        # With weight=0.5, hybrid should be average
        hybrid_half = provider.create_hybrid_label(df, quality_weight=0.5)
        assert hybrid_half.iloc[0] == 0.5
        assert hybrid_half.iloc[1] == 1.0

    def test_get_label_statistics_basic(self):
        """Test label statistics computation."""
        df = pd.DataFrame(
            {
                "label_llm_quality": [0.9, 0.8, 0.7, 0.6, 0.5],
                "label_llm_type": [
                    "ideal_pullback",
                    "mr_opportunity",
                    "ideal_pullback",
                    "neutral",
                    "weak_consolidation",
                ],
            },
            index=pd.date_range("2024-01-01", periods=5, freq="1H"),
        )

        provider = LLMLabelProvider(use_mock=True)
        stats = provider.get_label_statistics(df)

        assert stats["count"] == 5
        assert stats["mean_quality"] == pytest.approx(0.7)
        assert stats["median_quality"] == pytest.approx(0.7)
        assert "type_distribution" in stats
        assert stats["type_distribution"]["ideal_pullback"] == 2

    def test_get_label_statistics_empty(self):
        """Test label statistics with no data."""
        df = pd.DataFrame(
            {
                "label_llm_quality": [np.nan, np.nan],
                "label_llm_type": [None, None],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="1H"),
        )

        provider = LLMLabelProvider(use_mock=True)
        stats = provider.get_label_statistics(df)

        assert stats["count"] == 0

    def test_mock_labels_deterministic(self):
        """Test that mock labels are deterministic (same input -> same output)."""
        prices = np.linspace(100, 120, 60)
        df1 = make_ohlcv_df(prices.tolist())
        df1["label_long"] = 1
        df1["qa_trend_regime"] = 1

        df2 = df1.copy()

        provider = LLMLabelProvider(use_mock=True)
        result1 = provider.attach_llm_labels(df1)
        result2 = provider.attach_llm_labels(df2)

        # Results should be identical
        pd.testing.assert_series_equal(
            result1["label_llm_quality"],
            result2["label_llm_quality"],
            check_names=False,
        )

    def test_mock_labels_mr_opportunity(self):
        """Test mock labeling for mean reversion opportunity."""
        df = pd.DataFrame(
            {
                "close": [100] * 50 + [110],
                "high": [101] * 50 + [111],
                "low": [99] * 50 + [109],
                "open": [100] * 50 + [110],
                "volume": [1000] * 51,
                "label_long": [0] * 50 + [1],
                "qa_trend_regime": [0] * 51,
                "qa_pattern_consolidation": [1] * 51,
                "zscore_price": [0] * 50 + [-2.5],
            },
            index=pd.date_range("2024-01-01", periods=51, freq="1H"),
        )

        provider = LLMLabelProvider(use_mock=True)
        result = provider.attach_llm_labels(df)

        # Should boost quality for MR opportunity
        quality_last = result["label_llm_quality"].iloc[-1]
        assert quality_last > 0.5, "MR opportunity should have decent quality"

    def test_mock_labels_late_entry(self):
        """Test mock labeling for late entry (trend without pullback)."""
        df = pd.DataFrame(
            {
                "close": [100] * 51,
                "high": [101] * 51,
                "low": [99] * 51,
                "open": [100] * 51,
                "volume": [1000] * 51,
                "label_long": [1] * 51,
                "qa_trend_regime": [1] * 51,  # Strong uptrend
                "qa_pattern_is_pullback": [0] * 51,  # No pullback
            },
            index=pd.date_range("2024-01-01", periods=51, freq="1H"),
        )

        provider = LLMLabelProvider(use_mock=True)
        result = provider.attach_llm_labels(df)

        # Should reduce quality for late entry
        quality_last = result["label_llm_quality"].iloc[-1]
        label_type = result["label_llm_type"].iloc[-1]
        # Quality might be reduced, and type should reflect late entry
        assert label_type == "late_entry" or quality_last < 0.8

    def test_attach_labels_preserves_original_columns(self):
        """Test that attaching labels doesn't remove original columns."""
        prices = np.linspace(100, 120, 60)
        df = make_ohlcv_df(prices.tolist())
        df["label_long"] = 1

        original_cols = set(df.columns)

        provider = LLMLabelProvider(use_mock=True)
        result = provider.attach_llm_labels(df)

        # All original columns should still be present
        for col in original_cols:
            assert col in result.columns, f"Original column '{col}' was removed"

    def test_attach_labels_index_preserved(self):
        """Test that DataFrame index is preserved."""
        prices = np.linspace(100, 120, 60)
        df = make_ohlcv_df(prices.tolist())

        original_index = df.index.copy()

        provider = LLMLabelProvider(use_mock=True)
        result = provider.attach_llm_labels(df)

        pd.testing.assert_index_equal(result.index, original_index)
