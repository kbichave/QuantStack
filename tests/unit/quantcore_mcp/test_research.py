# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for research tools: walk-forward, signal validation, leakage diagnostics."""

import numpy as np
import pandas as pd
import pytest


class TestWalkForwardValidator:
    """Tests for walk-forward validation."""

    def test_walkforward_split_generation(self):
        """Test walk-forward split generation."""
        from quantstack.core.research.walkforward import WalkForwardValidator

        # Create test data (need min_train + n_splits * test_size bars)
        # 504 + 5 * 252 = 1764, so use 2000
        dates = pd.date_range("2020-01-01", periods=2000, freq="D")
        df = pd.DataFrame(
            {
                "close": np.linspace(100, 150, 2000),
            },
            index=dates,
        )

        validator = WalkForwardValidator(
            n_splits=5,
            test_size=252,
            min_train_size=504,
            expanding=True,
        )

        splits = list(validator.split(df))

        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(test_idx) == 252
            assert len(train_idx) >= 504

    def test_walkforward_insufficient_data(self):
        """Test error handling for insufficient data."""
        from quantstack.core.research.walkforward import WalkForwardValidator

        # Create small dataset
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"close": np.linspace(100, 110, 100)}, index=dates)

        validator = WalkForwardValidator(n_splits=5, test_size=252, min_train_size=504)

        with pytest.raises(ValueError):
            list(validator.split(df))


class TestSignalValidation:
    """Tests for signal validation tools."""

    def test_adf_stationary_signal(self):
        """Test ADF on stationary signal."""
        from quantstack.core.research.stat_tests import adf_test

        # Create stationary signal (white noise)
        np.random.seed(42)
        signal = pd.Series(np.random.randn(500))

        result = adf_test(signal)

        assert result.test_name == "ADF"
        assert result.is_significant  # White noise is stationary

    def test_adf_nonstationary_signal(self):
        """Test ADF on non-stationary signal."""
        from quantstack.core.research.stat_tests import adf_test

        # Create random walk (non-stationary)
        np.random.seed(42)
        signal = pd.Series(np.cumsum(np.random.randn(500)))

        result = adf_test(signal)

        assert not result.is_significant  # Random walk is non-stationary

    def test_lagged_correlation(self):
        """Test lagged cross-correlation."""
        from quantstack.core.research.stat_tests import lagged_cross_correlation

        np.random.seed(42)
        signal = pd.Series(np.random.randn(200))
        returns = pd.Series(np.random.randn(200))

        correlations = lagged_cross_correlation(signal, returns, max_lag=5)

        assert 1 in correlations
        assert 5 in correlations
        assert all(abs(v) < 1 for v in correlations.values() if not np.isnan(v))


class TestLeakageDiagnostics:
    """Tests for leakage detection."""

    def test_clean_features_pass(self):
        """Test that clean features don't trigger leakage."""
        from quantstack.core.research.leak_diagnostics import LeakageDiagnostics

        np.random.seed(42)
        n = 200

        # Create lagged features (no leakage)
        features = pd.DataFrame(
            {
                "lagged_return": np.random.randn(n),
                "momentum": np.random.randn(n),
            }
        )
        labels = pd.Series(np.random.randint(0, 2, n))
        prices = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.1))
        returns = prices.pct_change()

        diagnostics = LeakageDiagnostics()
        report = diagnostics.run_full_diagnostics(features, labels, prices, returns)

        # Should have low severity or no leakage
        assert report.severity in ["none", "low", "medium"]

    def test_leaky_feature_detected(self):
        """Test that obvious leakage is detected."""
        from quantstack.core.research.leak_diagnostics import LeakageDiagnostics

        np.random.seed(42)
        n = 200

        # Create feature that IS the future return (obvious leakage)
        returns = pd.Series(np.random.randn(n) * 0.01)
        features = pd.DataFrame(
            {
                "future_return": returns,  # This IS the return - perfect leakage
            }
        )
        labels = (returns > 0).astype(int)
        prices = pd.Series(100 + np.cumsum(returns))

        diagnostics = LeakageDiagnostics()
        report = diagnostics.run_full_diagnostics(features, labels, prices, returns)

        # Should detect leakage
        assert report.has_leakage
