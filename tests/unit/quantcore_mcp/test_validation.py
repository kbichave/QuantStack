# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for validation tools: purged CV, lookahead detection."""

import numpy as np
import pandas as pd


class TestPurgedCV:
    """Tests for purged cross-validation."""

    def test_purged_cv_splits(self):
        """Test purged CV generates correct splits."""
        from quantstack.core.validation.purged_cv import PurgedKFoldCV

        # Create test data
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        X = pd.DataFrame({"feature": np.random.randn(1000)}, index=dates)

        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.01)
        splits = list(cv.split(X))

        assert len(splits) == 5

        for split in splits:
            # Train and test should not overlap
            train_set = set(split.train_indices)
            test_set = set(split.test_indices)
            assert len(train_set.intersection(test_set)) == 0

    def test_embargo_applied(self):
        """Test embargo creates gap between train and test."""
        from quantstack.core.validation.purged_cv import PurgedKFoldCV

        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        X = pd.DataFrame({"feature": np.random.randn(1000)}, index=dates)

        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.05)  # 5% embargo
        splits = list(cv.split(X))

        # Check that train and test don't overlap
        for split in splits:
            train_set = set(split.train_indices)
            test_set = set(split.test_indices)

            # No overlap between train and test
            assert len(train_set.intersection(test_set)) == 0

            # Both train and test should have data
            assert len(train_set) > 0
            assert len(test_set) > 0


class TestLookaheadDetection:
    """Tests for lookahead bias detection."""

    def test_clean_features_pass(self):
        """Test clean features don't trigger lookahead."""
        from scipy import stats

        np.random.seed(42)
        n = 200

        # Feature that predicts forward return with lag
        feature = pd.Series(np.random.randn(n))
        forward_return = pd.Series(np.random.randn(n))

        corr, _ = stats.spearmanr(feature, forward_return)

        # Random features should have low correlation
        assert abs(corr) < 0.2

    def test_leaky_feature_detected(self):
        """Test obvious lookahead is detected."""
        from scipy import stats

        np.random.seed(42)
        n = 200

        # Feature IS the future return (obvious leakage)
        forward_return = pd.Series(np.random.randn(n))
        feature = forward_return  # Perfect leakage

        corr, _ = stats.spearmanr(feature, forward_return)

        # Should have perfect correlation
        assert corr == 1.0
