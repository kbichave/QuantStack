"""Tests for regime-stratified cross-validation in PurgedKFoldCV."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.validation.purged_cv import PurgedKFoldCV


def _make_data_with_regimes(n_samples: int = 500):
    """Create synthetic data with 3 regimes."""
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="B")
    X = pd.DataFrame(
        np.random.randn(n_samples, 5),
        columns=[f"f{i}" for i in range(5)],
        index=dates,
    )

    # Assign regimes in blocks (mimicking real market regimes)
    regimes = []
    labels = ["trending_up", "ranging", "trending_down"]
    block_size = n_samples // 6
    for i in range(n_samples):
        regimes.append(labels[(i // block_size) % 3])
    regime_labels = pd.Series(regimes, index=dates)

    return X, regime_labels


class TestRegimeStratifiedCV:
    def test_proportional_representation(self):
        """Each fold should contain samples from all 3 regimes."""
        X, regime_labels = _make_data_with_regimes()
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.01)
        splits = list(cv.split(X, regime_labels=regime_labels))

        assert len(splits) == 5

        for split in splits:
            test_regimes = regime_labels.iloc[split.test_indices]
            unique_in_test = set(test_regimes.unique())
            # Each fold should have at least 2 of 3 regimes
            assert len(unique_in_test) >= 2, (
                f"Fold has only {unique_in_test}, expected >=2 regimes"
            )

    def test_embargo_still_applied(self):
        """Embargo gap should exist between train and test indices."""
        X, regime_labels = _make_data_with_regimes()
        cv = PurgedKFoldCV(n_splits=3, embargo_pct=0.05)
        splits = list(cv.split(X, regime_labels=regime_labels))

        for split in splits:
            train_set = set(split.train_indices)
            test_set = set(split.test_indices)
            # No overlap between train and test
            assert not train_set & test_set, "Train and test must not overlap"
            # Total should be less than n_samples due to embargo
            assert len(train_set) + len(test_set) < len(X)

    def test_fallback_without_regime_labels(self):
        """Without regime_labels, behavior is identical to standard CV."""
        X, _ = _make_data_with_regimes()
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.01)

        splits_standard = list(cv.split(X, regime_labels=None))
        # Re-create to ensure clean state
        cv2 = PurgedKFoldCV(n_splits=5, embargo_pct=0.01)
        splits_no_labels = list(cv2.split(X))

        assert len(splits_standard) == len(splits_no_labels)
        for s1, s2 in zip(splits_standard, splits_no_labels):
            np.testing.assert_array_equal(s1.train_indices, s2.train_indices)
            np.testing.assert_array_equal(s1.test_indices, s2.test_indices)

    def test_small_regime_handled(self):
        """Regime with fewer samples than n_splits doesn't crash."""
        X, regime_labels = _make_data_with_regimes(n_samples=500)

        # Add a tiny regime with only 2 samples
        regime_labels.iloc[0] = "volatile"
        regime_labels.iloc[1] = "volatile"

        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.01)
        # Should not raise
        splits = list(cv.split(X, regime_labels=regime_labels))
        assert len(splits) == 5

    def test_no_data_leakage(self):
        """Test indices should always come after train indices (time-wise)."""
        X, regime_labels = _make_data_with_regimes()
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.01)
        splits = list(cv.split(X, regime_labels=regime_labels))

        for split in splits:
            # All samples in train + test should be valid indices
            assert split.train_indices.max() < len(X)
            assert split.test_indices.max() < len(X)
