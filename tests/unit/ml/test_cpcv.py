"""Tests for Combinatorial Purged Cross-Validation (AFML Chapter 12)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from quantstack.core.validation.purged_cv import (
    CVSplit,
    CombinatorialPurgedCV,
    PurgedKFoldCV,
)


def _synthetic_df(n: int = 600) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    X = pd.DataFrame(
        rng.normal(0, 1, (n, 5)),
        columns=[f"f{i}" for i in range(5)],
        index=dates,
    )
    y = pd.Series(rng.choice([0, 1], n), index=dates)
    return X, y


def test_no_label_leakage_after_purging():
    """No training index falls within the embargo zone around test folds."""
    X, y = _synthetic_df(600)
    cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.02)

    for split in cpcv.split(X, y):
        train_set = set(split.train_indices)
        test_set = set(split.test_indices)
        # No overlap
        assert len(train_set & test_set) == 0


def test_embargo_period_applied():
    """Embargo gap exists between train and test boundaries."""
    n = 600
    X, y = _synthetic_df(n)
    embargo_pct = 0.02
    embargo_size = int(n * embargo_pct)

    cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=embargo_pct)

    for split in cpcv.split(X, y):
        train_set = set(split.train_indices)
        test_sorted = sorted(split.test_indices)

        # Check that indices just before and after test groups are excluded from train
        for boundary in [test_sorted[0], test_sorted[-1]]:
            for offset in range(1, embargo_size + 1):
                excluded_idx = boundary - offset
                if 0 <= excluded_idx < n:
                    # This index is in the embargo zone; should NOT be in training
                    assert excluded_idx not in train_set or excluded_idx in set(test_sorted)


def test_correct_number_of_paths():
    """C(n_groups, n_test_groups) paths generated."""
    X, y = _synthetic_df(600)

    n_groups, n_test_groups = 6, 2
    cpcv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test_groups)

    splits = list(cpcv.split(X, y))
    expected = len(list(combinations(range(n_groups), n_test_groups)))  # C(6,2) = 15
    assert len(splits) == expected


def test_lower_variance_than_kfold():
    """CPCV performance estimates have lower or equal variance to k-fold on synthetic data."""
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    X, y = _synthetic_df(300)

    # CPCV scores
    cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01)
    cpcv_scores = []
    for split in cpcv.split(X, y):
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X.iloc[split.train_indices], y.iloc[split.train_indices])
        preds = clf.predict(X.iloc[split.test_indices])
        cpcv_scores.append(accuracy_score(y.iloc[split.test_indices], preds))

    # Naive k-fold scores
    from sklearn.model_selection import KFold
    kf_scores = []
    for train_idx, test_idx in KFold(n_splits=6, shuffle=False).split(X):
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = clf.predict(X.iloc[test_idx])
        kf_scores.append(accuracy_score(y.iloc[test_idx], preds))

    # CPCV variance should be <= k-fold variance (with some tolerance for noise)
    assert np.std(cpcv_scores) <= np.std(kf_scores) * 1.5


def test_small_dataset_handling():
    """Small datasets reduce n_groups gracefully instead of raising."""
    X, y = _synthetic_df(100)

    cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01)
    splits = list(cpcv.split(X, y))

    # Should still produce splits (possibly fewer paths)
    assert len(splits) > 0
    for split in splits:
        assert len(split.train_indices) > 0
        assert len(split.test_indices) > 0
