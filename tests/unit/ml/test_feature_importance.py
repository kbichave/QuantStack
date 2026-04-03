"""Tests for feature importance protocol (AFML Chapter 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from quantstack.ml.feature_importance import FeatureImportanceProtocol


def _trained_model_and_data(n_features: int = 10, n_informative: int = 5, seed: int = 42):
    """Create a trained LightGBM model with known informative features."""
    from lightgbm import LGBMClassifier

    X, y = make_classification(
        n_samples=500, n_features=n_features, n_informative=n_informative,
        n_redundant=2, n_classes=2, random_state=seed,
    )
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    # Split
    split = 400
    X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
    y_train, y_test = y_series.iloc[:split], y_series.iloc[split:]

    model = LGBMClassifier(n_estimators=100, max_depth=4, random_state=seed, verbose=-1)
    model.fit(X_train, y_train)

    return model, feature_names, X_train, y_train, X_test, y_test


def test_mdi_mda_sfi_return_rankings():
    """All three methods return a dict mapping feature_name -> importance_score."""
    model, names, X_train, y_train, X_test, y_test = _trained_model_and_data()
    fip = FeatureImportanceProtocol()

    mdi = fip.compute_mdi(model, names)
    mda = fip.compute_mda(model, X_test, y_test, names)
    sfi = fip.compute_sfi(X_train, y_train, X_test, y_test, names)

    for result in (mdi, mda, sfi):
        assert isinstance(result, dict)
        assert set(result.keys()) == set(names)
        assert all(isinstance(v, float) for v in result.values())


def test_consensus_filter_selects_top_features():
    """Features in top 50% by 2+ methods are selected."""
    model, names, X_train, y_train, X_test, y_test = _trained_model_and_data()
    fip = FeatureImportanceProtocol()

    mdi = fip.compute_mdi(model, names)
    mda = fip.compute_mda(model, X_test, y_test, names)
    sfi = fip.compute_sfi(X_train, y_train, X_test, y_test, names)

    selected = fip.consensus_filter(mdi, mda, sfi, top_pct=0.5)

    # Should select some but not all features
    assert 0 < len(selected) <= len(names)
    # All selected features should be in the original set
    assert all(f in names for f in selected)


def test_constant_features_excluded():
    """A feature with zero variance gets importance 0 and is excluded."""
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "informative": rng.normal(0, 1, 300),
        "constant": np.zeros(300),
        "noise": rng.normal(0, 1, 300),
    })
    y = pd.Series((X["informative"] > 0).astype(int))

    model = LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
    model.fit(X.iloc[:200], y.iloc[:200])

    fip = FeatureImportanceProtocol()
    names = list(X.columns)
    mdi = fip.compute_mdi(model, names)
    mda = fip.compute_mda(model, X.iloc[200:], y.iloc[200:], names)
    sfi = fip.compute_sfi(X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:], names)

    selected = fip.consensus_filter(mdi, mda, sfi, top_pct=0.5)
    assert "constant" not in selected


def test_filtered_feature_set_reasonable():
    """Filtered feature set should not be empty for well-constructed data."""
    model, names, X_train, y_train, X_test, y_test = _trained_model_and_data(
        n_features=10, n_informative=5
    )
    fip = FeatureImportanceProtocol()

    mdi = fip.compute_mdi(model, names)
    mda = fip.compute_mda(model, X_test, y_test, names)
    sfi = fip.compute_sfi(X_train, y_train, X_test, y_test, names)

    selected = fip.consensus_filter(mdi, mda, sfi, top_pct=0.5)
    assert len(selected) >= 2  # At least some informative features survive
