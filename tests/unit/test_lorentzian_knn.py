# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LorentzianKNN classifier."""

import numpy as np
import pytest

from quantstack.ml.lorentzian_knn import (
    LorentzianKNN,
    lorentzian_distance,
    lorentzian_distance_matrix,
)


# ---------------------------------------------------------------------------
# Lorentzian distance
# ---------------------------------------------------------------------------


class TestLorentzianDistance:
    def test_zero_distance_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert lorentzian_distance(a, a) == pytest.approx(0.0)

    def test_symmetric(self):
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([2.0, 3.0, 4.5])
        assert lorentzian_distance(a, b) == pytest.approx(lorentzian_distance(b, a))

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal(10)
            b = rng.standard_normal(10)
            assert lorentzian_distance(a, b) >= 0.0

    def test_known_value(self):
        """D([0], [0]) = ln(1+0) = 0; D([0], [e-1]) ≈ ln(e) = 1."""
        a = np.array([0.0])
        b = np.array([np.e - 1])
        assert lorentzian_distance(a, b) == pytest.approx(1.0, rel=1e-6)

    def test_matrix_matches_scalar(self):
        query = np.array([1.0, 2.0, 3.0])
        lib = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])
        dm = lorentzian_distance_matrix(query, lib)
        for i in range(3):
            assert dm[i] == pytest.approx(lorentzian_distance(query, lib[i]))


# ---------------------------------------------------------------------------
# LorentzianKNN
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_dataset():
    """200 bars, 5 features, linearly separable: first half y=1, second half y=0."""
    rng = np.random.default_rng(7)
    X_pos = rng.standard_normal((100, 5)) + np.array([2.0, 1.0, 0.5, -0.5, 1.5])
    X_neg = rng.standard_normal((100, 5)) + np.array([-2.0, -1.0, -0.5, 0.5, -1.5])
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    return X, y


class TestLorentzianKNNBasic:
    def test_repr(self):
        model = LorentzianKNN(k=8)
        assert "LorentzianKNN" in repr(model)
        assert "fitted=False" in repr(model)

    def test_fit_returns_self(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5)
        result = model.fit(X, y)
        assert result is model
        assert model._is_fitted

    def test_library_size_after_fit(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5, library_size=150)
        model.fit(X, y)
        assert model.library_size_actual == 150  # capped at library_size

    def test_predict_shape(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5).fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_predict_proba_in_range(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5).fit(X, y)
        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5).fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_accuracy_above_chance_on_separable_data(self, simple_dataset):
        """On clearly separable data, accuracy should exceed 65%."""
        X, y = simple_dataset
        # Train on first 150, test on last 50
        model = LorentzianKNN(k=5, min_bars_between_neighbours=1).fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        accuracy = np.mean(preds == y[150:])
        assert accuracy > 0.65, f"Accuracy {accuracy:.2f} below 0.65 on separable data"

    def test_unfitted_predict_raises(self):
        model = LorentzianKNN()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((3, 5)))

    def test_unfitted_predict_proba_raises(self):
        model = LorentzianKNN()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.zeros((3, 5)))


class TestLorentzianKNNUpdate:
    def test_update_expands_library(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5, library_size=300).fit(X[:100], y[:100])
        initial_size = model.library_size_actual
        model.update(X[100:120], y[100:120])
        assert model.library_size_actual == initial_size + 20

    def test_update_caps_at_library_size(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5, library_size=100).fit(X[:100], y[:100])
        model.update(X[100:150], y[100:150])  # would push to 150 but capped at 100
        assert model.library_size_actual == 100

    def test_update_before_fit_raises(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN()
        with pytest.raises(RuntimeError):
            model.update(X[:10], y[:10])


class TestLorentzianKNNNearestNeighbours:
    def test_returns_dict(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5).fit(X, y)
        result = model.nearest_neighbours(X[0])
        assert isinstance(result, dict)

    def test_required_keys(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5).fit(X, y)
        result = model.nearest_neighbours(X[0])
        assert {"indices", "distances", "labels", "vote_ratio"}.issubset(result.keys())

    def test_vote_ratio_in_range(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5).fit(X, y)
        result = model.nearest_neighbours(X[0])
        assert 0.0 <= result["vote_ratio"] <= 1.0

    def test_distances_nonnegative(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5).fit(X, y)
        result = model.nearest_neighbours(X[0])
        assert (result["distances"] >= 0).all()

    def test_gap_constraint_respected(self, simple_dataset):
        """Neighbours should be at least min_bars_between_neighbours apart."""
        X, y = simple_dataset
        gap = 10
        model = LorentzianKNN(k=5, min_bars_between_neighbours=gap).fit(X, y)
        result = model.nearest_neighbours(X[50])
        idxs = sorted(result["indices"])
        for i in range(len(idxs) - 1):
            assert idxs[i + 1] - idxs[i] >= gap


class TestLorentzianKNNEdgeCases:
    def test_single_feature(self):
        X = np.arange(50, dtype=float).reshape(-1, 1)
        y = (X[:, 0] > 25).astype(int)
        model = LorentzianKNN(k=3, min_bars_between_neighbours=1).fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_all_same_label(self):
        X = np.random.randn(30, 4)
        y = np.ones(30, dtype=int)
        model = LorentzianKNN(k=3).fit(X, y)
        proba = model.predict_proba(X[:5])
        assert (proba == 1.0).all()

    def test_single_bar_query(self, simple_dataset):
        X, y = simple_dataset
        model = LorentzianKNN(k=5).fit(X, y)
        proba = model.predict_proba(X[0:1])
        assert proba.shape == (1,)

    def test_feature_names_stored(self):
        model = LorentzianKNN(feature_names=["rsi", "macd", "atr"])
        assert model.feature_names == ["rsi", "macd", "atr"]
