# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Lorentzian k-NN Classifier.

A k-nearest-neighbour classifier that uses Lorentzian distance instead of
Euclidean distance to find historical analogues.

**Why Lorentzian distance?**

Euclidean distance in feature space treats all dimensions equally and is
dominated by large-valued features. Lorentzian distance:

    D(P, Q) = Σ_i  ln(1 + |P_i - Q_i|)

has logarithmic growth per dimension, making it:
1. Naturally bounded per feature (no single feature dominates)
2. More sensitive to small differences between similar bars
3. Robust to outliers (log compresses large distances)

This is the distance metric underlying the "Lorentzian Classification" indicator
popularised by jdehorty on TradingView. Implementation here is pure numpy —
no sklearn dependency — and fully causal (no lookahead).

**Usage**

1. Build a feature matrix X (rows = bars, cols = features) from your quantcore
   feature pipeline.
2. Build a label array y (1 = win, 0 = loss) from EventLabeler.
3. Call `fit(X, y)` to store the training library.
4. Call `predict(X_new)` to classify new bars.

**Design constraints**
- Neighbour search is O(N × k) per query bar. For live use, cap `library_size`
  to avoid performance degradation.
- Features must be normalised before training (use `StandardScaler` or
  `QuantileTransformer`). The model stores the fitted scaler and applies it
  automatically during prediction.
- Library is stored sorted by date; we never allow future neighbours (causal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


# ---------------------------------------------------------------------------
# Lorentzian distance
# ---------------------------------------------------------------------------


def lorentzian_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Lorentzian distance between two feature vectors.

    D(a, b) = Σ_i  ln(1 + |a_i - b_i|)

    Parameters
    ----------
    a, b : np.ndarray
        1-D feature vectors of the same length.

    Returns
    -------
    float — non-negative distance (0 when a == b).
    """
    return float(np.sum(np.log1p(np.abs(a - b))))


def lorentzian_distance_matrix(query: np.ndarray, library: np.ndarray) -> np.ndarray:
    """
    Compute Lorentzian distance from one query vector to all library vectors.

    Parameters
    ----------
    query   : (n_features,)
    library : (n_library, n_features)

    Returns
    -------
    (n_library,) distances
    """
    return np.sum(np.log1p(np.abs(library - query)), axis=1)


# ---------------------------------------------------------------------------
# Simple scaler (avoids sklearn dependency)
# ---------------------------------------------------------------------------


class _StandardScaler:
    """Minimal StandardScaler (mean=0, std=1) without sklearn."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_StandardScaler":
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # prevent division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Scaler not fitted."
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# LorentzianKNN
# ---------------------------------------------------------------------------


class LorentzianKNN:
    """
    Lorentzian k-Nearest-Neighbour classifier for bar-level trade labeling.

    Each bar's label is determined by the majority vote of its k nearest
    historical neighbours in Lorentzian feature space.

    Parameters
    ----------
    k : int
        Number of nearest neighbours to use. Default 8.
    library_size : int
        Maximum number of historical bars to keep in the search library.
        Older bars are dropped (FIFO) to bound search cost. Default 2000.
    min_bars_between_neighbours : int
        Minimum index gap between selected neighbours to avoid autocorrelation
        (neighbours too close in time share the same market regime).
        Default 4.
    feature_names : list[str] | None
        Names of the features (columns of X). Stored for interpretability.
    """

    def __init__(
        self,
        k: int = 8,
        library_size: int = 2000,
        min_bars_between_neighbours: int = 4,
        feature_names: list[str] | None = None,
    ) -> None:
        self.k = k
        self.library_size = library_size
        self.min_bars_between_neighbours = min_bars_between_neighbours
        self.feature_names = feature_names

        self._scaler = _StandardScaler()
        self._library_X: np.ndarray | None = None  # (N, n_features) scaled
        self._library_y: np.ndarray | None = None  # (N,) int labels
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LorentzianKNN":
        """
        Store the training library.

        Parameters
        ----------
        X : (n_bars, n_features) — feature matrix (unscaled)
        y : (n_bars,) — binary labels (1 = bullish/win, 0 = bearish/loss,
                         or -1/+1 convention — any two distinct values work)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # Fit scaler on full training set
        X_scaled = self._scaler.fit_transform(X)

        # Cap library size (keep most recent bars)
        if len(X_scaled) > self.library_size:
            X_scaled = X_scaled[-self.library_size :]
            y = y[-self.library_size :]

        self._library_X = X_scaled
        self._library_y = y
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability of the positive class (label == 1) for each bar.

        Parameters
        ----------
        X : (n_bars, n_features) — unscaled feature matrix

        Returns
        -------
        (n_bars,) — float in [0, 1]; probability of positive outcome
        """
        if not self._is_fitted:
            raise RuntimeError(
                "LorentzianKNN must be fitted before calling predict_proba."
            )

        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)

        lib_X = self._library_X
        lib_y = self._library_y
        n_lib = len(lib_X)
        n_query = len(X_scaled)
        proba = np.full(n_query, 0.5)

        for qi in range(n_query):
            q = X_scaled[qi]
            dists = lorentzian_distance_matrix(q, lib_X)

            # Select k diverse neighbours (gap constraint prevents autocorrelation)
            sorted_idx = np.argsort(dists)
            selected: list[int] = []
            for idx in sorted_idx:
                if len(selected) >= self.k:
                    break
                # Check minimum gap to all already-selected neighbours
                too_close = any(
                    abs(idx - s) < self.min_bars_between_neighbours for s in selected
                )
                if not too_close:
                    selected.append(int(idx))

            if not selected:
                continue

            neighbour_labels = lib_y[selected]
            positive_votes = np.sum(neighbour_labels == 1)
            proba[qi] = positive_votes / len(selected)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary label (1 or 0) for each bar.

        Parameters
        ----------
        X : (n_bars, n_features)

        Returns
        -------
        (n_bars,) int array — 1 or 0
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Online update (append new bars to library)
    # ------------------------------------------------------------------

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        Append new labelled bars to the search library (online learning).

        The library is capped at `library_size`; oldest bars are dropped.

        Parameters
        ----------
        X_new : (n_new, n_features) — unscaled new bars
        y_new : (n_new,) — labels for new bars
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before update().")

        X_new_scaled = self._scaler.transform(np.asarray(X_new, dtype=float))
        y_new = np.asarray(y_new, dtype=int)

        self._library_X = np.vstack([self._library_X, X_new_scaled])
        self._library_y = np.concatenate([self._library_y, y_new])

        # Trim to library_size
        if len(self._library_X) > self.library_size:
            excess = len(self._library_X) - self.library_size
            self._library_X = self._library_X[excess:]
            self._library_y = self._library_y[excess:]

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def nearest_neighbours(self, x: np.ndarray, k: int | None = None) -> dict[str, Any]:
        """
        Return the k nearest neighbours from the library for a single query bar.

        Parameters
        ----------
        x : (n_features,) — single unscaled feature vector
        k : optional override for self.k

        Returns
        -------
        dict with keys:
            indices    – library indices of nearest neighbours
            distances  – Lorentzian distances
            labels     – labels at those indices
            vote_ratio – fraction voting positive (1)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before nearest_neighbours().")

        k = k or self.k
        x_scaled = self._scaler.transform(x.reshape(1, -1))[0]
        dists = lorentzian_distance_matrix(x_scaled, self._library_X)

        sorted_idx = np.argsort(dists)
        selected: list[int] = []
        for idx in sorted_idx:
            if len(selected) >= k:
                break
            too_close = any(
                abs(idx - s) < self.min_bars_between_neighbours for s in selected
            )
            if not too_close:
                selected.append(int(idx))

        selected_arr = np.array(selected)
        labels = self._library_y[selected_arr]
        return {
            "indices": selected_arr,
            "distances": dists[selected_arr],
            "labels": labels,
            "vote_ratio": float(np.mean(labels == 1)),
        }

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def library_size_actual(self) -> int:
        """Number of bars currently in the search library."""
        return len(self._library_X) if self._library_X is not None else 0

    def __repr__(self) -> str:
        return (
            f"LorentzianKNN(k={self.k}, library_size={self.library_size}, "
            f"fitted={self._is_fitted}, library_bars={self.library_size_actual})"
        )
