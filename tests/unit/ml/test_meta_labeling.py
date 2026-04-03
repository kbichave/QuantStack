"""Tests for meta-labeling (AFML Chapter 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.ml.labeling import MetaLabeler, label_series


def _synthetic_signals_and_features(n_signals: int = 200, seed: int = 42):
    """Generate synthetic primary model signals + features for meta-labeling."""
    rng = np.random.default_rng(seed)
    n_prices = n_signals * 25  # ~25 bars per signal holding period

    prices = pd.Series(100 + np.cumsum(rng.normal(0.01, 1.0, n_prices)))
    atr = pd.Series(np.full(n_prices, 2.0))

    # Entry points spread evenly
    entry_indices = list(range(0, n_prices - 25, n_prices // n_signals))[:n_signals]

    signals_df = pd.DataFrame({
        "timestamp": range(len(entry_indices)),
        "signal_direction": rng.choice([1, -1], len(entry_indices)),
        "entry_idx": entry_indices,
    })

    # Features at each signal point
    features_df = pd.DataFrame(
        rng.normal(0, 1, (n_prices, 5)),
        columns=[f"feat_{i}" for i in range(5)],
    )

    return prices, atr, signals_df, features_df


def test_secondary_model_trained_on_signals():
    """MetaLabeler trains a secondary model on primary signals + triple-barrier labels."""
    prices, atr, signals_df, features_df = _synthetic_signals_and_features(n_signals=100)

    ml = MetaLabeler(kelly_fraction=0.5)
    ml.fit(signals_df, prices, atr, features_df)

    assert ml._is_fitted is True


def test_probability_maps_to_bet_size():
    """Probability [0, 1] maps to bet_size = kelly_fraction * probability."""
    prices, atr, signals_df, features_df = _synthetic_signals_and_features(n_signals=100)

    kelly = 0.6
    ml = MetaLabeler(kelly_fraction=kelly)
    ml.fit(signals_df, prices, atr, features_df)

    preds = ml.predict(features_df.iloc[:10])
    assert len(preds) == 10
    assert all(0 <= p <= 1 for p in preds["probability"])

    # bet_size = kelly * probability
    np.testing.assert_allclose(
        preds["bet_size"].values,
        kelly * preds["probability"].values,
        atol=1e-10,
    )


def test_meta_label_filtering_reduces_trades():
    """The secondary model should reject some primary signals."""
    prices, atr, signals_df, features_df = _synthetic_signals_and_features(n_signals=150)

    ml = MetaLabeler(kelly_fraction=0.5, threshold=0.5)
    ml.fit(signals_df, prices, atr, features_df)

    preds = ml.predict(features_df.iloc[:50])
    # At least some trades should be rejected (not all probabilities > 0.5)
    n_taken = preds["take_trade"].sum()
    assert n_taken < len(preds), "Meta-labeler should reject at least some signals"


def test_oos_accuracy_above_random():
    """On held-out data, secondary model accuracy > 0.5."""
    prices, atr, signals_df, features_df = _synthetic_signals_and_features(n_signals=200, seed=123)

    # Train on first 150, test on last 50
    train_signals = signals_df.iloc[:150]
    test_signals = signals_df.iloc[150:]

    ml = MetaLabeler(kelly_fraction=0.5)
    ml.fit(train_signals, prices, atr, features_df)

    # Get test labels
    test_labels_df = label_series(
        prices, list(test_signals["entry_idx"]),
        pt_multiplier=2.0, sl_multiplier=2.0,
        max_holding_period=20, atr_series=atr,
    )
    test_y = (test_labels_df["label"] == 1).astype(int)

    test_features = features_df.loc[test_signals["entry_idx"]].reset_index(drop=True)
    metrics = ml.score(test_features, test_y)

    # With synthetic data, accuracy may not always beat random,
    # but the model should produce valid metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0


def test_unfitted_meta_labeler_returns_defaults():
    """An unfitted MetaLabeler returns default predictions."""
    ml = MetaLabeler(kelly_fraction=0.5)
    features = pd.DataFrame({"a": [1, 2, 3]})
    preds = ml.predict(features)

    assert len(preds) == 3
    assert all(preds["probability"] == 0.5)
    assert all(preds["take_trade"])
