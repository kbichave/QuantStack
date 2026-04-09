# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Strategy-of-Strategies Meta-Model — dynamic capital allocation across strategies.

Trains a linear regression per strategy to predict forward returns from
market features (regime, IC, volatility, correlation, Sharpe). The
predicted returns are converted to portfolio weights via normalization.

Why linear regression:
  - Interpretable coefficients (auditable for a trading system).
  - Fast to train/retrain (seconds, not hours).
  - Regularization not needed at 5 features — overfitting risk is low
    with 30+ daily observations per retrain window.

Retrain cadence: every 30 calendar days or on-demand after regime shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from loguru import logger
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRAIN_INTERVAL_DAYS = 30
_MIN_WEIGHT = 0.0  # clip negatives — no shorting the meta-portfolio


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MetaModelFeatures:
    """Feature vector for predicting strategy-level forward returns."""

    regime: str
    avg_ic: float
    vol_30d: float
    cross_correlation: float
    recent_sharpe: float

    def to_array(self) -> np.ndarray:
        """Encode to numeric array for model input.

        Regime is hashed to a float in [0, 1) for simplicity.  A production
        upgrade would one-hot encode or use a regime ordinal map.
        """
        regime_hash = (hash(self.regime) % 1000) / 1000.0
        return np.array(
            [regime_hash, self.avg_ic, self.vol_30d, self.cross_correlation, self.recent_sharpe],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_meta_model(
    strategy_returns: dict[str, np.ndarray],
    feature_matrix: np.ndarray,
) -> dict:
    """Train a linear regression per strategy to predict returns from features.

    Parameters
    ----------
    strategy_returns:
        Mapping of strategy_id -> 1-D array of daily returns (same length
        as ``feature_matrix`` rows).
    feature_matrix:
        2-D array of shape (n_days, n_features). Each row corresponds to
        the feature snapshot for that day.

    Returns
    -------
    dict with keys:
        - "coefficients": {strategy_id: list[float]} — regression coefficients
        - "intercepts": {strategy_id: float}
        - "r2_scores": {strategy_id: float}
        - "trained_date": str (ISO format)
    """
    if feature_matrix.shape[0] < 10:
        raise ValueError(
            f"Need at least 10 observations to train, got {feature_matrix.shape[0]}"
        )

    coefficients: dict[str, list[float]] = {}
    intercepts: dict[str, float] = {}
    r2_scores: dict[str, float] = {}

    for strategy_id, returns in strategy_returns.items():
        if len(returns) != feature_matrix.shape[0]:
            logger.warning(
                "Skipping {sid}: returns length {rlen} != features length {flen}",
                sid=strategy_id,
                rlen=len(returns),
                flen=feature_matrix.shape[0],
            )
            continue

        model = LinearRegression()
        model.fit(feature_matrix, returns)

        coefficients[strategy_id] = model.coef_.tolist()
        intercepts[strategy_id] = float(model.intercept_)
        r2_scores[strategy_id] = round(float(model.score(feature_matrix, returns)), 4)

        logger.debug(
            "Meta-model {sid}: R2={r2:.4f}, coefs={coefs}",
            sid=strategy_id,
            r2=r2_scores[strategy_id],
            coefs=[round(c, 4) for c in model.coef_],
        )

    return {
        "coefficients": coefficients,
        "intercepts": intercepts,
        "r2_scores": r2_scores,
        "trained_date": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_strategy_weights(
    features: MetaModelFeatures,
    model_coefficients: dict,
) -> dict[str, float]:
    """Predict portfolio weights across strategies for the given features.

    Steps:
      1. Predict expected return per strategy from coefficients.
      2. Clip negatives to zero (no shorting at meta level).
      3. Normalize to sum=1. If all predictions are non-positive,
         return equal weights as a safe fallback.

    Parameters
    ----------
    features:
        Current market feature snapshot.
    model_coefficients:
        Output from ``train_meta_model``.

    Returns
    -------
    dict of strategy_id -> weight (float, sums to 1.0).
    """
    coefficients = model_coefficients["coefficients"]
    intercepts = model_coefficients["intercepts"]

    feature_vec = features.to_array()
    raw_predictions: dict[str, float] = {}

    for strategy_id, coefs in coefficients.items():
        coef_arr = np.array(coefs, dtype=np.float64)
        intercept = intercepts[strategy_id]
        predicted = float(np.dot(coef_arr, feature_vec) + intercept)
        raw_predictions[strategy_id] = predicted

    # Clip negatives
    clipped = {sid: max(pred, _MIN_WEIGHT) for sid, pred in raw_predictions.items()}
    total = sum(clipped.values())

    if total <= 0:
        # Fallback: equal weight when all predictions are non-positive
        n = len(clipped)
        logger.warning(
            "All meta-model predictions non-positive, falling back to equal weights"
        )
        return {sid: round(1.0 / n, 4) for sid in clipped} if n > 0 else {}

    weights = {sid: round(w / total, 4) for sid, w in clipped.items()}
    return weights


# ---------------------------------------------------------------------------
# Retrain gate
# ---------------------------------------------------------------------------


def should_retrain(last_train_date: date, today: date | None = None) -> bool:
    """Return True if the meta-model should be retrained.

    Retrain trigger: more than ``_RETRAIN_INTERVAL_DAYS`` days since last train.
    """
    if today is None:
        today = date.today()
    days_since = (today - last_train_date).days
    return days_since > _RETRAIN_INTERVAL_DAYS
