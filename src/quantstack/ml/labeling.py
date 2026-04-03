"""Triple-barrier labeling and meta-labeling (AFML Chapters 3-4).

Triple-barrier labeling replaces naive return-based labels with path-aware
labels that account for profit-taking, stop-loss, and time expiry. This
produces higher-quality training targets for ML models.

Meta-labeling separates direction prediction (primary model) from bet sizing
(secondary model), allowing each to be optimized independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def triple_barrier_label(
    prices: pd.Series,
    entry_idx: int,
    pt_multiplier: float,
    sl_multiplier: float,
    max_holding_period: int,
    volatility: float,
) -> int:
    """Compute triple-barrier label for a single entry point.

    Three barriers:
      Upper (profit-take): entry_price + pt_multiplier * volatility
      Lower (stop-loss):   entry_price - sl_multiplier * volatility
      Vertical (time):     max_holding_period bars after entry

    Returns:
       1 if upper barrier hit first
      -1 if lower barrier hit first
       0 if time expired without hitting either barrier
    """
    entry_price = prices.iloc[entry_idx]
    upper = entry_price + pt_multiplier * volatility
    lower = entry_price - sl_multiplier * volatility

    end_idx = min(entry_idx + max_holding_period, len(prices) - 1)

    for i in range(entry_idx + 1, end_idx + 1):
        if prices.iloc[i] >= upper:
            return 1
        if prices.iloc[i] <= lower:
            return -1

    return 0


def label_series(
    prices: pd.Series,
    entry_indices: list[int],
    pt_multiplier: float,
    sl_multiplier: float,
    max_holding_period: int,
    atr_series: pd.Series,
) -> pd.DataFrame:
    """Apply triple-barrier labeling to multiple entry points.

    Uses ATR at each entry point as the volatility estimate for barrier width.

    Returns DataFrame with columns: [entry_idx, entry_price, label,
    exit_idx, exit_price, holding_period].
    """
    rows = []
    for idx in entry_indices:
        if idx >= len(prices) or idx >= len(atr_series):
            continue

        entry_price = prices.iloc[idx]
        vol = atr_series.iloc[idx]
        upper = entry_price + pt_multiplier * vol
        lower = entry_price - sl_multiplier * vol
        end_idx = min(idx + max_holding_period, len(prices) - 1)

        label = 0
        exit_idx = end_idx
        for i in range(idx + 1, end_idx + 1):
            if prices.iloc[i] >= upper:
                label = 1
                exit_idx = i
                break
            if prices.iloc[i] <= lower:
                label = -1
                exit_idx = i
                break

        rows.append({
            "entry_idx": idx,
            "entry_price": entry_price,
            "label": label,
            "barrier_hit": {1: "upper", -1: "lower", 0: "time"}[label],
            "exit_idx": exit_idx,
            "exit_price": prices.iloc[exit_idx],
            "holding_period": exit_idx - idx,
        })

    return pd.DataFrame(rows)


class MetaLabeler:
    """Two-stage meta-labeling: direction from primary, sizing from secondary.

    The secondary model's probability output [0, 1] maps to bet size:
        bet_size = kelly_fraction * probability

    Args:
        kelly_fraction: From calibration (default 0.5 = half-Kelly).
        threshold: Minimum probability to take a trade (default 0.5).
    """

    def __init__(self, kelly_fraction: float = 0.5, threshold: float = 0.5):
        self.kelly_fraction = kelly_fraction
        self.threshold = threshold
        self._model: Any = None
        self._is_fitted = False

    def fit(
        self,
        signals_df: pd.DataFrame,
        prices: pd.Series,
        atr_series: pd.Series,
        features_df: pd.DataFrame,
        pt_multiplier: float = 2.0,
        sl_multiplier: float = 2.0,
        max_holding_period: int = 20,
    ) -> "MetaLabeler":
        """Train the secondary classifier on historical signal outcomes.

        Args:
            signals_df: DataFrame with columns [timestamp, signal_direction, entry_idx].
            prices: Price series.
            atr_series: ATR series for barrier width.
            features_df: Feature matrix aligned with signal entries.
            pt_multiplier: Profit-take barrier width.
            sl_multiplier: Stop-loss barrier width.
            max_holding_period: Max bars to hold.
        """
        from lightgbm import LGBMClassifier

        # Label each signal with triple-barrier outcome
        labels_df = label_series(
            prices,
            list(signals_df["entry_idx"]),
            pt_multiplier, sl_multiplier,
            max_holding_period, atr_series,
        )

        # Binary target: 1 if profitable (label=1), 0 otherwise
        y = (labels_df["label"] == 1).astype(int)

        # Align features with signal indices
        valid_mask = labels_df["entry_idx"].isin(features_df.index)
        X = features_df.loc[labels_df.loc[valid_mask, "entry_idx"]].reset_index(drop=True)
        y = y.loc[valid_mask].reset_index(drop=True)

        if len(X) < 10:
            logger.warning("MetaLabeler: insufficient data for training (<10 signals)")
            self._is_fitted = False
            return self

        self._model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(X, y)
        self._is_fitted = True
        logger.info(f"MetaLabeler trained on {len(X)} signals")
        return self

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict probability and bet size for each signal.

        Returns DataFrame with [probability, bet_size, take_trade].
        """
        if not self._is_fitted:
            return pd.DataFrame({
                "probability": np.full(len(features_df), 0.5),
                "bet_size": np.full(len(features_df), self.kelly_fraction * 0.5),
                "take_trade": np.ones(len(features_df), dtype=bool),
            })

        proba = self._model.predict_proba(features_df)[:, 1]
        return pd.DataFrame({
            "probability": proba,
            "bet_size": self.kelly_fraction * proba,
            "take_trade": proba >= self.threshold,
        })

    def score(self, features_df: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
        """OOS evaluation metrics."""
        if not self._is_fitted:
            return {"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "auc": 0.5}

        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        proba = self._model.predict_proba(features_df)[:, 1]
        preds = (proba >= self.threshold).astype(int)

        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "auc": roc_auc_score(labels, proba) if len(set(labels)) > 1 else 0.5,
        }
