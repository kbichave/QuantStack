"""Feature importance protocol: MDI + MDA + SFI with consensus filtering.

Implements the AFML feature importance framework (Chapter 8). No single
importance method is reliable alone:
- MDI is biased toward high-cardinality features
- MDA has high variance
- SFI misses interaction effects

Requiring agreement from 2-of-3 methods reduces both false inclusions
and false exclusions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score


class FeatureImportanceProtocol:
    """AFML feature importance: MDI + MDA + SFI with consensus filtering."""

    def compute_mdi(
        self, model: Any, feature_names: list[str]
    ) -> dict[str, float]:
        """Mean Decrease Impurity from tree model's built-in feature importance.

        Fast but biased toward high-cardinality features.
        """
        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
        else:
            raw = np.zeros(len(feature_names))

        total = raw.sum()
        normed = raw / total if total > 0 else raw
        return {name: float(v) for name, v in zip(feature_names, normed)}

    def compute_mda(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str],
        n_repeats: int = 10,
    ) -> dict[str, float]:
        """Mean Decrease Accuracy via permutation importance on OOS data.

        Shuffles each feature and measures accuracy drop. Unbiased but
        higher variance than MDI.
        """
        baseline = accuracy_score(y_test, model.predict(X_test))
        importances = {}

        rng = np.random.default_rng(42)

        for i, name in enumerate(feature_names):
            drops = []
            for _ in range(n_repeats):
                X_perm = X_test.copy()
                X_perm.iloc[:, i] = rng.permutation(X_perm.iloc[:, i].values)
                perm_score = accuracy_score(y_test, model.predict(X_perm))
                drops.append(baseline - perm_score)
            importances[name] = float(np.mean(drops))

        # Normalize to [0, 1] range
        values = np.array(list(importances.values()))
        total = values.sum()
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def compute_sfi(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Single Feature Importance.

        Trains a separate model on each feature alone and measures OOS
        performance. Captures individual predictive power but misses
        interaction effects.
        """
        from lightgbm import LGBMClassifier

        importances = {}

        for i, name in enumerate(feature_names):
            X_tr_single = X_train.iloc[:, [i]]
            X_te_single = X_test.iloc[:, [i]]

            # Skip constant features
            if X_tr_single.iloc[:, 0].std() < 1e-10:
                importances[name] = 0.0
                continue

            try:
                clf = LGBMClassifier(
                    n_estimators=50, max_depth=2, random_state=42, verbose=-1
                )
                clf.fit(X_tr_single, y_train)
                preds = clf.predict(X_te_single)
                importances[name] = float(accuracy_score(y_test, preds))
            except Exception:
                importances[name] = 0.0

        # Normalize
        values = np.array(list(importances.values()))
        total = values.sum()
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def consensus_filter(
        self,
        mdi: dict[str, float],
        mda: dict[str, float],
        sfi: dict[str, float],
        top_pct: float = 0.5,
    ) -> list[str]:
        """Select features that rank in top_pct by at least 2 of 3 methods.

        Returns list of selected feature names.
        """
        all_features = list(mdi.keys())
        n_top = max(1, int(len(all_features) * top_pct))

        # Rank features by each method (highest importance first)
        def _top_set(scores: dict[str, float]) -> set[str]:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return {name for name, _ in ranked[:n_top]}

        top_mdi = _top_set(mdi)
        top_mda = _top_set(mda)
        top_sfi = _top_set(sfi)

        selected = []
        for feat in all_features:
            votes = sum([
                feat in top_mdi,
                feat in top_mda,
                feat in top_sfi,
            ])
            if votes >= 2:
                selected.append(feat)

        if not selected:
            logger.warning("consensus_filter: no features passed 2-of-3 filter, returning top MDI features")
            selected = list(top_mdi)

        return selected
