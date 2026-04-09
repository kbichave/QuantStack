"""Conformal prediction wrapper for calibrated uncertainty intervals.

Provides distribution-free prediction intervals with guaranteed coverage.
Uses MAPIE when available, falls back to residual-based quantile estimation.
The position_size_scalar method converts interval width into a sizing
multiplier: narrow (confident) intervals scale up, wide intervals scale down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    from mapie.regression import MapieRegressor

    _HAS_MAPIE = True
except ImportError:
    _HAS_MAPIE = False
    logger.info("mapie not installed — conformal predictor will use residual fallback")


@dataclass
class ConformalResult:
    """Container for conformal prediction output.

    Attributes:
        point: Point predictions array of shape (n_samples,).
        intervals: Prediction intervals keyed by coverage level string
            ("80", "90", "95"). Each value is a (lower, upper) tuple of
            arrays with the same shape as *point*.
    """

    point: np.ndarray
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
    )


class ConformalPredictor:
    """Conformal prediction wrapper that calibrates any point-prediction model.

    Parameters:
        base_model: A scikit-learn compatible regressor (must support fit/predict).
        method: MAPIE method — "plus", "base", "minmax", or "cv_plus".
    """

    _ALPHA_TO_LABEL = {0.20: "80", 0.10: "90", 0.05: "95"}

    def __init__(self, base_model: Any, method: str = "plus") -> None:
        self._base_model = base_model
        self._method = method
        self._mapie: Any | None = None
        # Fallback state: sorted absolute residuals from calibration set
        self._residuals: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> ConformalPredictor:
        """Fit conformal predictor on calibration data.

        If MAPIE is available the full conformal+ procedure is used.
        Otherwise we fit the base model and store sorted absolute residuals
        for quantile-based interval construction.
        """
        if _HAS_MAPIE:
            self._mapie = MapieRegressor(
                estimator=self._base_model,
                method=self._method,
            )
            self._mapie.fit(X_cal, y_cal)
            logger.debug(
                "ConformalPredictor fitted via MAPIE ({}) on {} samples",
                self._method,
                len(y_cal),
            )
        else:
            self._base_model.fit(X_cal, y_cal)
            preds = self._base_model.predict(X_cal)
            self._residuals = np.sort(np.abs(y_cal - preds))
            logger.debug(
                "ConformalPredictor fitted via residual fallback on {} samples",
                len(y_cal),
            )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        alpha: list[float] | None = None,
    ) -> ConformalResult:
        """Produce point predictions and calibrated prediction intervals.

        Parameters:
            X: Feature matrix of shape (n_samples, n_features).
            alpha: Miscoverage rates. Defaults to [0.20, 0.10, 0.05]
                corresponding to 80%, 90%, 95% coverage.

        Returns:
            ConformalResult with point estimates and per-coverage intervals.
        """
        if not self._fitted:
            raise RuntimeError("ConformalPredictor.fit() must be called before predict()")

        if alpha is None:
            alpha = [0.20, 0.10, 0.05]

        if _HAS_MAPIE and self._mapie is not None:
            return self._predict_mapie(X, alpha)
        return self._predict_residual(X, alpha)

    def _predict_mapie(
        self, X: np.ndarray, alpha: list[float]
    ) -> ConformalResult:
        """Prediction via MAPIE — returns native conformal intervals."""
        point, intervals_3d = self._mapie.predict(X, alpha=alpha)
        # intervals_3d shape: (n_samples, 2, n_alpha)
        result_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for i, a in enumerate(alpha):
            label = self._ALPHA_TO_LABEL.get(a, str(int((1 - a) * 100)))
            lower = intervals_3d[:, 0, i]
            upper = intervals_3d[:, 1, i]
            result_intervals[label] = (lower, upper)

        return ConformalResult(point=point, intervals=result_intervals)

    def _predict_residual(
        self, X: np.ndarray, alpha: list[float]
    ) -> ConformalResult:
        """Fallback: intervals from sorted calibration residuals."""
        assert self._residuals is not None
        point = self._base_model.predict(X)
        n = len(self._residuals)

        result_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for a in alpha:
            quantile_idx = int(np.ceil((1 - a) * (n + 1))) - 1
            quantile_idx = np.clip(quantile_idx, 0, n - 1)
            q = self._residuals[quantile_idx]
            label = self._ALPHA_TO_LABEL.get(a, str(int((1 - a) * 100)))
            result_intervals[label] = (point - q, point + q)

        return ConformalResult(point=point, intervals=result_intervals)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_coverage(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Compute empirical coverage and average interval width on held-out data.

        Returns:
            Dictionary with keys: coverage_80, coverage_90, coverage_95,
            avg_width_80, avg_width_90, avg_width_95, calibration_ok.
            calibration_ok is True when all empirical coverages are within
            5 percentage points of the nominal level.
        """
        result = self.predict(X_test)
        metrics: dict[str, float] = {}
        all_ok = True

        for label, nominal in [("80", 0.80), ("90", 0.90), ("95", 0.95)]:
            if label not in result.intervals:
                continue
            lower, upper = result.intervals[label]
            covered = (y_test >= lower) & (y_test <= upper)
            empirical = float(np.mean(covered))
            width = float(np.mean(upper - lower))
            metrics[f"coverage_{label}"] = empirical
            metrics[f"avg_width_{label}"] = width
            if abs(empirical - nominal) > 0.05:
                all_ok = False

        metrics["calibration_ok"] = float(all_ok)
        logger.info("Conformal coverage: {}", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def position_size_scalar(
        self, X: np.ndarray, base_alpha: float = 0.10
    ) -> np.ndarray:
        """Convert prediction confidence into a position-size multiplier.

        Narrow intervals (relative to the median width) scale positions up
        to 1.5x, wide intervals scale down to 0.5x. The mapping is linear
        between these bounds.

        Parameters:
            X: Feature matrix.
            base_alpha: Miscoverage rate for the interval used to gauge width.

        Returns:
            Array of scalars in [0.5, 1.5], one per sample.
        """
        result = self.predict(X, alpha=[base_alpha])
        label = self._ALPHA_TO_LABEL.get(
            base_alpha, str(int((1 - base_alpha) * 100))
        )
        lower, upper = result.intervals[label]
        widths = upper - lower

        median_width = float(np.median(widths))
        if median_width <= 0:
            return np.ones(len(X))

        # ratio > 1 means wider than median (less confident)
        ratio = widths / median_width
        # Map: ratio 0 -> 1.5, ratio 1 -> 1.0, ratio 2 -> 0.5
        scalar = 1.5 - 0.5 * ratio
        return np.clip(scalar, 0.5, 1.5)
