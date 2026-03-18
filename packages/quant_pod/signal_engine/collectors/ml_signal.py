# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
ML signal collector — loads trained models and runs inference.

Wraps the quantcore Predictor/SHAPExplainer stack behind the standard
collector interface.  Returns {} if no trained model exists for the symbol
(expected for most symbols until the ML pipeline has been run).  Never raises.
"""

import asyncio
from typing import Any

from loguru import logger

from quantcore.data.storage import DataStore


async def collect_ml_signal(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Run ML inference for *symbol* using the latest trained model.

    Returns a dict with keys:
        ml_prediction     : float 0-1 — predicted probability of positive forward return
        ml_direction      : str — "bullish" (>0.55), "bearish" (<0.45), "neutral"
        ml_confidence     : float 0-1 — calibrated confidence (distance from 0.5)
        ml_model_type     : str — model type used (e.g., "lightgbm")
        ml_top_features   : list[str] — top 3 features by SHAP importance
        ml_model_age_days : int | None — days since model was last trained

    Returns {} if no trained model exists or data is insufficient.
    """
    try:
        return await asyncio.to_thread(_collect_ml_signal_sync, symbol, store)
    except Exception as exc:
        logger.debug(f"[ml_signal] {symbol}: {exc} — returning empty")
        return {}


def _collect_ml_signal_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    """Synchronous ML signal collection — called via asyncio.to_thread."""

    # All imports are deferred because quantcore.models is an optional dependency
    # that requires trained artifacts on disk.
    try:
        from quantcore.models.predictor import Predictor
        from quantcore.models.trainer import TrainingResult
    except ImportError:
        logger.debug("[ml_signal] quantcore.models not importable — skipping")
        return {}

    # --- Load trained model ---
    try:
        training_result = _load_latest_training_result(symbol)
    except Exception:
        return {}  # No trained model — expected and normal

    if training_result is None:
        return {}

    predictor = Predictor(training_result)

    # --- Load features for the current bar ---
    try:
        from quantcore.config.timeframes import Timeframe
        from quantcore.features.momentum import MomentumFeatures
        from quantcore.features.technical_indicators import TechnicalIndicators
        from quantcore.features.volatility import VolatilityFeatures
    except ImportError:
        logger.debug("[ml_signal] quantcore.features not importable — skipping")
        return {}

    daily_df = store.load_ohlcv(symbol, Timeframe.DAILY)
    if daily_df is None or len(daily_df) < 60:
        return {}

    try:
        ti = TechnicalIndicators(Timeframe.DAILY, enable_hilbert=False)
        mf = MomentumFeatures(Timeframe.DAILY)
        vf = VolatilityFeatures(Timeframe.DAILY)

        features_df = ti.compute(daily_df)
        features_df = mf.compute(features_df)
        features_df = vf.compute(features_df)
    except Exception as exc:
        logger.debug(f"[ml_signal] {symbol}: feature computation failed: {exc}")
        return {}

    if features_df.empty:
        return {}

    # --- Predict ---
    try:
        prob = float(predictor.predict_proba(features_df.iloc[[-1]])[0])
    except Exception as exc:
        logger.debug(f"[ml_signal] {symbol}: prediction failed: {exc}")
        return {}

    # --- SHAP feature importance (optional, best-effort) ---
    top_features = _get_top_features(predictor, training_result, features_df)

    direction = "bullish" if prob > 0.55 else ("bearish" if prob < 0.45 else "neutral")

    return {
        "ml_prediction": round(prob, 4),
        "ml_direction": direction,
        "ml_confidence": round(abs(prob - 0.5) * 2, 4),
        "ml_model_type": getattr(training_result, "config", None)
        and getattr(training_result.config, "model_type", "unknown")
        or "unknown",
        "ml_top_features": top_features[:3],
        "ml_model_age_days": getattr(training_result, "age_days", None),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MODEL_DIR = "models"  # relative to project data dir


def _load_latest_training_result(symbol: str) -> Any | None:
    """Attempt to load the latest TrainingResult for *symbol* from disk.

    Returns None if no serialized model is found.  Uses joblib/pickle
    convention: ``models/{symbol}_latest.joblib``.
    """
    import pathlib

    try:
        import joblib
    except ImportError:
        import pickle as joblib  # type: ignore[no-redef]

    candidates = [
        pathlib.Path(_MODEL_DIR) / f"{symbol}_latest.joblib",
        pathlib.Path(_MODEL_DIR) / f"{symbol.lower()}_latest.joblib",
        pathlib.Path(_MODEL_DIR) / f"{symbol}_latest.pkl",
    ]

    for path in candidates:
        if path.exists():
            return joblib.load(path)

    return None


def _get_top_features(
    predictor: Any,
    training_result: Any,
    features_df: Any,
) -> list[str]:
    """Best-effort SHAP top features. Returns [] on any failure."""
    try:
        from quantcore.models.explainer import SHAPExplainer

        explainer = SHAPExplainer(training_result)
        sample = features_df.tail(100)
        explainer.fit(sample)
        importance = explainer.global_importance()
        if importance is not None and hasattr(importance, "head"):
            return list(importance.head(3).index)
    except Exception:
        pass

    # Fallback: use built-in feature_importance from training result
    try:
        fi = training_result.feature_importance
        if fi:
            sorted_feats = sorted(fi, key=fi.get, reverse=True)  # type: ignore[arg-type]
            return sorted_feats[:3]
    except Exception:
        pass

    return []
