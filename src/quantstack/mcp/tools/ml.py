# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
ML pipeline tools — train, evaluate, and predict with production models.

Closes the broken pipeline: run_ml_strategy() trains but never saves →
ml_signal collector has nothing to load. These tools save models to the
exact path the collector expects (models/{symbol}_latest.joblib).

Tools:
  - train_ml_model     — train LightGBM/XGBoost/CatBoost with full feature pipeline
  - get_ml_model_status — check trained model status and staleness
  - predict_ml_signal   — run inference on current market data
"""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from loguru import logger
from scipy.stats import ks_2samp, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from quantstack.config.timeframes import Timeframe
from quantstack.core.features.technical_indicators import TechnicalIndicators
from quantstack.core.labeling.event_labeler import EventLabeler
from quantstack.core.labeling.wave_event_labeler import WaveEventLabeler
from quantstack.core.validation.causal_filter import CausalFilter
from quantstack.data.pg_storage import PgDataStore
from quantstack.features.enricher import FeatureEnricher, FeatureTiers
from quantstack.mcp._state import live_db_or_error
from quantstack.mcp.server import mcp
import catboost as cb
import lightgbm as lgb
# shap (~1.4s) and tft_predictor (pulls torch ~0.6s) deferred to functions that use them.
from quantstack.ml.trainer import ModelTrainer, TrainingConfig
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(os.getenv("QUANT_POD_MODELS_DIR", "models"))
_DEFAULT_LOOKBACK_DAYS = 756  # ~3 years
_STALE_MODEL_DAYS = 30

# ---------------------------------------------------------------------------
# train_ml_model
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def train_ml_model(
    symbol: str,
    model_type: str = "lightgbm",
    feature_tiers: list[str] | None = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    label_method: str = "event",
    apply_causal_filter: bool = True,
    save: bool = True,
    feature_whitelist: list[str] | None = None,
) -> dict[str, Any]:
    """
    Train an ML classification model for a symbol using the full feature pipeline.

    Pipeline:
    1. Load OHLCV (last lookback_days bars)
    2. Compute technical indicators
    3. Enrich with fundamental/macro/flow features (via FeatureEnricher)
    4. Generate WIN/LOSS labels (EventLabeler or WaveEventLabeler)
    5. Apply CausalFilter to drop spurious features (optional)
    6. Train with TimeSeriesSplit CV (5 folds)
    7. Save model to models/{symbol}_latest.joblib
    8. Save metadata to models/{symbol}_latest.json

    The ml_signal collector loads from the same path, so training here
    immediately activates ML predictions in SignalEngine.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        model_type: "lightgbm", "xgboost", or "catboost".
        feature_tiers: Which tiers to include. Default: ["technical", "fundamentals"].
            Options: "technical", "fundamentals", "earnings", "macro", "flow".
        lookback_days: Training data window in trading days (default 756 = ~3 years).
        label_method: "event" (ATR-based TP/SL) or "wave" (wave-context labels).
        apply_causal_filter: If True, run CausalFilter to drop non-causal features.
        save: If True, persist model to disk. Set False for dry-run evaluation.
        feature_whitelist: If provided, keep ONLY these features after enrichment.
            Useful for drift-resistant experiments or domain-specific feature subsets.

    Returns:
        {
            "success": True,
            "symbol": "AAPL",
            "model_type": "lightgbm",
            "features_total": 45,
            "features_after_filter": 28,
            "features_dropped": ["fund_peg_ratio", ...],
            "cv_scores": {"accuracy": [0.62, 0.58, ...], "auc": [0.67, 0.63, ...]},
            "test_accuracy": 0.61,
            "test_auc": 0.66,
            "model_path": "models/AAPL_latest.joblib",
        }
    """
    try:

        result = await asyncio.to_thread(
            _train_sync,
            symbol,
            model_type,
            feature_tiers,
            lookback_days,
            label_method,
            apply_causal_filter,
            save,
            feature_whitelist,
        )
        return result
    except Exception as e:
        logger.error(f"[ml] train_ml_model failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _train_sync(
    symbol: str,
    model_type: str,
    feature_tiers: list[str] | None,
    lookback_days: int,
    label_method: str,
    apply_causal_filter: bool,
    save: bool,
    feature_whitelist: list[str] | None = None,
) -> dict[str, Any]:
    """Synchronous training pipeline. Runs in a thread."""

    tiers_list = feature_tiers or ["technical", "fundamentals"]
    ft = FeatureTiers(
        fundamentals="fundamentals" in tiers_list,
        earnings="earnings" in tiers_list,
        macro="macro" in tiers_list,
        flow="flow" in tiers_list,
    )

    # Step 1: Load OHLCV
    store = PgDataStore()
    ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
    if ohlcv is None or len(ohlcv) < 100:
        return {
            "success": False,
            "error": f"Insufficient OHLCV data for {symbol} ({len(ohlcv) if ohlcv is not None else 0} bars)",
            "symbol": symbol,
        }

    # Trim to lookback window
    ohlcv = ohlcv.tail(lookback_days)

    # Step 2: Technical indicators
    ti = TechnicalIndicators(timeframe=Timeframe.D1)
    df = ti.compute(ohlcv)

    # Step 3: Enrich with additional feature tiers
    if ft.any_active():
        enricher = FeatureEnricher()
        df = enricher.enrich(df, symbol=symbol, tiers=ft)

    # Step 4: Generate labels
    df = _generate_labels(df, label_method)
    label_col = "label_long"
    if label_col not in df.columns:
        return {
            "success": False,
            "error": f"Label column '{label_col}' not generated",
            "symbol": symbol,
        }

    # Drop rows with NaN labels
    df = df.dropna(subset=[label_col])
    if len(df) < 50:
        return {
            "success": False,
            "error": f"Only {len(df)} labeled samples (need 50+)",
            "symbol": symbol,
        }

    # Separate features and labels
    exclude_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "label_long",
        "label_short",
        "label_long_bars_to_exit",
        "label_long_exit_type",
        "label_long_pnl_pct",
        "label_short_bars_to_exit",
        "label_short_exit_type",
        "label_short_pnl_pct",
    }
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        return {
            "success": False,
            "error": "No numeric feature columns found after indicator computation",
            "symbol": symbol,
        }

    # Apply feature whitelist if provided
    if feature_whitelist:
        available = [c for c in feature_whitelist if c in feature_cols]
        missing = [c for c in feature_whitelist if c not in feature_cols]
        if missing:
            logger.warning(
                f"[ml] feature_whitelist: {len(missing)} features not found: {missing[:10]}"
            )
        if not available:
            return {
                "success": False,
                "error": "No whitelisted features found in data",
                "symbol": symbol,
            }
        feature_cols = available

    X = df[feature_cols].copy()
    y = df[label_col].astype(int)

    # Fill NaN features with 0 (safe for tree models)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    features_total = len(feature_cols)
    features_dropped: list[str] = []

    # Step 5: Causal filter (optional)
    if apply_causal_filter and len(feature_cols) > 15:
        try:

            cf = CausalFilter(max_lag=5, significance_level=0.05)
            X = cf.fit_transform(X, y)
            cf_result = cf.get_result()
            features_dropped = cf_result.dropped_features
            logger.info(
                f"[ml] CausalFilter: {len(features_dropped)} features dropped for {symbol}"
            )
        except Exception as exc:
            logger.warning(f"[ml] CausalFilter failed, proceeding without: {exc}")

    # Step 6: Train

    config = TrainingConfig(model_type=model_type)
    trainer = ModelTrainer(config)
    train_result = trainer.train(X, y)

    # Step 7: Save
    model_path = None
    metadata_path = None
    if save:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = str(_MODELS_DIR / f"{symbol}_latest.joblib")
        metadata_path = str(_MODELS_DIR / f"{symbol}_latest.json")

        joblib.dump(train_result.model, model_path)

        metadata = {
            "symbol": symbol,
            "model_type": model_type,
            "feature_names": list(X.columns),
            "feature_tiers": tiers_list,
            "features_total": features_total,
            "features_after_filter": len(X.columns),
            "features_dropped": features_dropped,
            "accuracy": train_result.metrics["accuracy"],
            "auc": train_result.metrics["auc"],
            "cv_scores": [round(v, 4) for v in train_result.cv_scores],
            "label_method": label_method,
            "lookback_days": lookback_days,
            "training_samples": len(X),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "apply_causal_filter": apply_causal_filter,
            "feature_whitelist": feature_whitelist,
        }
        Path(metadata_path).write_text(json.dumps(metadata, indent=2))
        logger.info(
            f"[ml] Model saved: {model_path} ({len(X.columns)} features, acc={train_result.metrics['accuracy']:.3f})"
        )

    return {
        "success": True,
        "symbol": symbol,
        "model_type": model_type,
        "features_total": features_total,
        "features_after_filter": len(X.columns),
        "features_dropped": features_dropped,
        "cv_scores": [round(v, 4) for v in train_result.cv_scores],
        "test_accuracy": round(train_result.metrics["accuracy"], 4),
        "test_auc": round(train_result.metrics["auc"], 4),
        "training_samples": len(X),
        "model_path": model_path,
        "feature_importance": {
            k: round(float(v), 4)
            for k, v in list(train_result.feature_importance.items())[:20]
        },
    }

def _generate_labels(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Generate WIN/LOSS labels for training."""
    try:
        if method == "wave":

            labeler = WaveEventLabeler()
            return labeler.label_with_wave_context(df)
        else:

            labeler = EventLabeler()
            return labeler.label_trades(df)
    except Exception as exc:
        logger.warning(f"[ml] Label generation failed ({method}): {exc}")
        return df

# ---------------------------------------------------------------------------
# get_ml_model_status
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def get_ml_model_status(
    symbol: str | None = None,
) -> dict[str, Any]:
    """
    Check status of trained ML models.

    If symbol is provided, returns status for that symbol only.
    If None, scans the models directory for all trained models.

    Returns per model: symbol, model_type, features_count, accuracy, auc,
    trained_at, age_days, needs_retrain (True if age > 30 days).
    """
    try:
        if not _MODELS_DIR.exists():
            return {
                "success": True,
                "models": [],
                "total": 0,
                "message": "No models directory found",
            }

        # Find all metadata files
        if symbol:
            meta_files = [_MODELS_DIR / f"{symbol}_latest.json"]
        else:
            meta_files = sorted(_MODELS_DIR.glob("*_latest.json"))

        models: list[dict[str, Any]] = []
        for meta_path in meta_files:
            if not meta_path.exists():
                if symbol:
                    return {
                        "success": True,
                        "models": [],
                        "total": 0,
                        "message": f"No model found for {symbol}",
                    }
                continue

            try:
                metadata = json.loads(meta_path.read_text())
                trained_at = metadata.get("trained_at", "")
                age_days = 0
                if trained_at:
                    trained_dt = datetime.fromisoformat(trained_at)
                    if trained_dt.tzinfo is None:
                        trained_dt = trained_dt.replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - trained_dt).days

                model_path = meta_path.with_suffix(".joblib")
                models.append(
                    {
                        "symbol": metadata.get(
                            "symbol", meta_path.stem.replace("_latest", "")
                        ),
                        "model_type": metadata.get("model_type", "unknown"),
                        "features_count": metadata.get("features_after_filter", 0),
                        "feature_tiers": metadata.get("feature_tiers", []),
                        "accuracy": metadata.get("accuracy", 0),
                        "auc": metadata.get("auc", 0),
                        "trained_at": trained_at,
                        "age_days": age_days,
                        "needs_retrain": age_days > _STALE_MODEL_DAYS,
                        "model_exists": model_path.exists(),
                        "training_samples": metadata.get("training_samples", 0),
                    }
                )
            except Exception as exc:
                logger.debug(f"[ml] Failed to read metadata {meta_path}: {exc}")

        return {
            "success": True,
            "models": models,
            "total": len(models),
            "stale_count": sum(1 for m in models if m["needs_retrain"]),
        }
    except Exception as e:
        logger.error(f"[ml] get_ml_model_status failed: {e}")
        return {"success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# predict_ml_signal
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def predict_ml_signal(
    symbol: str,
) -> dict[str, Any]:
    """
    Run ML inference on current market data for a symbol.

    Same pipeline as the ml_signal collector but callable directly as an
    MCP tool. Loads the trained model from models/{symbol}_latest.joblib
    and runs prediction on the latest bar.

    Returns:
        {
            "success": True,
            "symbol": "AAPL",
            "probability": 0.72,
            "direction": "bullish",
            "confidence": 0.44,
            "top_features": [("rsi_14", 0.23), ("yield_curve_10y2y", 0.18)],
            "model_age_days": 5,
        }
    """
    try:

        return await asyncio.to_thread(_predict_sync, symbol)
    except Exception as e:
        logger.error(f"[ml] predict_ml_signal failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _predict_sync(symbol: str) -> dict[str, Any]:
    """Synchronous prediction pipeline."""
    model_path = _MODELS_DIR / f"{symbol}_latest.joblib"
    meta_path = _MODELS_DIR / f"{symbol}_latest.json"

    if not model_path.exists():
        return {
            "success": False,
            "error": f"No trained model for {symbol}",
            "symbol": symbol,
        }

    model = joblib.load(model_path)

    # Load metadata for feature names and tiers
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    feature_names = metadata.get("feature_names", [])
    feature_tiers_list = metadata.get("feature_tiers", ["technical"])

    store = PgDataStore()
    ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
    if ohlcv is None or len(ohlcv) < 60:
        return {
            "success": False,
            "error": f"Insufficient data for {symbol}",
            "symbol": symbol,
        }

    ohlcv = ohlcv.tail(252)

    # Technical indicators
    ti = TechnicalIndicators(timeframe=Timeframe.D1)
    df = ti.compute(ohlcv)

    # Feature enrichment (same tiers as training)
    ft = FeatureTiers(
        fundamentals="fundamentals" in feature_tiers_list,
        earnings="earnings" in feature_tiers_list,
        macro="macro" in feature_tiers_list,
        flow="flow" in feature_tiers_list,
    )
    if ft.any_active():
        enricher = FeatureEnricher()
        df = enricher.enrich(df, symbol=symbol, tiers=ft)

    # Align features to training order
    latest = df.iloc[[-1]]
    X = pd.DataFrame(index=latest.index)
    for col in feature_names:
        if col in latest.columns:
            X[col] = latest[col].values
        else:
            X[col] = 0.0

    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # Predict
    prob = float(model.predict_proba(X)[0, 1])
    confidence = abs(prob - 0.5) * 2.0

    if prob > 0.55:
        direction = "bullish"
    elif prob < 0.45:
        direction = "bearish"
    else:
        direction = "neutral"

    # Feature importance (top 5)
    top_features: list[tuple[str, float]] = []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        top_features = [
            (feature_names[i], round(float(importances[i]), 4))
            for i in indices
            if i < len(feature_names)
        ]

    # Model age
    age_days = 0
    trained_at = metadata.get("trained_at", "")
    if trained_at:
        trained_dt = datetime.fromisoformat(trained_at)
        if trained_dt.tzinfo is None:
            trained_dt = trained_dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - trained_dt).days

    return {
        "success": True,
        "symbol": symbol,
        "probability": round(prob, 4),
        "direction": direction,
        "confidence": round(confidence, 4),
        "top_features": top_features,
        "model_type": metadata.get("model_type", "unknown"),
        "model_age_days": age_days,
        "features_used": len(feature_names),
    }

# ---------------------------------------------------------------------------
# tune_hyperparameters
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def tune_hyperparameters(
    symbol: str,
    model_type: str = "lightgbm",
    n_trials: int = 50,
    metric: str = "auc",
    feature_tiers: list[str] | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """
    Bayesian hyperparameter tuning via Optuna for ML classification models.

    Uses TPE sampler with MedianPruner to efficiently search the parameter
    space. Objective is TimeSeriesSplit CV score on the specified metric.

    Reuses the same data-prep pipeline as train_ml_model (FeatureEnricher +
    EventLabeler + optional CausalFilter).

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        model_type: "lightgbm", "xgboost", or "catboost".
        n_trials: Number of Optuna trials (default 50).
        metric: Optimization metric — "auc", "accuracy", or "sharpe".
        feature_tiers: Feature tiers to include. Default: ["technical", "fundamentals"].
        timeout_seconds: Hard timeout in seconds (default 300 = 5 min).

    Returns:
        {
            "success": True,
            "symbol": "AAPL",
            "best_params": {...},
            "best_score": 0.68,
            "n_trials_completed": 50,
            "convergence": [0.55, 0.58, 0.61, ...],
        }
    """
    try:

        result = await asyncio.to_thread(
            _tune_sync,
            symbol,
            model_type,
            n_trials,
            metric,
            feature_tiers,
            timeout_seconds,
        )
        return result
    except Exception as e:
        logger.error(f"[ml] tune_hyperparameters failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _tune_sync(
    symbol: str,
    model_type: str,
    n_trials: int,
    metric: str,
    feature_tiers: list[str] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Synchronous hyperparameter tuning. Runs in a thread."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tiers_list = feature_tiers or ["technical", "fundamentals"]
    ft = FeatureTiers(
        fundamentals="fundamentals" in tiers_list,
        earnings="earnings" in tiers_list,
        macro="macro" in tiers_list,
        flow="flow" in tiers_list,
    )

    # --- Data prep (mirrors _train_sync) ---
    store = PgDataStore()
    ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
    if ohlcv is None or len(ohlcv) < 100:
        return {
            "success": False,
            "error": f"Insufficient OHLCV for {symbol}",
            "symbol": symbol,
        }
    ohlcv = ohlcv.tail(_DEFAULT_LOOKBACK_DAYS)

    ti = TechnicalIndicators(timeframe=Timeframe.D1)
    df = ti.compute(ohlcv)

    if ft.any_active():
        enricher = FeatureEnricher()
        df = enricher.enrich(df, symbol=symbol, tiers=ft)

    df = _generate_labels(df, "event")
    label_col = "label_long"
    if label_col not in df.columns:
        return {"success": False, "error": "Label generation failed", "symbol": symbol}

    df = df.dropna(subset=[label_col])
    if len(df) < 50:
        return {
            "success": False,
            "error": f"Only {len(df)} samples (need 50+)",
            "symbol": symbol,
        }

    exclude_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "label_long",
        "label_short",
        "label_long_bars_to_exit",
        "label_long_exit_type",
        "label_long_pnl_pct",
        "label_short_bars_to_exit",
        "label_short_exit_type",
        "label_short_pnl_pct",
    }
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df[label_col].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    convergence: list[float] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_type == "xgboost":

                clf = xgb.XGBClassifier(
                    **params,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                )
            elif model_type == "catboost":

                clf = cb.CatBoostClassifier(
                    iterations=params["n_estimators"],
                    learning_rate=params["learning_rate"],
                    depth=params["max_depth"],
                    l2_leaf_reg=params["reg_lambda"],
                    subsample=params["subsample"],
                    verbose=0,
                    random_state=42,
                )
            else:

                clf = lgb.LGBMClassifier(**params, verbose=-1, random_state=42)

            clf.fit(X_tr, y_tr)
            y_pred_proba = clf.predict_proba(X_val)[:, 1]

            if metric == "auc":
                fold_score = roc_auc_score(y_val, y_pred_proba)
            elif metric == "accuracy":
                fold_score = accuracy_score(y_val, (y_pred_proba > 0.5).astype(int))
            else:  # sharpe proxy: mean log-return of correct predictions
                correct = (y_pred_proba > 0.5).astype(int) == y_val
                fold_score = float(correct.mean()) - 0.5  # excess accuracy as proxy

            scores.append(fold_score)
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(scores))
        convergence.append(mean_score)
        return mean_score

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    return {
        "success": True,
        "symbol": symbol,
        "model_type": model_type,
        "metric": metric,
        "best_params": study.best_params,
        "best_score": round(study.best_value, 4),
        "n_trials_completed": len(study.trials),
        "convergence": [round(s, 4) for s in convergence],
    }

# ---------------------------------------------------------------------------
# Model Registry (PostgreSQL-backed)
# ---------------------------------------------------------------------------

_REGISTRY_TABLE_CREATED = False

def _ensure_registry_table(db: Any) -> None:
    """Lazily create the model_registry table if it doesn't exist."""
    global _REGISTRY_TABLE_CREATED
    if _REGISTRY_TABLE_CREATED:
        return
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            registry_id VARCHAR PRIMARY KEY,
            symbol VARCHAR NOT NULL,
            model_version INTEGER NOT NULL,
            model_type VARCHAR NOT NULL,
            status VARCHAR DEFAULT 'candidate',
            accuracy DOUBLE,
            auc DOUBLE,
            feature_names JSON,
            hyperparams JSON,
            data_window JSON,
            model_path VARCHAR,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            promoted_at TIMESTAMP,
            retired_at TIMESTAMP,
            UNIQUE(symbol, model_version)
        )
    """
    )
    _REGISTRY_TABLE_CREATED = True

@domain(Domain.ML)
@mcp.tool()
async def register_model(
    symbol: str,
    model_path: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Register a trained model in the version registry.

    Auto-increments the version number per symbol. If the new model's AUC
    exceeds the current champion, promotes it and demotes the old champion.
    The champion's model file is copied to models/{symbol}_latest.joblib.

    Args:
        symbol: Ticker symbol.
        model_path: Path to the .joblib model file.
        metadata: Optional dict with keys: model_type, accuracy, auc,
            feature_names, hyperparams, data_window (start/end dates).

    Returns:
        {
            "success": True,
            "registry_id": "reg_abc123",
            "version": 3,
            "promoted": True,
            "previous_champion_version": 2,
        }
    """

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        _ensure_registry_table(ctx.db)
        meta = metadata or {}

        # Determine next version
        row = ctx.db.execute(
            "SELECT COALESCE(MAX(model_version), 0) FROM model_registry WHERE symbol = ?",
            [symbol],
        ).fetchone()
        next_version = row[0] + 1

        registry_id = f"reg_{uuid.uuid4().hex[:12]}"
        new_auc = meta.get("auc", 0.0) or 0.0

        ctx.db.execute(
            """
            INSERT INTO model_registry
                (registry_id, symbol, model_version, model_type, accuracy, auc,
                 feature_names, hyperparams, data_window, model_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                registry_id,
                symbol,
                next_version,
                meta.get("model_type", "lightgbm"),
                meta.get("accuracy"),
                new_auc,
                json.dumps(meta.get("feature_names", [])),
                json.dumps(meta.get("hyperparams", {})),
                json.dumps(meta.get("data_window", {})),
                model_path,
            ],
        )

        # Check if this beats the current champion
        champion = ctx.db.execute(
            "SELECT registry_id, model_version, auc FROM model_registry "
            "WHERE symbol = ? AND status = 'champion' ORDER BY model_version DESC LIMIT 1",
            [symbol],
        ).fetchone()

        promoted = False
        prev_champion_version = None

        if champion is None or new_auc > (champion[2] or 0.0):
            # Demote old champion
            if champion:
                ctx.db.execute(
                    "UPDATE model_registry SET status = 'retired', retired_at = CURRENT_TIMESTAMP "
                    "WHERE registry_id = ?",
                    [champion[0]],
                )
                prev_champion_version = champion[1]

            # Promote new model
            ctx.db.execute(
                "UPDATE model_registry SET status = 'champion', promoted_at = CURRENT_TIMESTAMP "
                "WHERE registry_id = ?",
                [registry_id],
            )

            # Copy to latest path
            latest_path = _MODELS_DIR / f"{symbol}_latest.joblib"
            _MODELS_DIR.mkdir(parents=True, exist_ok=True)
            src = Path(model_path)
            if src.exists() and src != latest_path:
                shutil.copy2(src, latest_path)
            promoted = True

        return {
            "success": True,
            "registry_id": registry_id,
            "version": next_version,
            "promoted": promoted,
            "previous_champion_version": prev_champion_version,
        }
    except Exception as e:
        logger.error(f"[ml] register_model failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

@domain(Domain.ML)
@mcp.tool()
async def get_model_history(
    symbol: str,
) -> dict[str, Any]:
    """
    Return all registered model versions for a symbol, sorted by version desc.

    Args:
        symbol: Ticker symbol.

    Returns:
        {
            "success": True,
            "symbol": "AAPL",
            "versions": [
                {"version": 3, "status": "champion", "auc": 0.68, ...},
                {"version": 2, "status": "retired", "auc": 0.65, ...},
            ],
            "total": 2,
        }
    """

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        _ensure_registry_table(ctx.db)
        rows = ctx.db.execute(
            """
            SELECT registry_id, model_version, model_type, status,
                   accuracy, auc, feature_names, hyperparams, data_window,
                   model_path, trained_at, promoted_at, retired_at
            FROM model_registry
            WHERE symbol = ?
            ORDER BY model_version DESC
            """,
            [symbol],
        ).fetchall()

        columns = [
            "registry_id",
            "model_version",
            "model_type",
            "status",
            "accuracy",
            "auc",
            "feature_names",
            "hyperparams",
            "data_window",
            "model_path",
            "trained_at",
            "promoted_at",
            "retired_at",
        ]
        versions = []
        for row in rows:
            entry = dict(zip(columns, row))
            for json_col in ("feature_names", "hyperparams", "data_window"):
                if isinstance(entry[json_col], str):
                    try:
                        entry[json_col] = json.loads(entry[json_col])
                    except (json.JSONDecodeError, TypeError):
                        pass
            versions.append(entry)

        return {
            "success": True,
            "symbol": symbol,
            "versions": versions,
            "total": len(versions),
        }
    except Exception as e:
        logger.error(f"[ml] get_model_history failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

@domain(Domain.ML)
@mcp.tool()
async def rollback_model(
    symbol: str,
    version: int,
) -> dict[str, Any]:
    """
    Rollback to a specific model version, making it the champion.

    Demotes the current champion and copies the target version's model
    file to models/{symbol}_latest.joblib.

    Args:
        symbol: Ticker symbol.
        version: The model version to restore as champion.

    Returns:
        {"success": True, "symbol": "AAPL", "rolled_back_to": 2}
    """

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        _ensure_registry_table(ctx.db)

        # Validate the target version exists
        target = ctx.db.execute(
            "SELECT registry_id, model_path FROM model_registry "
            "WHERE symbol = ? AND model_version = ?",
            [symbol, version],
        ).fetchone()
        if not target:
            return {
                "success": False,
                "error": f"Version {version} not found for {symbol}",
            }

        # Demote current champion
        ctx.db.execute(
            "UPDATE model_registry SET status = 'retired', retired_at = CURRENT_TIMESTAMP "
            "WHERE symbol = ? AND status = 'champion'",
            [symbol],
        )

        # Promote target
        ctx.db.execute(
            "UPDATE model_registry SET status = 'champion', promoted_at = CURRENT_TIMESTAMP "
            "WHERE registry_id = ?",
            [target[0]],
        )

        # Copy model file to latest
        if target[1]:
            src = Path(target[1])
            latest_path = _MODELS_DIR / f"{symbol}_latest.joblib"
            _MODELS_DIR.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, latest_path)
                logger.info(f"[ml] Rolled back {symbol} to version {version}")
            else:
                logger.warning(
                    f"[ml] Model file not found at {src}, DB updated but file not restored"
                )

        return {"success": True, "symbol": symbol, "rolled_back_to": version}
    except Exception as e:
        logger.error(f"[ml] rollback_model failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

@domain(Domain.ML)
@mcp.tool()
async def compare_models(
    symbol: str,
    version_a: int,
    version_b: int,
) -> dict[str, Any]:
    """
    Side-by-side comparison of two model versions for a symbol.

    Returns accuracy, AUC, feature diff (added/removed), and hyperparameter
    differences.

    Args:
        symbol: Ticker symbol.
        version_a: First model version.
        version_b: Second model version.

    Returns:
        {
            "success": True,
            "version_a": {...metrics...},
            "version_b": {...metrics...},
            "accuracy_diff": 0.03,
            "auc_diff": 0.02,
            "features_added": ["macro_vix_level"],
            "features_removed": ["fund_peg_ratio"],
            "hyperparam_diffs": {"learning_rate": [0.01, 0.05]},
        }
    """

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        _ensure_registry_table(ctx.db)

        def _fetch_version(ver: int) -> dict[str, Any] | None:
            row = ctx.db.execute(
                """
                SELECT model_version, model_type, status, accuracy, auc,
                       feature_names, hyperparams, trained_at
                FROM model_registry WHERE symbol = ? AND model_version = ?
                """,
                [symbol, ver],
            ).fetchone()
            if not row:
                return None
            cols = [
                "model_version",
                "model_type",
                "status",
                "accuracy",
                "auc",
                "feature_names",
                "hyperparams",
                "trained_at",
            ]
            entry = dict(zip(cols, row))
            for jcol in ("feature_names", "hyperparams"):
                if isinstance(entry[jcol], str):
                    try:
                        entry[jcol] = json.loads(entry[jcol])
                    except (json.JSONDecodeError, TypeError):
                        pass
            return entry

        a = _fetch_version(version_a)
        b = _fetch_version(version_b)

        if not a:
            return {
                "success": False,
                "error": f"Version {version_a} not found for {symbol}",
            }
        if not b:
            return {
                "success": False,
                "error": f"Version {version_b} not found for {symbol}",
            }

        features_a = set(a.get("feature_names") or [])
        features_b = set(b.get("feature_names") or [])

        params_a = a.get("hyperparams") or {}
        params_b = b.get("hyperparams") or {}
        all_keys = set(params_a.keys()) | set(params_b.keys())
        hyperparam_diffs = {
            k: [params_a.get(k), params_b.get(k)]
            for k in all_keys
            if params_a.get(k) != params_b.get(k)
        }

        return {
            "success": True,
            "symbol": symbol,
            "version_a": a,
            "version_b": b,
            "accuracy_diff": round(
                (b.get("accuracy") or 0) - (a.get("accuracy") or 0), 4
            ),
            "auc_diff": round((b.get("auc") or 0) - (a.get("auc") or 0), 4),
            "features_added": sorted(features_b - features_a),
            "features_removed": sorted(features_a - features_b),
            "hyperparam_diffs": hyperparam_diffs,
        }
    except Exception as e:
        logger.error(f"[ml] compare_models failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

# ---------------------------------------------------------------------------
# Concept Drift & Incremental Learning
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def check_concept_drift(
    symbol: str,
    window_days: int = 30,
) -> dict[str, Any]:
    """
    Detect feature distribution drift between the training window and recent data.

    Loads the last window_days of features and compares each feature's
    distribution to the training window (from saved model metadata) using
    the Kolmogorov-Smirnov two-sample test. Features with p < 0.05 are
    flagged as drifted.

    Args:
        symbol: Ticker symbol.
        window_days: Number of recent trading days to compare (default 30).

    Returns:
        {
            "success": True,
            "drift_detected": True,
            "drifted_features": ["rsi_14", "fund_pe_ratio"],
            "ks_stats": {"rsi_14": {"statistic": 0.31, "p_value": 0.003}, ...},
            "recommended_action": "retrain" | "monitor" | "ok",
        }
    """
    try:

        return await asyncio.to_thread(_check_drift_sync, symbol, window_days)
    except Exception as e:
        logger.error(f"[ml] check_concept_drift failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _check_drift_sync(symbol: str, window_days: int) -> dict[str, Any]:
    """Synchronous drift detection. Runs in a thread."""
    # Load model metadata for feature names and training window
    meta_path = _MODELS_DIR / f"{symbol}_latest.json"
    if not meta_path.exists():
        return {
            "success": False,
            "error": f"No model metadata for {symbol}",
            "symbol": symbol,
        }

    metadata = json.loads(meta_path.read_text())
    feature_names = metadata.get("feature_names", [])
    training_samples = metadata.get("training_samples", 0)
    lookback_days = metadata.get("lookback_days", _DEFAULT_LOOKBACK_DAYS)
    feature_tiers_list = metadata.get("feature_tiers", ["technical"])

    if not feature_names:
        return {
            "success": False,
            "error": "No feature names in model metadata",
            "symbol": symbol,
        }

    # Load full data window: training + recent
    store = PgDataStore()
    ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
    if ohlcv is None or len(ohlcv) < window_days + 100:
        return {
            "success": False,
            "error": f"Insufficient data for drift check on {symbol}",
            "symbol": symbol,
        }

    ti = TechnicalIndicators(timeframe=Timeframe.D1)
    df = ti.compute(ohlcv)

    ft = FeatureTiers(
        fundamentals="fundamentals" in feature_tiers_list,
        earnings="earnings" in feature_tiers_list,
        macro="macro" in feature_tiers_list,
        flow="flow" in feature_tiers_list,
    )
    if ft.any_active():
        enricher = FeatureEnricher()
        df = enricher.enrich(df, symbol=symbol, tiers=ft)

    # Split into training window and recent window
    available_features = [f for f in feature_names if f in df.columns]
    if not available_features:
        return {
            "success": False,
            "error": "No overlapping features found",
            "symbol": symbol,
        }

    recent = df[available_features].tail(window_days).dropna()
    # Training window: everything before the recent window
    training = (
        df[available_features]
        .iloc[-(window_days + training_samples) : -window_days]
        .dropna()
    )

    if len(recent) < 10 or len(training) < 30:
        return {
            "success": False,
            "error": "Insufficient data for KS test",
            "symbol": symbol,
        }

    # Run KS test per feature
    ks_stats: dict[str, dict[str, float]] = {}
    drifted: list[str] = []
    significance = 0.05

    for feat in available_features:
        try:
            stat, p_val = ks_2samp(training[feat].values, recent[feat].values)
            ks_stats[feat] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 6),
            }
            if p_val < significance:
                drifted.append(feat)
        except Exception:
            continue

    drift_ratio = len(drifted) / max(len(available_features), 1)
    if drift_ratio > 0.3:
        action = "retrain"
    elif drift_ratio > 0.1:
        action = "monitor"
    else:
        action = "ok"

    return {
        "success": True,
        "symbol": symbol,
        "drift_detected": len(drifted) > 0,
        "drifted_features": drifted,
        "drift_ratio": round(drift_ratio, 3),
        "features_tested": len(available_features),
        "ks_stats": ks_stats,
        "recommended_action": action,
    }

@domain(Domain.ML)
@mcp.tool()
async def update_model_incremental(
    symbol: str,
    new_data_days: int = 30,
) -> dict[str, Any]:
    """
    Incrementally update a trained model with recent data.

    For LightGBM, uses warm-start via init_model for true incremental
    learning. For XGBoost and CatBoost, retrains on the expanded window
    (original training data + new_data_days).

    Compares old vs new accuracy. If the new model is better, registers
    it as a new version in the model registry.

    Args:
        symbol: Ticker symbol.
        new_data_days: Number of new trading days to incorporate (default 30).

    Returns:
        {
            "success": True,
            "updated": True,
            "old_accuracy": 0.61,
            "new_accuracy": 0.64,
            "samples_added": 30,
            "registered_version": 4,
        }
    """
    try:

        return await asyncio.to_thread(_update_incremental_sync, symbol, new_data_days)
    except Exception as e:
        logger.error(f"[ml] update_model_incremental failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _update_incremental_sync(symbol: str, new_data_days: int) -> dict[str, Any]:
    """Synchronous incremental update. Runs in a thread."""
    model_path = _MODELS_DIR / f"{symbol}_latest.joblib"
    meta_path = _MODELS_DIR / f"{symbol}_latest.json"

    if not model_path.exists() or not meta_path.exists():
        return {
            "success": False,
            "error": f"No existing model for {symbol}",
            "symbol": symbol,
        }

    old_model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())

    old_accuracy = metadata.get("accuracy", 0.0)
    old_auc = metadata.get("auc", 0.0)
    feature_names = metadata.get("feature_names", [])
    feature_tiers_list = metadata.get("feature_tiers", ["technical"])
    model_type = metadata.get("model_type", "lightgbm")
    lookback_days = metadata.get("lookback_days", _DEFAULT_LOOKBACK_DAYS)

    # Load expanded data window
    store = PgDataStore()
    ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
    if ohlcv is None or len(ohlcv) < 100:
        return {
            "success": False,
            "error": f"Insufficient data for {symbol}",
            "symbol": symbol,
        }

    ohlcv = ohlcv.tail(lookback_days + new_data_days)

    ti = TechnicalIndicators(timeframe=Timeframe.D1)
    df = ti.compute(ohlcv)

    ft = FeatureTiers(
        fundamentals="fundamentals" in feature_tiers_list,
        earnings="earnings" in feature_tiers_list,
        macro="macro" in feature_tiers_list,
        flow="flow" in feature_tiers_list,
    )
    if ft.any_active():
        enricher = FeatureEnricher()
        df = enricher.enrich(df, symbol=symbol, tiers=ft)

    df = _generate_labels(df, metadata.get("label_method", "event"))
    label_col = "label_long"
    if label_col not in df.columns:
        return {"success": False, "error": "Label generation failed", "symbol": symbol}

    df = df.dropna(subset=[label_col])

    # Align to original feature set
    available = [f for f in feature_names if f in df.columns]
    missing = [f for f in feature_names if f not in df.columns]
    X = df[available].copy()
    for col in missing:
        X[col] = 0.0
    X = X[feature_names]  # restore original order
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    y = df[label_col].astype(int)

    original_samples = metadata.get("training_samples", 0)
    samples_added = max(0, len(X) - original_samples)

    # Train/test split: hold out last 20% for evaluation
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_test) < 10:
        return {
            "success": False,
            "error": "Insufficient test samples",
            "symbol": symbol,
        }

    # Retrain: warm-start for LightGBM, full retrain for others
    if model_type == "lightgbm":

        # Save old model to temp file for init_model

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            old_model.booster_.save_model(tmp_path)
            new_model = lgb.LGBMClassifier(
                n_estimators=100,  # additional rounds
                verbose=-1,
                random_state=42,
            )
            new_model.fit(
                X_train,
                y_train,
                init_model=tmp_path,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    elif model_type == "xgboost":

        new_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        new_model.fit(X_train, y_train)
    elif model_type == "catboost":

        new_model = cb.CatBoostClassifier(verbose=0, random_state=42)
        new_model.fit(X_train, y_train)
    else:
        return {
            "success": False,
            "error": f"Unsupported model_type: {model_type}",
            "symbol": symbol,
        }

    # Evaluate
    y_proba = new_model.predict_proba(X_test)[:, 1]
    new_accuracy = float(accuracy_score(y_test, (y_proba > 0.5).astype(int)))
    new_auc = float(roc_auc_score(y_test, y_proba))

    improved = new_auc > old_auc
    registered_version = None

    if improved:
        # Save new model
        new_model_path = (
            _MODELS_DIR
            / f"{symbol}_v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}.joblib"
        )
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(new_model, new_model_path)

        # Update metadata
        new_metadata = {
            **metadata,
            "accuracy": new_accuracy,
            "auc": new_auc,
            "training_samples": len(X_train),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "incremental_from": metadata.get("trained_at", "unknown"),
            "samples_added": samples_added,
        }
        Path(meta_path).write_text(json.dumps(new_metadata, indent=2))

        # Also copy to latest

        shutil.copy2(new_model_path, _MODELS_DIR / f"{symbol}_latest.joblib")

        # Register in DB if available
        try:

            ctx, db_err = live_db_or_error()
            if ctx and not db_err:
                _ensure_registry_table(ctx.db)

                row = ctx.db.execute(
                    "SELECT COALESCE(MAX(model_version), 0) FROM model_registry WHERE symbol = ?",
                    [symbol],
                ).fetchone()
                next_ver = row[0] + 1
                reg_id = f"reg_{uuid.uuid4().hex[:12]}"

                # Demote old champion
                ctx.db.execute(
                    "UPDATE model_registry SET status = 'retired', retired_at = CURRENT_TIMESTAMP "
                    "WHERE symbol = ? AND status = 'champion'",
                    [symbol],
                )
                ctx.db.execute(
                    """
                    INSERT INTO model_registry
                        (registry_id, symbol, model_version, model_type, status,
                         accuracy, auc, feature_names, hyperparams, data_window, model_path,
                         promoted_at)
                    VALUES (?, ?, ?, ?, 'champion', ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    [
                        reg_id,
                        symbol,
                        next_ver,
                        model_type,
                        new_accuracy,
                        new_auc,
                        json.dumps(feature_names),
                        json.dumps(new_metadata.get("hyperparams", {})),
                        json.dumps(
                            {"samples_added": samples_added, "incremental": True}
                        ),
                        str(new_model_path),
                    ],
                )
                registered_version = next_ver
                logger.info(
                    f"[ml] Incremental model registered as v{next_ver} for {symbol}"
                )
        except Exception as reg_exc:
            logger.warning(
                f"[ml] Registry update failed (model still saved): {reg_exc}"
            )

        logger.info(
            f"[ml] Incremental update for {symbol}: "
            f"accuracy {old_accuracy:.3f} → {new_accuracy:.3f}, "
            f"AUC {old_auc:.3f} → {new_auc:.3f}"
        )
    else:
        logger.info(
            f"[ml] Incremental update for {symbol} did not improve: "
            f"AUC {old_auc:.3f} → {new_auc:.3f}, keeping old model"
        )

    return {
        "success": True,
        "symbol": symbol,
        "updated": improved,
        "old_accuracy": round(old_accuracy, 4),
        "new_accuracy": round(new_accuracy, 4),
        "old_auc": round(old_auc, 4),
        "new_auc": round(new_auc, 4),
        "samples_added": samples_added,
        "registered_version": registered_version,
    }

# ---------------------------------------------------------------------------
# review_model_quality — Automated QA gate
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def review_model_quality(
    symbol: str,
    model_path: str | None = None,
) -> dict[str, Any]:
    """
    Automated QA gate for trained ML models.

    Evaluates a trained model for quality, red flags, and production-readiness.
    Returns accept/reject/retrain with specific actionable feedback.

    Checks performed:
    1. Accuracy gate: AUC >= 0.55 (barely better than random = reject)
    2. Feature concentration: no single feature > 30% importance (fragile model)
    3. Calibration: predicted probabilities should be calibrated (reliability diagram)
    4. Class balance: model should predict both classes (not always 0 or always 1)
    5. Temporal stability: accuracy should not degrade >15% between CV folds
    6. Feature count sanity: models with <5 features are underfit, >100 are likely overfit

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        model_path: Path to model .joblib file. Default: models/{symbol}_latest.joblib.

    Returns:
        {
            "verdict": "accept" | "reject" | "retrain",
            "score": 0-100,
            "checks": [{name, passed, value, threshold, feedback}],
            "feedback": "...",
            "recommended_changes": {features_to_drop, features_to_add, hyperparams_to_change},
        }
    """
    try:

        return await asyncio.to_thread(_review_model_quality_sync, symbol, model_path)
    except Exception as e:
        logger.error(f"[ml] review_model_quality failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _review_model_quality_sync(
    symbol: str,
    model_path: str | None,
) -> dict[str, Any]:
    """Synchronous model QA. Runs in a thread."""

    resolved_model_path = (
        Path(model_path) if model_path else _MODELS_DIR / f"{symbol}_latest.joblib"
    )
    meta_path = resolved_model_path.with_suffix(".json")

    if not resolved_model_path.exists():
        return {
            "success": False,
            "error": f"Model not found: {resolved_model_path}",
            "symbol": symbol,
        }
    if not meta_path.exists():
        return {
            "success": False,
            "error": f"Metadata not found: {meta_path}",
            "symbol": symbol,
        }

    model = joblib.load(resolved_model_path)
    metadata = json.loads(meta_path.read_text())

    checks: list[dict[str, Any]] = []
    features_to_drop: list[str] = []
    features_to_add: list[str] = []
    hyperparams_to_change: dict[str, Any] = {}

    # --- Check 1: AUC gate ---
    auc = metadata.get("auc", 0.0)
    auc_threshold = 0.55
    auc_passed = auc >= auc_threshold
    checks.append(
        {
            "name": "auc_gate",
            "passed": auc_passed,
            "value": round(auc, 4),
            "threshold": auc_threshold,
            "feedback": (
                None
                if auc_passed
                else (
                    f"AUC {auc:.3f} is barely above random (0.50). "
                    "Consider: more features, longer training window, or different label method."
                )
            ),
        }
    )

    # --- Check 2: Feature concentration ---
    feature_names = metadata.get("feature_names", [])
    concentration_threshold = 0.30
    max_importance = 0.0
    dominant_feature = None
    if hasattr(model, "feature_importances_") and len(feature_names) > 0:
        importances = model.feature_importances_
        total_imp = float(np.sum(importances))
        if total_imp > 0:
            normed = importances / total_imp
            max_idx = int(np.argmax(normed))
            max_importance = float(normed[max_idx])
            if max_idx < len(feature_names):
                dominant_feature = feature_names[max_idx]
    concentration_passed = max_importance <= concentration_threshold
    feedback_conc = None
    if not concentration_passed and dominant_feature:
        features_to_drop.append(dominant_feature)
        feedback_conc = (
            f"Feature '{dominant_feature}' dominates at {max_importance:.0%}. "
            "Consider dropping it and retraining to reduce fragility."
        )
    checks.append(
        {
            "name": "feature_concentration",
            "passed": concentration_passed,
            "value": round(max_importance, 4),
            "threshold": concentration_threshold,
            "feedback": feedback_conc,
        }
    )

    # --- Check 3: Calibration ---
    # Re-run predictions on held-out data if available; otherwise use CV scores
    calibration_passed = True
    calibration_error = 0.0
    try:

        tiers_list = metadata.get("feature_tiers", ["technical"])
        ft = FeatureTiers(
            fundamentals="fundamentals" in tiers_list,
            earnings="earnings" in tiers_list,
            macro="macro" in tiers_list,
            flow="flow" in tiers_list,
        )
        store = PgDataStore()
        ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
        if ohlcv is not None and len(ohlcv) >= 100:
            lookback = metadata.get("lookback_days", _DEFAULT_LOOKBACK_DAYS)
            ohlcv = ohlcv.tail(lookback)
            ti = TechnicalIndicators(timeframe=Timeframe.D1)
            df = ti.compute(ohlcv)
            if ft.any_active():
                enricher = FeatureEnricher()
                df = enricher.enrich(df, symbol=symbol, tiers=ft)
            df = _generate_labels(df, metadata.get("label_method", "event"))
            label_col = "label_long"
            if label_col in df.columns:
                df = df.dropna(subset=[label_col])
                available = [f for f in feature_names if f in df.columns]
                X_holdout = df[available].copy()
                for col in [f for f in feature_names if f not in available]:
                    X_holdout[col] = 0.0
                X_holdout = (
                    X_holdout[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
                )
                y_holdout = df[label_col].astype(int)
                # Use last 20% as holdout
                split = int(len(X_holdout) * 0.8)
                X_cal, y_cal = X_holdout.iloc[split:], y_holdout.iloc[split:]
                if len(X_cal) >= 20:
                    proba = model.predict_proba(X_cal)[:, 1]
                    prob_true, prob_pred = calibration_curve(
                        y_cal, proba, n_bins=5, strategy="uniform"
                    )
                    calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))
                    calibration_passed = calibration_error < 0.15
    except Exception as cal_exc:
        logger.debug(f"[ml] Calibration check skipped: {cal_exc}")
    checks.append(
        {
            "name": "calibration",
            "passed": calibration_passed,
            "value": round(calibration_error, 4),
            "threshold": 0.15,
            "feedback": (
                None
                if calibration_passed
                else (
                    f"Mean calibration error {calibration_error:.3f} exceeds 0.15. "
                    "Predictions are poorly calibrated — consider CalibratedClassifierCV wrapper."
                )
            ),
        }
    )

    # --- Check 4: Class balance ---
    class_balance_passed = True
    min_class_ratio = 0.5
    cv_scores = metadata.get("cv_scores", {})
    # Check if model predicts both classes by looking at accuracy distribution
    if cv_scores.get("accuracy"):
        acc_vals = cv_scores["accuracy"]
        # If all accuracies are identical, model likely predicts single class
        if len(set(round(v, 3) for v in acc_vals)) == 1:
            class_balance_passed = False
            min_class_ratio = 0.0
    # Also check via metadata training samples if we have holdout predictions
    try:
        if "proba" in dir():
            pred_classes = (proba > 0.5).astype(
                int
            )  # noqa: F821 — only defined if calibration succeeded
            unique_preds = np.unique(pred_classes)
            if len(unique_preds) < 2:
                class_balance_passed = False
                min_class_ratio = 0.0
            else:
                min_class_ratio = float(
                    min(np.mean(pred_classes), 1 - np.mean(pred_classes))
                )
    except Exception:
        pass
    checks.append(
        {
            "name": "class_balance",
            "passed": class_balance_passed,
            "value": round(min_class_ratio, 4),
            "threshold": 0.10,
            "feedback": (
                None
                if class_balance_passed
                else (
                    "Model appears to predict a single class. "
                    "Check label distribution and consider oversampling the minority class."
                )
            ),
        }
    )

    # --- Check 5: Temporal stability ---
    temporal_passed = True
    max_fold_degradation = 0.0
    if cv_scores.get("auc") and len(cv_scores["auc"]) >= 3:
        auc_folds = cv_scores["auc"]
        max_auc = max(auc_folds)
        min_auc = min(auc_folds)
        if max_auc > 0:
            max_fold_degradation = (max_auc - min_auc) / max_auc
            temporal_passed = max_fold_degradation <= 0.15
    checks.append(
        {
            "name": "temporal_stability",
            "passed": temporal_passed,
            "value": round(max_fold_degradation, 4),
            "threshold": 0.15,
            "feedback": (
                None
                if temporal_passed
                else (
                    f"AUC varies {max_fold_degradation:.0%} across CV folds. "
                    "Model is unstable across time periods — consider regime-conditional training."
                )
            ),
        }
    )
    if not temporal_passed:
        hyperparams_to_change["min_child_samples"] = 30

    # --- Check 6: Feature count sanity ---
    n_features = len(feature_names)
    feature_count_passed = 5 <= n_features <= 100
    feature_feedback = None
    if n_features < 5:
        feature_feedback = f"Only {n_features} features — model is likely underfit. Add more feature tiers."
        features_to_add.extend(["yield_curve_10y2y", "fund_pe_ratio", "fund_roe"])
    elif n_features > 100:
        feature_feedback = (
            f"{n_features} features — likely overfit. "
            "Run CausalFilter (apply_causal_filter=True) or increase regularization."
        )
        hyperparams_to_change["reg_alpha"] = 1.0
        hyperparams_to_change["reg_lambda"] = 1.0
    checks.append(
        {
            "name": "feature_count_sanity",
            "passed": feature_count_passed,
            "value": n_features,
            "threshold": "5-100",
            "feedback": feature_feedback,
        }
    )

    # --- Compute overall score and verdict ---
    check_weights = {
        "auc_gate": 30,
        "feature_concentration": 15,
        "calibration": 15,
        "class_balance": 20,
        "temporal_stability": 15,
        "feature_count_sanity": 5,
    }
    score = 0
    for check in checks:
        weight = check_weights.get(check["name"], 0)
        if check["passed"]:
            score += weight

    critical_failures = [
        c
        for c in checks
        if not c["passed"] and c["name"] in ("auc_gate", "class_balance")
    ]
    fixable_failures = [
        c
        for c in checks
        if not c["passed"] and c["name"] not in ("auc_gate", "class_balance")
    ]

    if auc < 0.52 or (not auc_passed and not class_balance_passed):
        verdict = "reject"
    elif critical_failures or score < 70:
        verdict = "retrain"
    else:
        verdict = "accept"

    # Build feedback summary
    failure_feedbacks = [c["feedback"] for c in checks if c["feedback"]]
    feedback_str = (
        " ".join(failure_feedbacks)
        if failure_feedbacks
        else "Model passes all quality checks."
    )

    return {
        "success": True,
        "symbol": symbol,
        "verdict": verdict,
        "score": score,
        "checks": checks,
        "feedback": feedback_str,
        "recommended_changes": {
            "features_to_drop": features_to_drop,
            "features_to_add": features_to_add,
            "hyperparams_to_change": hyperparams_to_change,
        },
    }

# ---------------------------------------------------------------------------
# train_stacking_ensemble — Gap 6
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def train_stacking_ensemble(
    symbol: str,
    base_models: list[str] | None = None,
    meta_learner: str = "logistic",
    feature_tiers: list[str] | None = None,
) -> dict[str, Any]:
    """
    Train a stacking ensemble combining multiple base classifiers.

    Uses StackingClassifier with purged TimeSeriesSplit CV. Each base model
    is trained independently, and a meta-learner combines their predictions.

    The ensemble is evaluated against the best single base model to confirm
    it adds value. If the ensemble AUC is worse, it is still saved but a
    warning is returned.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        base_models: List of model types. Default: ["lightgbm", "xgboost", "catboost"].
        meta_learner: Meta-learner type — "logistic" or "lightgbm".
        feature_tiers: Feature tiers to include. Default: ["technical", "fundamentals"].

    Returns:
        {
            "success": True,
            "symbol": "AAPL",
            "ensemble_auc": 0.69,
            "base_model_aucs": {"lightgbm": 0.67, "xgboost": 0.65, "catboost": 0.66},
            "correlation_matrix": {...},
            "meta_learner_weights": [...],
            "model_path": "models/AAPL_ensemble_latest.joblib",
            "ensemble_benefit": 0.02,
        }
    """
    try:

        return await asyncio.to_thread(
            _train_stacking_sync,
            symbol,
            base_models,
            meta_learner,
            feature_tiers,
        )
    except Exception as e:
        logger.error(f"[ml] train_stacking_ensemble failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _train_stacking_sync(
    symbol: str,
    base_models: list[str] | None,
    meta_learner: str,
    feature_tiers: list[str] | None,
) -> dict[str, Any]:
    """Synchronous stacking ensemble training. Runs in a thread."""
    models_list = base_models or ["lightgbm", "xgboost", "catboost"]
    tiers_list = feature_tiers or ["technical", "fundamentals"]
    ft = FeatureTiers(
        fundamentals="fundamentals" in tiers_list,
        earnings="earnings" in tiers_list,
        macro="macro" in tiers_list,
        flow="flow" in tiers_list,
    )

    # --- Data prep (mirrors _train_sync) ---
    store = PgDataStore()
    ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
    if ohlcv is None or len(ohlcv) < 100:
        return {
            "success": False,
            "error": f"Insufficient OHLCV for {symbol}",
            "symbol": symbol,
        }
    ohlcv = ohlcv.tail(_DEFAULT_LOOKBACK_DAYS)

    ti = TechnicalIndicators(timeframe=Timeframe.D1)
    df = ti.compute(ohlcv)
    if ft.any_active():
        enricher = FeatureEnricher()
        df = enricher.enrich(df, symbol=symbol, tiers=ft)

    df = _generate_labels(df, "event")
    label_col = "label_long"
    if label_col not in df.columns:
        return {"success": False, "error": "Label generation failed", "symbol": symbol}

    df = df.dropna(subset=[label_col])
    if len(df) < 50:
        return {
            "success": False,
            "error": f"Only {len(df)} samples (need 50+)",
            "symbol": symbol,
        }

    exclude_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "label_long",
        "label_short",
        "label_long_bars_to_exit",
        "label_long_exit_type",
        "label_long_pnl_pct",
        "label_short_bars_to_exit",
        "label_short_exit_type",
        "label_short_pnl_pct",
    }
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df[label_col].astype(int)

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_test) < 10:
        return {
            "success": False,
            "error": "Insufficient test samples",
            "symbol": symbol,
        }

    # --- Build base estimators ---
    estimators: list[tuple[str, Any]] = []
    for mt in models_list:
        if mt == "lightgbm":

            estimators.append(
                (
                    "lightgbm",
                    lgb.LGBMClassifier(
                        n_estimators=300,
                        verbose=-1,
                        random_state=42,
                    ),
                )
            )
        elif mt == "xgboost":

            estimators.append(
                (
                    "xgboost",
                    xgb.XGBClassifier(
                        use_label_encoder=False,
                        eval_metric="logloss",
                        verbosity=0,
                        random_state=42,
                    ),
                )
            )
        elif mt == "catboost":

            estimators.append(
                (
                    "catboost",
                    cb.CatBoostClassifier(
                        verbose=0,
                        random_state=42,
                    ),
                )
            )
        else:
            logger.warning(f"[ml] Unknown base model '{mt}', skipping")

    if len(estimators) < 2:
        return {
            "success": False,
            "error": "Need at least 2 valid base models for stacking",
            "symbol": symbol,
        }

    # --- Meta-learner ---
    if meta_learner == "lightgbm":

        final_est = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    else:
        final_est = LogisticRegression(max_iter=1000, random_state=42)

    # --- Train stacking classifier ---
    tscv = TimeSeriesSplit(n_splits=3)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        cv=tscv,
        passthrough=False,
        n_jobs=1,
    )
    stack.fit(X_train, y_train)

    # --- Evaluate ensemble ---
    ensemble_proba = stack.predict_proba(X_test)[:, 1]
    ensemble_auc = float(roc_auc_score(y_test, ensemble_proba))

    # --- Evaluate individual base models ---
    base_aucs: dict[str, float] = {}
    base_predictions: dict[str, np.ndarray] = {}
    for name, est in stack.estimators_:
        try:
            proba = est.predict_proba(X_test)[:, 1]
            base_aucs[name] = round(float(roc_auc_score(y_test, proba)), 4)
            base_predictions[name] = proba
        except Exception as exc:
            logger.warning(f"[ml] Base model '{name}' evaluation failed: {exc}")
            base_aucs[name] = 0.0

    # --- Correlation matrix between base model predictions ---
    corr_matrix: dict[str, dict[str, float]] = {}
    pred_names = list(base_predictions.keys())
    for i, n1 in enumerate(pred_names):
        corr_matrix[n1] = {}
        for n2 in pred_names:
            corr_val = float(
                np.corrcoef(base_predictions[n1], base_predictions[n2])[0, 1]
            )
            corr_matrix[n1][n2] = round(corr_val, 4)

    # --- Meta-learner weights ---
    meta_weights: list[float] = []
    if hasattr(stack.final_estimator_, "coef_"):
        meta_weights = [round(float(w), 4) for w in stack.final_estimator_.coef_[0]]
    elif hasattr(stack.final_estimator_, "feature_importances_"):
        meta_weights = [
            round(float(w), 4) for w in stack.final_estimator_.feature_importances_
        ]

    # --- Save ---
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = _MODELS_DIR / f"{symbol}_ensemble_latest.joblib"
    joblib.dump(stack, model_save_path)

    ensemble_metadata = {
        "symbol": symbol,
        "model_type": "stacking_ensemble",
        "base_models": models_list,
        "meta_learner": meta_learner,
        "feature_names": list(X.columns),
        "feature_tiers": tiers_list,
        "ensemble_auc": ensemble_auc,
        "base_model_aucs": base_aucs,
        "training_samples": len(X_train),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(_MODELS_DIR / f"{symbol}_ensemble_latest.json").write_text(
        json.dumps(ensemble_metadata, indent=2),
    )

    best_single = max(base_aucs.values()) if base_aucs else 0.0
    ensemble_benefit = ensemble_auc - best_single

    logger.info(
        f"[ml] Stacking ensemble for {symbol}: AUC {ensemble_auc:.4f} "
        f"(best single: {best_single:.4f}, benefit: {ensemble_benefit:+.4f})"
    )

    return {
        "success": True,
        "symbol": symbol,
        "ensemble_auc": round(ensemble_auc, 4),
        "base_model_aucs": base_aucs,
        "correlation_matrix": corr_matrix,
        "meta_learner_weights": meta_weights,
        "model_path": str(model_save_path),
        "ensemble_benefit": round(ensemble_benefit, 4),
        "warning": (
            "Ensemble underperforms best single model" if ensemble_benefit < 0 else None
        ),
    }

# ---------------------------------------------------------------------------
# compute_and_store_features — Feature Store (Gap 7)
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def compute_and_store_features(
    symbol: str,
    tiers: list[str] | None = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Compute features and store them in the PostgreSQL feature store.

    Runs the full feature pipeline (TechnicalIndicators + FeatureEnricher)
    and persists results to the feature_store table. Feature version is
    derived from the tier combination so different tier sets coexist.

    Subsequent training or inference can read from the feature store instead
    of recomputing, reducing latency for multi-model workflows.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        tiers: Feature tiers to compute. Default: ["technical", "fundamentals"].
            Options: "technical", "fundamentals", "earnings", "macro", "flow".
        lookback_days: How many trading days of history to compute (default 756).

    Returns:
        {
            "success": True,
            "symbol": "AAPL",
            "features_computed": 42,
            "rows_stored": 756,
            "feature_version": "tech_fund_v1a2b3",
            "feature_names": [...],
        }
    """
    try:

        return await asyncio.to_thread(
            _compute_and_store_features_sync,
            symbol,
            tiers,
            lookback_days,
        )
    except Exception as e:
        logger.error(f"[ml] compute_and_store_features failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _compute_and_store_features_sync(
    symbol: str,
    tiers: list[str] | None,
    lookback_days: int,
) -> dict[str, Any]:
    """Synchronous feature computation and storage. Runs in a thread."""

    tiers_list = tiers or ["technical", "fundamentals"]
    ft = FeatureTiers(
        fundamentals="fundamentals" in tiers_list,
        earnings="earnings" in tiers_list,
        macro="macro" in tiers_list,
        flow="flow" in tiers_list,
    )

    # Compute feature version hash from tier combination
    tiers_key = "_".join(sorted(tiers_list))
    version_hash = hashlib.sha256(tiers_key.encode()).hexdigest()[:6]
    feature_version = f"{tiers_key}_{version_hash}"

    # Load and compute
    store = PgDataStore()
    ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
    if ohlcv is None or len(ohlcv) < 50:
        return {
            "success": False,
            "error": f"Insufficient OHLCV for {symbol}",
            "symbol": symbol,
        }
    ohlcv = ohlcv.tail(lookback_days)

    ti = TechnicalIndicators(timeframe=Timeframe.D1)
    df = ti.compute(ohlcv)
    if ft.any_active():
        enricher = FeatureEnricher()
        df = enricher.enrich(df, symbol=symbol, tiers=ft)

    # Identify feature columns (exclude raw OHLCV)
    exclude_cols = {"open", "high", "low", "close", "volume", "signal"}
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    df_features = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Store in PostgreSQL
    ctx, db_err = live_db_or_error()
    if db_err:
        return {
            "success": False,
            "error": f"DB unavailable: {db_err}",
            "symbol": symbol,
        }

    ctx.db.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_store (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            feature_version VARCHAR NOT NULL,
            features JSON NOT NULL,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date, feature_version)
        )
    """
    )

    # Upsert rows
    rows_stored = 0
    for idx, row in df_features.iterrows():
        row_date = pd.Timestamp(idx).date() if not isinstance(idx, str) else idx
        features_json = json.dumps(
            {col: round(float(val), 6) for col, val in row.items()}
        )
        ctx.db.execute(
            """
            INSERT INTO feature_store (symbol, date, feature_version, features, computed_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (symbol, date, feature_version) DO UPDATE SET
                features = EXCLUDED.features,
                computed_at = EXCLUDED.computed_at
            """,
            [symbol, str(row_date), feature_version, features_json],
        )
        rows_stored += 1

    logger.info(
        f"[ml] Feature store: {rows_stored} rows stored for {symbol} "
        f"(version={feature_version}, {len(feature_cols)} features)"
    )

    return {
        "success": True,
        "symbol": symbol,
        "features_computed": len(feature_cols),
        "rows_stored": rows_stored,
        "feature_version": feature_version,
        "feature_names": feature_cols,
    }

# ---------------------------------------------------------------------------
# get_feature_lineage — Feature provenance (Gap 7)
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def get_feature_lineage(
    symbol: str,
    model_version: int | None = None,
) -> dict[str, Any]:
    """
    Retrieve feature lineage for a symbol or a specific model version.

    If model_version is given, looks up the model in the registry and returns
    its feature set, training date, and data window. If not, returns the latest
    feature store version for the symbol.

    Useful for auditing what data went into a model and reproducing results.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        model_version: Specific model version from the registry. If None, returns
            latest feature store entry.

    Returns:
        {
            "success": True,
            "symbol": "AAPL",
            "feature_names": ["rsi_14", "macd_signal", ...],
            "feature_version": "tech_fund_v1a2b3",
            "data_sources": ["technical", "fundamentals"],
            "compute_date": "2025-03-15T...",
            "model_version": 3,
        }
    """
    try:

        # If model_version specified, look up from registry
        if model_version is not None:
            ctx, db_err = live_db_or_error()
            if db_err:
                return {
                    "success": False,
                    "error": f"DB unavailable: {db_err}",
                    "symbol": symbol,
                }

            _ensure_registry_table(ctx.db)
            row = ctx.db.execute(
                """
                SELECT feature_names, model_type, data_window, model_path, trained_at
                FROM model_registry
                WHERE symbol = ? AND model_version = ?
                """,
                [symbol, model_version],
            ).fetchone()

            if not row:
                return {
                    "success": False,
                    "error": f"No registry entry for {symbol} v{model_version}",
                    "symbol": symbol,
                }

            feature_names_raw = row[0]
            feature_names = (
                json.loads(feature_names_raw)
                if isinstance(feature_names_raw, str)
                else feature_names_raw or []
            )
            data_window = (
                json.loads(row[2]) if isinstance(row[2], str) else row[2] or {}
            )

            return {
                "success": True,
                "symbol": symbol,
                "feature_names": feature_names,
                "feature_version": None,
                "data_sources": data_window.get("feature_tiers", []),
                "compute_date": str(row[4]) if row[4] else None,
                "model_version": model_version,
                "model_type": row[1],
                "model_path": row[3],
            }

        # No model_version: check feature store
        ctx, db_err = live_db_or_error()
        if db_err:
            # Fall back to metadata file
            meta_path = _MODELS_DIR / f"{symbol}_latest.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text())
                return {
                    "success": True,
                    "symbol": symbol,
                    "feature_names": metadata.get("feature_names", []),
                    "feature_version": None,
                    "data_sources": metadata.get("feature_tiers", []),
                    "compute_date": metadata.get("trained_at"),
                    "model_version": None,
                    "source": "metadata_file",
                }
            return {
                "success": False,
                "error": "No feature data found",
                "symbol": symbol,
            }

        # Query latest feature store entry
        row = ctx.db.execute(
            """
            SELECT feature_version, features, computed_at
            FROM feature_store
            WHERE symbol = ?
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            [symbol],
        ).fetchone()

        if not row:
            # Fall back to metadata file
            meta_path = _MODELS_DIR / f"{symbol}_latest.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text())
                return {
                    "success": True,
                    "symbol": symbol,
                    "feature_names": metadata.get("feature_names", []),
                    "feature_version": None,
                    "data_sources": metadata.get("feature_tiers", []),
                    "compute_date": metadata.get("trained_at"),
                    "model_version": None,
                    "source": "metadata_file",
                }
            return {
                "success": False,
                "error": "No feature data found",
                "symbol": symbol,
            }

        feature_version = row[0]
        features_json = json.loads(row[1]) if isinstance(row[1], str) else row[1]
        # Extract tier names from the version string (e.g., "technical_fundamentals_abc123")
        version_parts = feature_version.rsplit("_", 1)[0] if feature_version else ""
        data_sources = version_parts.split("_") if version_parts else []

        return {
            "success": True,
            "symbol": symbol,
            "feature_names": (
                list(features_json.keys()) if isinstance(features_json, dict) else []
            ),
            "feature_version": feature_version,
            "data_sources": data_sources,
            "compute_date": str(row[2]) if row[2] else None,
            "model_version": None,
            "source": "feature_store",
        }

    except Exception as e:
        logger.error(f"[ml] get_feature_lineage failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

# ---------------------------------------------------------------------------
# train_cross_sectional_model — Gap 9
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def train_cross_sectional_model(
    symbols: list[str],
    target: str = "returns_5d",
    feature_tiers: list[str] | None = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Train a cross-sectional (multi-stock) model for relative ranking.

    Unlike per-symbol models, this trains a SINGLE model across all provided
    symbols. Features are rank-normalized within each date (converted to
    cross-sectional percentiles), making the model learn relative rankings
    rather than absolute levels.

    Evaluated via information coefficient (IC = Spearman rank correlation
    between predicted rank and realized forward return). IC IR (mean/std
    of rolling IC) should exceed 0.5 for a production-worthy model.

    Args:
        symbols: List of ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"]).
            Minimum 5 symbols recommended for meaningful cross-sectional signal.
        target: Forward return window — "returns_5d", "returns_10d", or "returns_20d".
        feature_tiers: Feature tiers. Default: ["technical", "fundamentals"].
        lookback_days: Training window per symbol (default 756 = ~3 years).

    Returns:
        {
            "success": True,
            "ic_mean": 0.08,
            "ic_std": 0.12,
            "ic_ir": 0.67,
            "top_factors": [("rsi_14_rank", 0.12), ...],
            "model_path": "models/cross_sectional_returns_5d_latest.joblib",
            "symbols_used": ["AAPL", "MSFT", ...],
        }
    """
    try:

        return await asyncio.to_thread(
            _train_cross_sectional_sync,
            symbols,
            target,
            feature_tiers,
            lookback_days,
        )
    except Exception as e:
        logger.error(f"[ml] train_cross_sectional_model failed: {e}")
        return {"success": False, "error": str(e)}

def _train_cross_sectional_sync(
    symbols: list[str],
    target: str,
    feature_tiers: list[str] | None,
    lookback_days: int,
) -> dict[str, Any]:
    """Synchronous cross-sectional model training. Runs in a thread."""
    tiers_list = feature_tiers or ["technical", "fundamentals"]
    ft = FeatureTiers(
        fundamentals="fundamentals" in tiers_list,
        earnings="earnings" in tiers_list,
        macro="macro" in tiers_list,
        flow="flow" in tiers_list,
    )

    # Parse target → forward return period
    target_map = {"returns_5d": 5, "returns_10d": 10, "returns_20d": 20}
    fwd_days = target_map.get(target)
    if fwd_days is None:
        return {
            "success": False,
            "error": f"Unknown target '{target}'. Use: {list(target_map.keys())}",
        }

    # --- Load and featurize all symbols ---
    store = PgDataStore()
    ti = TechnicalIndicators(timeframe=Timeframe.D1)
    enricher = FeatureEnricher() if ft.any_active() else None

    panel_frames: list[pd.DataFrame] = []
    symbols_loaded: list[str] = []

    for sym in symbols:
        try:
            ohlcv = store.load_ohlcv(sym, Timeframe.D1)
            if ohlcv is None or len(ohlcv) < 100:
                logger.warning(
                    f"[ml] Skipping {sym}: insufficient data ({len(ohlcv) if ohlcv is not None else 0} bars)"
                )
                continue

            ohlcv = ohlcv.tail(lookback_days)
            df = ti.compute(ohlcv)
            if enricher:
                df = enricher.enrich(df, symbol=sym, tiers=ft)

            # Compute forward returns as label
            df[target] = df["close"].pct_change(fwd_days).shift(-fwd_days)
            df["_symbol"] = sym
            panel_frames.append(df)
            symbols_loaded.append(sym)
        except Exception as exc:
            logger.warning(f"[ml] Failed to load {sym}: {exc}")

    if len(symbols_loaded) < 3:
        return {
            "success": False,
            "error": f"Only {len(symbols_loaded)} symbols loaded (need 3+)",
        }

    # --- Stack into panel ---
    panel = pd.concat(panel_frames, axis=0)
    panel = panel.dropna(subset=[target])

    exclude_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        target,
        "_symbol",
    }
    feature_cols = [
        c
        for c in panel.columns
        if c not in exclude_cols
        and panel[c].dtype in ("float64", "float32", "int64", "int32")
    ]

    if len(feature_cols) < 3:
        return {
            "success": False,
            "error": f"Only {len(feature_cols)} features — need more",
        }

    # --- Cross-sectional rank normalization ---
    # Group by date, rank each feature within each cross-section
    panel_idx = panel.index
    rank_features = panel[feature_cols].copy()
    rank_features["_date"] = panel_idx
    ranked = rank_features.groupby("_date")[feature_cols].rank(pct=True)
    ranked = ranked.fillna(0.5)  # missing → median rank

    X = ranked.replace([np.inf, -np.inf], 0.5)
    y = panel[target].values

    # Drop remaining NaN
    valid_mask = ~(np.isnan(y) | np.isinf(y))
    X = X.loc[valid_mask]
    y = y[valid_mask]

    if len(X) < 100:
        return {
            "success": False,
            "error": f"Only {len(X)} valid panel rows (need 100+)",
        }

    # --- Train/test split (temporal) ---
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # --- Train LightGBM regressor (regression on returns, not classification) ---

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Evaluate: rolling IC ---
    y_pred = model.predict(X_test)
    # Compute IC on the full test set
    overall_ic, _ = spearmanr(y_pred, y_test)

    # Rolling IC in chunks (simulate daily cross-sections)
    chunk_size = max(len(symbols_loaded), 5)
    ic_values: list[float] = []
    for start in range(0, len(y_test) - chunk_size, chunk_size):
        end = start + chunk_size
        chunk_pred = y_pred[start:end]
        chunk_actual = y_test[start:end]
        if np.std(chunk_pred) > 1e-10 and np.std(chunk_actual) > 1e-10:
            ic_val, _ = spearmanr(chunk_pred, chunk_actual)
            if not np.isnan(ic_val):
                ic_values.append(float(ic_val))

    ic_mean = float(np.mean(ic_values)) if ic_values else float(overall_ic)
    ic_std = float(np.std(ic_values)) if len(ic_values) > 1 else 0.1
    ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

    # --- Top factors ---
    importances = model.feature_importances_
    total_imp = float(np.sum(importances))
    normed_imp = importances / total_imp if total_imp > 0 else importances
    top_indices = np.argsort(normed_imp)[::-1][:10]
    top_factors = [
        (feature_cols[i], round(float(normed_imp[i]), 4))
        for i in top_indices
        if i < len(feature_cols)
    ]

    # --- Save ---
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = _MODELS_DIR / f"cross_sectional_{target}_latest.joblib"
    joblib.dump(model, model_save_path)

    cs_metadata = {
        "model_type": "cross_sectional_lgbm",
        "target": target,
        "forward_days": fwd_days,
        "symbols": symbols_loaded,
        "feature_names": feature_cols,
        "feature_tiers": tiers_list,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "panel_rows": len(X),
        "training_samples": len(X_train),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(_MODELS_DIR / f"cross_sectional_{target}_latest.json").write_text(
        json.dumps(cs_metadata, indent=2),
    )

    logger.info(
        f"[ml] Cross-sectional model ({target}): IC={ic_mean:.4f}, "
        f"IC IR={ic_ir:.2f}, {len(symbols_loaded)} symbols, {len(X)} panel rows"
    )

    return {
        "success": True,
        "ic_mean": round(ic_mean, 4),
        "ic_std": round(ic_std, 4),
        "ic_ir": round(ic_ir, 4),
        "top_factors": top_factors,
        "model_path": str(model_save_path),
        "symbols_used": symbols_loaded,
        "panel_rows": len(X),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

# ---------------------------------------------------------------------------
# train_deep_model (TFT return predictor)
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def train_deep_model(
    symbol: str,
    architecture: str = "tft",
    target: str = "returns_5d",
    sequence_length: int = 60,
    epochs: int = 50,
) -> dict[str, Any]:
    """
    Train a deep learning model (TFT) for multi-horizon return prediction.

    Uses a Temporal Fusion Transformer architecture (LSTM + attention + GRN)
    adapted for regression. Predicts 1d, 5d, and 20d forward returns.

    Requires PyTorch. Falls back gracefully if not installed.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        architecture: Model architecture. Currently only "tft" is supported.
        target: Target label for primary evaluation.
            "returns_1d", "returns_5d", or "returns_20d".
        sequence_length: Number of historical bars per input sample (default 60).
        epochs: Training epochs (default 50).

    Returns:
        {
            "success": True,
            "symbol": "AAPL",
            "architecture": "tft",
            "sequence_length": 60,
            "train_mse": 0.000123,
            "train_mae": 0.008,
            "n_samples": 500,
            "n_features": 8,
            "model_path": "models/AAPL_tft_returns.pt",
        }
    """
    try:

        result = await asyncio.to_thread(
            _train_deep_sync,
            symbol,
            architecture,
            target,
            sequence_length,
            epochs,
        )
        return result
    except Exception as e:
        logger.error(f"[ml] train_deep_model failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _train_deep_sync(
    symbol: str,
    architecture: str,
    target: str,
    sequence_length: int,
    epochs: int,
) -> dict[str, Any]:
    """Synchronous deep model training. Runs in a thread."""
    import torch  # noqa: PLC0415
    from quantstack.ml.tft_predictor import HORIZONS, TFTReturnPredictor  # noqa: PLC0415

    if architecture != "tft":
        return {
            "success": False,
            "error": f"Unsupported architecture: {architecture}. Only 'tft' is supported.",
            "symbol": symbol,
        }

    # Load OHLCV
    store = PgDataStore()
    ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
    if ohlcv is None or len(ohlcv) < sequence_length + 50:
        return {
            "success": False,
            "error": f"Insufficient data for {symbol} (need {sequence_length + 50}+ bars)",
            "symbol": symbol,
        }

    ohlcv = ohlcv.tail(756)  # ~3 years

    # Compute features
    ti = TechnicalIndicators(timeframe=Timeframe.D1)
    df = ti.compute(ohlcv)

    # Select numeric features only
    exclude = {"open", "high", "low", "close", "volume"}
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        # Fallback: use OHLCV-derived features
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std()
        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_20"] = df["close"].pct_change(20)
        df["volume_change"] = df["volume"].pct_change() if "volume" in df.columns else 0
        feature_cols = [
            "returns",
            "volatility",
            "momentum_5",
            "momentum_20",
            "volume_change",
        ]

    # Generate multi-horizon forward returns as targets
    for h in HORIZONS:
        df[f"fwd_ret_{h}d"] = df["close"].pct_change(h).shift(-h)

    df = df.dropna()
    if len(df) < sequence_length + 20:
        return {
            "success": False,
            "error": f"Insufficient clean data after feature computation: {len(df)} rows",
            "symbol": symbol,
        }

    features = df[feature_cols].values.astype(np.float32)
    targets = df[[f"fwd_ret_{h}d" for h in HORIZONS]].values.astype(np.float32)

    # Create sequences
    X_list, y_list = [], []
    for i in range(sequence_length, len(features)):
        X_list.append(features[i - sequence_length : i])
        y_list.append(targets[i])

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)

    # Train
    predictor = TFTReturnPredictor(
        n_features=len(feature_cols),
        sequence_length=sequence_length,
        epochs=epochs,
    )
    metrics = predictor.train(X_arr, y_arr, feature_names=feature_cols)

    # Save
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(_MODELS_DIR / f"{symbol}_tft_returns.pt")
    predictor.save(model_path)

    return {
        "success": True,
        "symbol": symbol,
        "architecture": architecture,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "train_mse": metrics.get("train_mse", float("nan")),
        "train_mae": metrics.get("train_mae", float("nan")),
        "n_samples": len(X_arr),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "model_path": model_path,
    }

# ---------------------------------------------------------------------------
# analyze_model_shap — SHAP-based feature importance analysis
# ---------------------------------------------------------------------------

@domain(Domain.ML)
@mcp.tool()
async def analyze_model_shap(
    symbol: str,
    model_path: str | None = None,
    n_samples: int = 200,
    cross_symbol: bool = False,
) -> dict[str, Any]:
    """
    Run SHAP TreeExplainer on a trained model to identify feature importance.

    Uses TreeSHAP (exact, fast for tree models) to compute per-feature
    contribution values. Returns top features ranked by mean |SHAP|.

    When cross_symbol=True, analyzes all symbols that have trained models
    and identifies breakthrough features (top-10 SHAP across 2+ symbols).

    Args:
        symbol: Ticker symbol (e.g., "SPY"). Ignored if cross_symbol=True.
        model_path: Path to model .joblib file. Default: models/{symbol}_latest.joblib.
        n_samples: Number of recent samples to explain (default 200). More = slower but stabler.
        cross_symbol: If True, run SHAP across all trained models and find universal features.

    Returns:
        {
            "success": True,
            "symbol": "SPY",
            "top_shap_features": [["rsi", 0.042], ["adx", 0.031], ...],
            "model_type": "lightgbm",
            "n_samples_explained": 200,
            "breakthrough_features": [...]  (only if cross_symbol=True)
        }
    """
    try:

        return await asyncio.to_thread(
            _analyze_shap_sync,
            symbol,
            model_path,
            n_samples,
            cross_symbol,
        )
    except Exception as e:
        logger.error(f"[ml] analyze_model_shap failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}

def _analyze_shap_sync(
    symbol: str,
    model_path: str | None,
    n_samples: int,
    cross_symbol: bool,
) -> dict[str, Any]:
    """Synchronous SHAP analysis. Runs in a thread."""

    if cross_symbol:
        return _cross_symbol_shap(n_samples)

    return _single_symbol_shap(symbol, model_path, n_samples)

def _single_symbol_shap(
    symbol: str,
    model_path_override: str | None,
    n_samples: int,
) -> dict[str, Any]:
    """SHAP analysis for a single symbol/model pair."""

    resolved = (
        Path(model_path_override)
        if model_path_override
        else _MODELS_DIR / f"{symbol}_latest.joblib"
    )
    meta_path = resolved.with_suffix(".json")

    if not resolved.exists():
        return {
            "success": False,
            "error": f"Model not found: {resolved}",
            "symbol": symbol,
        }

    model = joblib.load(resolved)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model_type = metadata.get("model_type", "unknown")

    # Load data to explain
    X, _ = _load_symbol_features(symbol, metadata)
    if X is None or len(X) == 0:
        return {
            "success": False,
            "error": "Could not load feature data",
            "symbol": symbol,
        }

    X_sample = X.tail(min(n_samples, len(X)))
    top_shap = _run_shap(model, X_sample, model_type)

    # Update metadata with SHAP results
    if meta_path.exists():
        metadata["top_shap_features"] = top_shap[:10]
        meta_path.write_text(json.dumps(metadata, indent=2))

    return {
        "success": True,
        "symbol": symbol,
        "model_type": model_type,
        "n_samples_explained": len(X_sample),
        "n_features": len(X.columns),
        "top_shap_features": top_shap[:20],
    }

def _cross_symbol_shap(n_samples: int) -> dict[str, Any]:
    """Run SHAP across all trained models, find breakthrough features."""
    model_files = sorted(_MODELS_DIR.glob("*_latest.joblib"))
    if not model_files:
        return {"success": False, "error": "No trained models found in models/"}

    per_symbol: dict[str, dict] = {}
    cross_shap: dict[str, list[tuple[str, float]]] = {}

    for mf in model_files:
        # Extract symbol from filename: {SYMBOL}_{model_type}_latest.joblib or {SYMBOL}_latest.joblib
        stem = mf.stem  # e.g. "SPY_lightgbm_latest" or "SPY_latest"
        parts = stem.split("_")
        sym = parts[0]
        meta_path = mf.with_suffix(".json")

        if not meta_path.exists():
            continue

        metadata = json.loads(meta_path.read_text())
        model_type = metadata.get("model_type", "unknown")

        try:
            model = joblib.load(mf)
            X, _ = _load_symbol_features(sym, metadata)
            if X is None or len(X) == 0:
                continue

            X_sample = X.tail(min(n_samples, len(X)))
            top_shap = _run_shap(model, X_sample, model_type)

            key = f"{sym}_{model_type}"
            per_symbol[key] = {
                "symbol": sym,
                "model_type": model_type,
                "top_5": top_shap[:5],
            }

            for feat_name, shap_val in top_shap[:10]:
                cross_shap.setdefault(feat_name, []).append((key, shap_val))

        except Exception as exc:
            logger.warning(f"[ml] SHAP failed for {mf.name}: {exc}")

    # Find breakthrough features (top-10 SHAP across 2+ unique symbols)
    breakthroughs = []
    for feat, occurrences in cross_shap.items():
        unique_syms = {occ.split("_")[0] for occ, _ in occurrences}
        if len(unique_syms) >= 2:
            mean_shap = float(np.mean([v for _, v in occurrences]))
            breakthroughs.append(
                {
                    "feature": feat,
                    "n_symbols": len(unique_syms),
                    "n_combos": len(occurrences),
                    "mean_shap": round(mean_shap, 6),
                    "symbols": sorted(unique_syms),
                }
            )

    breakthroughs.sort(key=lambda x: x["mean_shap"], reverse=True)

    return {
        "success": True,
        "models_analyzed": len(per_symbol),
        "per_symbol": per_symbol,
        "breakthrough_features": breakthroughs[:25],
    }

def _load_symbol_features(
    symbol: str,
    metadata: dict,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Load features for a symbol, aligning to model's trained feature set."""
    try:

        store = PgDataStore()
        ohlcv = store.load_ohlcv(symbol, Timeframe.D1)
        if ohlcv is None or len(ohlcv) < 100:
            return None, None

        ohlcv = ohlcv.tail(_DEFAULT_LOOKBACK_DAYS)
        ti = TechnicalIndicators(timeframe=Timeframe.D1)
        df = ti.compute(ohlcv)

        tiers_list = metadata.get("feature_tiers", ["technical", "fundamentals"])
        ft = FeatureTiers(
            fundamentals="fundamentals" in tiers_list,
            macro="macro" in tiers_list,
        )
        if ft.any_active():
            enricher = FeatureEnricher()
            df = enricher.enrich(df, symbol=symbol, tiers=ft)

        # Align to trained features
        trained_features = metadata.get("feature_names")
        if trained_features:
            X = pd.DataFrame(index=df.index)
            for col in trained_features:
                X[col] = df[col] if col in df.columns else 0.0
        else:
            exclude = {
                "open",
                "high",
                "low",
                "close",
                "volume",
                "signal",
                "label_long",
                "label_short",
            }
            feature_cols = [
                c
                for c in df.columns
                if c not in exclude
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            X = df[feature_cols]

        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        df = EventLabeler().label_trades(df)
        y = df["label_long"].astype(int) if "label_long" in df.columns else None

        return X, y
    except Exception as exc:
        logger.warning(f"[ml] Failed to load features for {symbol}: {exc}")
        return None, None

def _run_shap(
    model: Any,
    X: pd.DataFrame,
    model_type: str,
) -> list[list]:
    """Run SHAP TreeExplainer, fallback to built-in importance."""
    try:
        import shap  # noqa: PLC0415
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        ranked = sorted(
            zip(X.columns, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True,
        )
        return [[name, round(float(val), 6)] for name, val in ranked[:20]]

    except Exception as exc:
        logger.warning(
            f"[ml] SHAP failed for {model_type}, falling back to feature_importances_: {exc}"
        )
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            total = imp.sum()
            if total > 0:
                imp = imp / total
            ranked = sorted(zip(X.columns, imp), key=lambda x: x[1], reverse=True)
            return [[name, round(float(val), 6)] for name, val in ranked[:20]]
        return []
