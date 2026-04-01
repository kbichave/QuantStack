# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
ML Training Service — core training logic extracted from MCP tools.

This module contains the business logic for ML model training, independent
of the MCP server. Both MCP tools and the autonomous orchestrator call
these functions directly.

Layer: ml (L5) — depends only on config, core, data, features, ml.trainer.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.core.features.technical_indicators import TechnicalIndicators
from quantstack.core.labeling.event_labeler import EventLabeler
from quantstack.core.labeling.wave_event_labeler import WaveEventLabeler
from quantstack.core.validation.causal_filter import CausalFilter
from quantstack.data.storage import DataStore
from quantstack.features.enricher import FeatureEnricher, FeatureTiers
from quantstack.ml.trainer import ModelTrainer, TrainingConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(os.getenv("QUANTSTACK_MODELS_DIR", "models"))
_DEFAULT_LOOKBACK_DAYS = 756  # ~3 years


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------


async def train_model(
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
    Train an ML classification model for a symbol.

    This is the service-layer entry point. The MCP tool delegates here,
    and the autonomous orchestrator can call this directly without importing
    from the MCP layer.

    Returns:
        Dict with training results (success, metrics, model_path, etc.).
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
        logger.error(f"[ml.training_service] train_model failed for {symbol}: {e}")
        return {"success": False, "error": str(e), "symbol": symbol}


# ---------------------------------------------------------------------------
# Synchronous training pipeline (runs in a thread)
# ---------------------------------------------------------------------------


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
    store = DataStore()
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
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]

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


def _generate_labels(df, method: str):
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
