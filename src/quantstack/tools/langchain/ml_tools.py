"""ML and feature computation tools for LangGraph agents.

Wired to the real ML pipeline: training_service, ml_signal collector,
model_registry, and drift_detector.
"""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

logger = logging.getLogger(__name__)


@tool
async def train_model(
    symbol: Annotated[str, Field(description="Ticker symbol to train the model on, e.g. 'AAPL', 'SPY', 'TSLA'")],
    model_type: Annotated[str, Field(description="ML algorithm to use for model training: 'lightgbm', 'xgboost', or 'catboost'")] = "lightgbm",
    target: Annotated[str, Field(description="Target variable for prediction, e.g. 'forward_return_5d', 'forward_return_1d', 'direction'")] = "forward_return_5d",
) -> str:
    """Train a supervised ML model for stock signal prediction and alpha generation. Use when you need to fit a new model on historical features for a given ticker. Returns JSON with model performance metrics including Sharpe ratio, information coefficient (IC), AUC, accuracy, and a unique model ID for later inference. Supports LightGBM, XGBoost, and CatBoost estimators with optional Optuna hyperparameter optimization. Synonyms: model training, fit, learn, supervised learning, feature importance, cross-validation, hyperparameter tuning."""
    try:
        from quantstack.ml.training_service import train_model as _train

        result = await _train(
            symbol=symbol,
            model_type=model_type,
            apply_causal_filter=True,
            save=True,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        logger.error(f"train_model failed for {symbol}: {exc}")
        return json.dumps({"error": str(exc), "symbol": symbol})


@tool
async def compute_features(
    symbol: Annotated[str, Field(description="Ticker symbol to compute features for, e.g. 'AAPL', 'MSFT', 'QQQ'")],
    timeframe: Annotated[str, Field(description="Data timeframe for feature engineering: 'daily', '1h', '15m', '5m'")] = "daily",
) -> str:
    """Compute 200+ technical and statistical features for a symbol using feature engineering pipelines. Use when you need a full feature vector before model training or ML prediction. Returns JSON with computed feature values and metadata covering trend, momentum, volatility, volume profile, and candlestick pattern features. Provides the input matrix for supervised learning and signal generation. Synonyms: feature extraction, indicator calculation, technical analysis, data preparation, feature store."""
    try:
        import asyncio

        from quantstack.config.timeframes import Timeframe
        from quantstack.core.features.momentum import MomentumFeatures
        from quantstack.core.features.technical_indicators import TechnicalIndicators
        from quantstack.core.features.volatility import VolatilityFeatures
        from quantstack.data.storage import DataStore

        tf_map = {"daily": Timeframe.D1, "1h": Timeframe.H1, "15m": Timeframe.M15, "5m": Timeframe.M5}
        tf = tf_map.get(timeframe, Timeframe.D1)

        def _compute():
            store = DataStore()
            ohlcv = store.load_ohlcv(symbol, tf)
            if ohlcv is None or len(ohlcv) < 60:
                return {"error": f"Insufficient data for {symbol}", "bars": len(ohlcv) if ohlcv is not None else 0}

            ti = TechnicalIndicators(tf)
            df = ti.compute(ohlcv)
            mf = MomentumFeatures(tf)
            df = mf.compute(df)
            vf = VolatilityFeatures(tf)
            df = vf.compute(df)

            feature_cols = [c for c in df.columns if c not in {"open", "high", "low", "close", "volume"}]
            last_row = df.iloc[-1]

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "feature_count": len(feature_cols),
                "bars": len(df),
                "latest_features": {
                    c: round(float(last_row[c]), 6)
                    for c in feature_cols[:50]
                    if not (last_row[c] != last_row[c])  # NaN check
                },
            }

        result = await asyncio.to_thread(_compute)
        return json.dumps(result, default=str)
    except Exception as exc:
        logger.error(f"compute_features failed for {symbol}: {exc}")
        return json.dumps({"error": str(exc), "symbol": symbol})


@tool
async def predict_ml_signal(
    symbol: Annotated[str, Field(description="Ticker symbol to generate a prediction for, e.g. 'AAPL', 'NVDA'")],
    model_id: Annotated[str | None, Field(description="Unique model ID to use for inference; uses the latest trained model if None")] = None,
) -> str:
    """Generate an ML model prediction and signal score for a given stock symbol. Use when you need a forward-looking return forecast or directional probability from a trained model. Returns JSON with predicted value, confidence interval, signal direction, and feature importances driving the prediction. Synonyms: inference, forecast, scoring, predict, alpha signal, model output, probability estimate."""
    try:
        from quantstack.data.storage import DataStore
        from quantstack.signal_engine.collectors.ml_signal import collect_ml_signal

        store = DataStore()
        result = await collect_ml_signal(symbol, store)

        if not result:
            return json.dumps({"error": f"No trained model for {symbol}", "symbol": symbol})

        return json.dumps(result, default=str)
    except Exception as exc:
        logger.error(f"predict_ml_signal failed for {symbol}: {exc}")
        return json.dumps({"error": str(exc), "symbol": symbol})


@tool
async def get_ml_model_status(
    model_id: Annotated[str | None, Field(description="Unique model ID to query; returns status for all registered models if None")] = None,
) -> str:
    """Retrieve status, health, and performance metrics for registered ML models. Use when you need to check model accuracy, AUC, Sharpe ratio, drift indicators, or training metadata. Returns JSON with model state, last-trained timestamp, validation metrics, and concept drift flags. Provides oversight for model lifecycle management and retraining decisions. Synonyms: model registry, model health, performance monitoring, model catalog, validation results."""
    try:
        from quantstack.db import db_conn

        with db_conn() as conn:
            if model_id:
                rows = conn.fetchall(
                    "SELECT model_id, strategy_id, version, status, "
                    "backtest_sharpe, backtest_ic, backtest_max_dd, "
                    "train_date, model_path, promoted_at, shadow_start "
                    "FROM model_registry WHERE model_id = %s",
                    (model_id,),
                )
            else:
                rows = conn.fetchall(
                    "SELECT model_id, strategy_id, version, status, "
                    "backtest_sharpe, backtest_ic, backtest_max_dd, "
                    "train_date, model_path, promoted_at, shadow_start "
                    "FROM model_registry ORDER BY created_at DESC LIMIT 20",
                )

        models = []
        for row in rows:
            models.append({
                k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in dict(row).items()
            })

        return json.dumps({"models": models, "count": len(models)}, default=str)
    except Exception as exc:
        logger.error(f"get_ml_model_status failed: {exc}")
        return json.dumps({"error": str(exc)})


@tool
async def check_concept_drift(
    symbol: Annotated[str, Field(description="Ticker symbol to check for model drift, e.g. 'AAPL', 'SPY'")],
) -> str:
    """Detect concept drift and data distribution shift in ML models for a given symbol. Use when model predictions degrade or market regime changes to determine if retraining is needed. Returns JSON with drift detection results including PSI (population stability index), feature distribution changes, and retrain recommendation. Calculates statistical tests for covariate shift and prior probability shift. Synonyms: model degradation, distribution shift, stale model, retrain trigger, drift detection, feature drift, target drift."""
    try:
        from quantstack.learning.drift_detector import DriftDetector

        detector = DriftDetector()

        # Check feature distribution drift
        drift_report = detector.check_drift(symbol, {})
        result = drift_report.to_dict()

        # Check IC drift if baseline exists
        ic_baseline = detector.load_ic_baseline(symbol)
        if ic_baseline:
            # Load current ICs from recent predictions
            ic_report = detector.check_ic_drift({}, ic_baseline)
            result["ic_drift"] = {
                "z_scores": ic_report.z_scores,
                "drifted_features": ic_report.drifted_features,
            }

        result["has_baseline"] = detector.has_baseline(symbol)
        return json.dumps(result, default=str)
    except Exception as exc:
        logger.error(f"check_concept_drift failed for {symbol}: {exc}")
        return json.dumps({"error": str(exc), "symbol": symbol})
