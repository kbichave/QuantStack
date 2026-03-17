# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Regime collector — replaces regime_detector_ic.

Uses WeeklyRegimeClassifier directly on OHLCV already in the local DataStore.
No LLM. No network call. Returns trend regime, volatility regime, and confidence.
"""

import asyncio
from typing import Any

from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.storage import DataStore
from quantcore.hierarchy.regime_classifier import RegimeType, WeeklyRegimeClassifier

_MIN_BARS = 60


async def collect_regime(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Classify market regime for *symbol* from locally stored daily OHLCV.

    Returns a dict with keys:
        trend_regime   : "trending_up" | "trending_down" | "ranging" | "unknown"
        volatility_regime: "low" | "normal" | "high" | "extreme"
        confidence     : float [0, 1]
        regime_label   : "BULL" | "BEAR" | "SIDEWAYS" (raw WeeklyRegimeClassifier output)
        ema_alignment  : int (-1, 0, 1)
        momentum_score : float
        bars_in_regime : int
    """
    try:
        return await asyncio.to_thread(_collect_regime_sync, symbol, store)
    except Exception as exc:
        logger.warning(f"[regime] {symbol}: {exc}")
        return {"trend_regime": "unknown", "volatility_regime": "normal", "confidence": 0.0}


def _collect_regime_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    df = store.load_ohlcv(symbol, Timeframe.DAILY)
    if df is None or len(df) < _MIN_BARS:
        return {"trend_regime": "unknown", "volatility_regime": "normal", "confidence": 0.0}

    classifier = WeeklyRegimeClassifier()
    ctx = classifier.classify(df)

    # Map WeeklyRegimeClassifier output to QuantPod regime taxonomy.
    trend_regime = _map_trend(ctx.regime, ctx.ema_alignment, ctx.momentum_score)
    vol_regime = _map_vol(ctx.volatility_regime)

    return {
        "trend_regime":       trend_regime,
        "volatility_regime":  vol_regime,
        "confidence":         round(ctx.confidence, 3),
        "regime_label":       ctx.regime.value,
        "ema_alignment":      ctx.ema_alignment,
        "momentum_score":     round(ctx.momentum_score, 3),
        "bars_in_regime":     ctx.bars_in_regime,
    }


def _map_trend(regime: RegimeType, ema_alignment: int, momentum_score: float) -> str:
    """Map WeeklyRegimeClassifier result to QuantPod trend taxonomy."""
    if regime == RegimeType.BULL:
        return "trending_up"
    if regime == RegimeType.BEAR:
        return "trending_down"
    # SIDEWAYS — distinguish ranging vs weak trend by ema_alignment
    if regime == RegimeType.SIDEWAYS:
        if abs(momentum_score) < 0.15:
            return "ranging"
        return "trending_up" if ema_alignment > 0 else "trending_down"
    return "unknown"


def _map_vol(vol_regime_int: int) -> str:
    """Map volatility_regime int (-1, 0, 1) to string label."""
    return {-1: "low", 0: "normal", 1: "high"}.get(vol_regime_int, "normal")
