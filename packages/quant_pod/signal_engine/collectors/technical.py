# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Technical collector — replaces trend_momentum_ic + volatility_ic + market_snapshot_ic.

Loads OHLCV from local DataStore (read-only, no network call).
Runs TechnicalIndicators, MomentumFeatures, VolatilityFeatures on daily bars.
Also loads weekly bars for MTF trend alignment.
Returns last-row indicator values as a flat dict.
"""

import asyncio
from typing import Any

import pandas as pd
from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.storage import DataStore
from quantcore.features.momentum import MomentumFeatures, PercentRExhaustion
from quantcore.features.technical_indicators import TechnicalIndicators
from quantcore.features.trend import HullMovingAverage, IchimokuCloud, SupertrendIndicator
from quantcore.features.volatility import VolatilityFeatures, WilliamsVIXFix

# Minimum bars required — indicators need lookback; 252 = 1 year daily, 60 = ~1 year weekly.
_MIN_DAILY_BARS = 60
_MIN_WEEKLY_BARS = 30
_DAILY_LOOKBACK = 504   # 2 years of daily bars


async def collect_technical(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Compute technical indicators for *symbol* from locally stored OHLCV.

    Returns a flat dict of indicator values (last bar only) plus weekly
    MTF alignment context.  Returns {} if insufficient data is available.
    """
    try:
        return await asyncio.to_thread(_collect_technical_sync, symbol, store)
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: {exc}")
        return {}


def _collect_technical_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    daily_df = store.load_ohlcv(symbol, Timeframe.DAILY)
    if daily_df is None or len(daily_df) < _MIN_DAILY_BARS:
        logger.warning(
            f"[technical] {symbol}: insufficient daily bars "
            f"({len(daily_df) if daily_df is not None else 0} < {_MIN_DAILY_BARS})"
        )
        return {}

    # Use at most 2 years to keep compute fast; older history not needed for indicators.
    daily_df = daily_df.iloc[-_DAILY_LOOKBACK:]

    # Run feature classes — each returns a DataFrame with indicators added as columns.
    ti = TechnicalIndicators(Timeframe.DAILY, enable_hilbert=False)
    mf = MomentumFeatures(Timeframe.DAILY)
    vf = VolatilityFeatures(Timeframe.DAILY)

    df = ti.compute(daily_df)
    df = mf.compute(df)
    df = vf.compute(df)

    last = df.iloc[-1].to_dict()
    result = _extract_key_indicators(last)

    # Phase 1 advanced indicators — computed on the full daily window.
    # These are standalone (not FeatureBase subclasses) so we call them directly
    # and extract the last-row value.
    hi, lo, cl = df["high"], df["low"], df["close"]

    try:
        st_df = SupertrendIndicator(atr_length=10, multiplier=3.0).compute(hi, lo, cl)
        result["supertrend"] = _safe_float(st_df["supertrend"].iloc[-1])
        result["st_direction"] = int(st_df["st_direction"].iloc[-1])
        result["st_uptrend"] = bool(st_df["st_uptrend"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[technical] {symbol}: Supertrend failed: {exc}")

    try:
        ichi_df = IchimokuCloud().compute(hi, lo, cl)
        result["tenkan_sen"] = _safe_float(ichi_df["tenkan_sen"].iloc[-1])
        result["kijun_sen"] = _safe_float(ichi_df["kijun_sen"].iloc[-1])
        result["cloud_bullish"] = int(ichi_df["cloud_bullish"].iloc[-1])
        result["price_above_cloud"] = int(ichi_df["price_above_cloud"].iloc[-1])
        result["price_below_cloud"] = int(ichi_df["price_below_cloud"].iloc[-1])
        result["tenkan_above_kijun"] = int(ichi_df["tenkan_above_kijun"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[technical] {symbol}: Ichimoku failed: {exc}")

    try:
        hma_df = HullMovingAverage(period=20).compute(cl)
        result["hma"] = _safe_float(hma_df["hma"].iloc[-1])
        result["hma_uptrend"] = int(hma_df["hma_uptrend"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[technical] {symbol}: HMA failed: {exc}")

    try:
        pct_r_df = PercentRExhaustion(short=14, long=112).compute(hi, lo, cl)
        result["pct_r_short"] = _safe_float(pct_r_df["pct_r_short"].iloc[-1])
        result["pct_r_long"] = _safe_float(pct_r_df["pct_r_long"].iloc[-1])
        result["exhaustion_top"] = int(pct_r_df["exhaustion_top"].iloc[-1])
        result["exhaustion_bottom"] = int(pct_r_df["exhaustion_bottom"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[technical] {symbol}: %R Exhaustion failed: {exc}")

    try:
        wvf_df = WilliamsVIXFix(lookback=22, bb_period=20, bb_dev=2.0).compute(hi, lo, cl)
        result["wvf"] = _safe_float(wvf_df["wvf"].iloc[-1])
        result["wvf_extreme"] = int(wvf_df["wvf_extreme"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[technical] {symbol}: Williams VIX Fix failed: {exc}")

    # Weekly MTF alignment
    weekly_df = store.load_ohlcv(symbol, Timeframe.WEEKLY)
    if weekly_df is not None and len(weekly_df) >= _MIN_WEEKLY_BARS:
        wti = TechnicalIndicators(Timeframe.WEEKLY, enable_hilbert=False)
        wdf = wti.compute(weekly_df)
        wlast = wdf.iloc[-1].to_dict()
        result["weekly_rsi"] = _safe_float(wlast.get("rsi_14") or wlast.get("RSI"))
        result["weekly_trend"] = _weekly_trend(wlast)
    else:
        result["weekly_rsi"] = None
        result["weekly_trend"] = "unknown"

    return result


def _extract_key_indicators(row: dict) -> dict[str, Any]:
    """Pull the indicators that synthesis logic uses, with safe fallbacks."""
    def _f(key: str, alt: str | None = None) -> float | None:
        v = row.get(key)
        if v is None and alt:
            v = row.get(alt)
        return _safe_float(v)

    return {
        # Momentum / oscillators
        "rsi_14":           _f("rsi_14", "RSI"),
        "macd_hist":        _f("macd_hist", "MACD_Hist"),
        "macd_line":        _f("macd_line", "MACD"),
        "macd_signal":      _f("macd_signal", "MACD_Signal"),
        "stoch_k":          _f("stoch_k", "SlowK"),
        "stoch_d":          _f("stoch_d", "SlowD"),
        # Trend strength
        "adx_14":           _f("adx_14", "ADX"),
        "plus_di":          _f("plus_di", "plus_DI"),
        "minus_di":         _f("minus_di", "minus_DI"),
        # Moving averages
        "sma_20":           _f("sma_20", "SMA_20"),
        "sma_50":           _f("sma_50", "SMA_50"),
        "sma_200":          _f("sma_200", "SMA_200"),
        "ema_20":           _f("ema_20", "EMA_20"),
        # Price context
        "close":            _f("close"),
        # Volatility
        "atr_14":           _f("atr_14", "ATR"),
        "atr_pct":          _f("atr_pct"),        # ATR as % of price
        "bb_upper":         _f("bb_upper", "BBands_Upper"),
        "bb_lower":         _f("bb_lower", "BBands_Lower"),
        "bb_middle":        _f("bb_middle", "BBands_Middle"),
        "bb_pct":           _f("bb_pct"),          # (close - lower) / (upper - lower)
        "bb_width":         _f("bb_width"),         # (upper - lower) / middle
        "realized_vol_20":  _f("realized_vol_20"),
        "vol_zscore":       _f("vol_zscore"),       # current vol vs rolling avg
    }


def _weekly_trend(row: dict) -> str:
    """Classify weekly trend from last-row indicator dict."""
    sma20 = _safe_float(row.get("sma_20") or row.get("SMA_20"))
    sma50 = _safe_float(row.get("sma_50") or row.get("SMA_50"))
    close = _safe_float(row.get("close"))

    if None in (sma20, sma50, close):
        return "unknown"

    if close > sma20 > sma50:
        return "bullish"
    if close < sma20 < sma50:
        return "bearish"
    return "neutral"


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f)  # NaN check
        else f
    except (TypeError, ValueError):
        return None
