# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Volume collector — replaces structure_levels_ic.

Computes volume profile in-process: identifies High Volume Nodes (HVN) and
Low Volume Nodes (LVN) relative to current price. No network call.
"""

import asyncio
from typing import Any

import numpy as np
from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.storage import DataStore

_LOOKBACK_DAYS = 60   # volume profile window
_MIN_BARS = 20
_N_BINS = 20          # price bins for volume profile


async def collect_volume(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Compute volume profile and identify HVN/LVN zones near current price.

    Returns a dict with keys:
        vwap                 : float — volume-weighted average price (lookback window)
        volume_trend         : "increasing" | "decreasing" | "flat"
        adv_20               : float — average daily volume (20-day)
        at_hvn               : bool — current price at or within 0.5% of HVN
        at_lvn               : bool — current price at or within 0.5% of LVN
        hvn_levels           : list[float] — top 3 HVN price levels
        lvn_levels           : list[float] — top 3 LVN price levels
        vol_confirms_move    : bool — volume > 1.5x ADV on last bar
    """
    try:
        return await asyncio.to_thread(_collect_volume_sync, symbol, store)
    except Exception as exc:
        logger.warning(f"[volume] {symbol}: {exc}")
        return {}


def _collect_volume_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    df = store.load_ohlcv(symbol, Timeframe.DAILY)
    if df is None or len(df) < _MIN_BARS:
        return {}

    df = df.iloc[-_LOOKBACK_DAYS:].copy()
    close = df["close"].values
    volume = df["volume"].values
    current_price = float(close[-1])

    # VWAP over the lookback window
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = float((typical_price * df["volume"]).sum() / df["volume"].sum())

    # ADV — 20-day average daily volume
    adv_20 = float(df["volume"].iloc[-20:].mean())

    # Volume trend (last 5 bars vs prior 5 bars)
    recent_vol = float(df["volume"].iloc[-5:].mean())
    prior_vol = float(df["volume"].iloc[-10:-5].mean()) if len(df) >= 10 else recent_vol
    if prior_vol > 0:
        vol_ratio = recent_vol / prior_vol
        volume_trend = "increasing" if vol_ratio > 1.1 else "decreasing" if vol_ratio < 0.9 else "flat"
    else:
        volume_trend = "flat"

    # Volume profile: bin price range and sum volume per bin
    price_min, price_max = float(close.min()), float(close.max())
    if price_min == price_max:
        return {
            "vwap": vwap, "volume_trend": volume_trend, "adv_20": adv_20,
            "at_hvn": False, "at_lvn": False, "hvn_levels": [], "lvn_levels": [],
            "vol_confirms_move": float(df["volume"].iloc[-1]) > adv_20 * 1.5,
        }

    bins = np.linspace(price_min, price_max, _N_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    vol_by_bin, _ = np.histogram(close, bins=bins, weights=volume)

    # HVN: top third of volume; LVN: bottom third
    threshold_high = np.percentile(vol_by_bin[vol_by_bin > 0], 70) if vol_by_bin.any() else 0
    threshold_low = np.percentile(vol_by_bin[vol_by_bin > 0], 30) if vol_by_bin.any() else 0

    hvn_levels = sorted(
        [float(bin_centers[i]) for i, v in enumerate(vol_by_bin) if v >= threshold_high],
        key=lambda p: abs(p - current_price)
    )[:3]
    lvn_levels = sorted(
        [float(bin_centers[i]) for i, v in enumerate(vol_by_bin) if 0 < v <= threshold_low],
        key=lambda p: abs(p - current_price)
    )[:3]

    proximity_pct = 0.005  # 0.5%
    at_hvn = any(abs(p - current_price) / current_price <= proximity_pct for p in hvn_levels)
    at_lvn = any(abs(p - current_price) / current_price <= proximity_pct for p in lvn_levels)

    return {
        "vwap":              round(vwap, 4),
        "volume_trend":      volume_trend,
        "adv_20":            round(adv_20, 0),
        "at_hvn":            at_hvn,
        "at_lvn":            at_lvn,
        "hvn_levels":        [round(p, 4) for p in hvn_levels],
        "lvn_levels":        [round(p, 4) for p in lvn_levels],
        "vol_confirms_move": float(df["volume"].iloc[-1]) > adv_20 * 1.5,
    }
