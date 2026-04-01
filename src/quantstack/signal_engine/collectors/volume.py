# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Volume collector — replaces structure_levels_ic.

Computes volume profile in-process: identifies High Volume Nodes (HVN) and
Low Volume Nodes (LVN) relative to current price. Also computes VPOC/VAH/VAL
(Volume Point of Control + Value Area), Anchored VWAP from the recent swing
low, and microstructure liquidity estimates (Amihud, Roll, Corwin-Schultz,
realized variance decomposition, overnight gap persistence). No network call.
"""

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.core.features.microstructure import (
    AmihudIlliquidity,
    CorwinSchultzSpread,
    OvernightGapPersistence,
    RealizedVarianceDecomposition,
    RollImpliedSpread,
    VWAPSessionDeviation,
)
from quantstack.core.features.volume import AnchoredVWAP, VolumePointOfControl

_LOOKBACK_DAYS = 60  # volume profile window
_MIN_BARS = 20
_N_BINS = 20  # price bins for volume profile


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
    df = store.load_ohlcv(symbol, Timeframe.D1)
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
        volume_trend = (
            "increasing"
            if vol_ratio > 1.1
            else "decreasing" if vol_ratio < 0.9 else "flat"
        )
    else:
        volume_trend = "flat"

    # Volume profile: bin price range and sum volume per bin
    price_min, price_max = float(close.min()), float(close.max())
    if price_min == price_max:
        return {
            "vwap": vwap,
            "volume_trend": volume_trend,
            "adv_20": adv_20,
            "at_hvn": False,
            "at_lvn": False,
            "hvn_levels": [],
            "lvn_levels": [],
            "vol_confirms_move": float(df["volume"].iloc[-1]) > adv_20 * 1.5,
        }

    bins = np.linspace(price_min, price_max, _N_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    vol_by_bin, _ = np.histogram(close, bins=bins, weights=volume)

    # HVN: top third of volume; LVN: bottom third
    threshold_high = (
        np.percentile(vol_by_bin[vol_by_bin > 0], 70) if vol_by_bin.any() else 0
    )
    threshold_low = (
        np.percentile(vol_by_bin[vol_by_bin > 0], 30) if vol_by_bin.any() else 0
    )

    hvn_levels = sorted(
        [
            float(bin_centers[i])
            for i, v in enumerate(vol_by_bin)
            if v >= threshold_high
        ],
        key=lambda p: abs(p - current_price),
    )[:3]
    lvn_levels = sorted(
        [
            float(bin_centers[i])
            for i, v in enumerate(vol_by_bin)
            if 0 < v <= threshold_low
        ],
        key=lambda p: abs(p - current_price),
    )[:3]

    proximity_pct = 0.005  # 0.5%
    at_hvn = any(
        abs(p - current_price) / current_price <= proximity_pct for p in hvn_levels
    )
    at_lvn = any(
        abs(p - current_price) / current_price <= proximity_pct for p in lvn_levels
    )

    result: dict[str, Any] = {
        "vwap": round(vwap, 4),
        "volume_trend": volume_trend,
        "adv_20": round(adv_20, 0),
        "at_hvn": at_hvn,
        "at_lvn": at_lvn,
        "hvn_levels": [round(p, 4) for p in hvn_levels],
        "lvn_levels": [round(p, 4) for p in lvn_levels],
        "vol_confirms_move": float(df["volume"].iloc[-1]) > adv_20 * 1.5,
    }

    # --- VPOC / VAH / VAL ---
    try:
        vpoc_df = VolumePointOfControl(lookback=_LOOKBACK_DAYS, n_bins=_N_BINS).compute(
            df["high"], df["low"], df["close"], df["volume"]
        )
        result["vpoc"] = _sfloat(vpoc_df["vpoc"].iloc[-1])
        result["vah"] = _sfloat(vpoc_df["vah"].iloc[-1])
        result["val"] = _sfloat(vpoc_df["val"].iloc[-1])
        result["price_in_value_area"] = int(vpoc_df["in_value_area"].iloc[-1])
        result["price_above_vpoc"] = (
            int(current_price > result["vpoc"]) if result["vpoc"] else None
        )
    except Exception as exc:
        logger.debug(f"[volume] {symbol}: VPOC failed: {exc}")

    # --- Anchored VWAP (anchored to the bar with the lowest close in the lookback window) ---
    try:
        anchor_pos = int(df["close"].values.argmin())
        avwap_df = AnchoredVWAP(anchor=anchor_pos).compute(
            df["high"], df["low"], df["close"], df["volume"]
        )
        result["avwap"] = _sfloat(avwap_df["avwap"].iloc[-1])
        result["avwap_deviation"] = _sfloat(avwap_df["avwap_deviation"].iloc[-1])
        result["above_avwap"] = int(avwap_df["above_avwap"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[volume] {symbol}: AnchoredVWAP failed: {exc}")

    # --- Microstructure liquidity signals ---
    try:
        amihud_df = AmihudIlliquidity(period=22).compute(df["close"], df["volume"])
        result["amihud"] = _sfloat(amihud_df["amihud"].iloc[-1])
        result["amihud_zscore"] = _sfloat(amihud_df["amihud_zscore"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[volume] {symbol}: Amihud failed: {exc}")

    try:
        roll_df = RollImpliedSpread(period=22).compute(df["close"])
        result["roll_spread_pct"] = _sfloat(roll_df["roll_spread_pct"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[volume] {symbol}: Roll spread failed: {exc}")

    try:
        cs_df = CorwinSchultzSpread(period=22).compute(
            df["high"], df["low"], df["close"]
        )
        result["cs_spread_pct"] = _sfloat(cs_df["cs_spread_pct"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[volume] {symbol}: Corwin-Schultz failed: {exc}")

    try:
        vwap_dev_df = VWAPSessionDeviation(period=20).compute(
            df["high"], df["low"], df["close"], df["volume"]
        )
        result["vwap_deviation"] = _sfloat(vwap_dev_df["vwap_deviation"].iloc[-1])
        result["vwap_deviation_zscore"] = _sfloat(
            vwap_dev_df["vwap_deviation_zscore"].iloc[-1]
        )
    except Exception as exc:
        logger.debug(f"[volume] {symbol}: VWAPSessionDeviation failed: {exc}")

    if "open" in df.columns:
        try:
            rv_df = RealizedVarianceDecomposition(period=22).compute(
                df["open"], df["high"], df["low"], df["close"]
            )
            result["rv_overnight_ratio"] = _sfloat(
                rv_df["overnight_var_ratio"].iloc[-1]
            )
        except Exception as exc:
            logger.debug(f"[volume] {symbol}: RV decomp failed: {exc}")

        try:
            gap_df = OvernightGapPersistence(min_gap_pct=0.2).compute(
                df["open"], df["close"]
            )
            result["gap_filled_pct"] = _sfloat(gap_df["gap_filled_pct"].iloc[-1])
            result["gap_up"] = int(gap_df["gap_up"].iloc[-1])
            result["gap_down"] = int(gap_df["gap_down"].iloc[-1])
            result["gap_persisted"] = int(gap_df["gap_persisted"].iloc[-1])
        except Exception as exc:
            logger.debug(f"[volume] {symbol}: Overnight gap failed: {exc}")

    return result


def _sfloat(v: Any) -> float | None:
    """Return float or None — never NaN (breaks JSON serialisation)."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else round(f, 6)  # NaN check
    except (TypeError, ValueError):
        return None
