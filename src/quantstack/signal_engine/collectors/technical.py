# Copyright 2024 QuantStack Contributors
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

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.core.features.flow import (
    CumulativeVolumeDelta,
    FootprintApproximation,
    HawkesIntensity,
    VPIN,
)
from quantstack.core.features.koncorde import Koncorde
from quantstack.core.features.momentum import (
    LaguerreRSI,
    MomentumFeatures,
    PercentRExhaustion,
)
from quantstack.core.features.smart_money import (
    BreakerBlockDetector,
    EqualHighsLows,
    FairValueGapDetector,
    ICTKillZones,
    ICTPowerOfThree,
    MMXMCycle,
    OrderBlockDetector,
    OTELevels,
    SilverBullet,
    StructureAnalysis,
)
from quantstack.core.features.microstructure import OvernightGapPersistence
from quantstack.core.features.statistical import (
    EntropyFeatures,
    HurstExponent,
    VarianceRatioTest,
    YangZhangVolatility,
)
from quantstack.core.features.rates import DualMomentum
from quantstack.core.features.technical_indicators import TechnicalIndicators
from quantstack.core.features.trend import (
    HullMovingAverage,
    IchimokuCloud,
    SupertrendIndicator,
)
from quantstack.core.features.volatility import VolatilityFeatures, WilliamsVIXFix

# Minimum bars required — indicators need lookback; 252 = 1 year daily, 60 = ~1 year weekly.
_MIN_DAILY_BARS = 60
_MIN_WEEKLY_BARS = 30
_DAILY_LOOKBACK = 504  # 2 years of daily bars


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
    daily_df = store.load_ohlcv(symbol, Timeframe.D1)
    if daily_df is None or len(daily_df) < _MIN_DAILY_BARS:
        logger.warning(
            f"[technical] {symbol}: insufficient daily bars "
            f"({len(daily_df) if daily_df is not None else 0} < {_MIN_DAILY_BARS})"
        )
        return {}

    # Use at most 2 years to keep compute fast; older history not needed for indicators.
    daily_df = daily_df.iloc[-_DAILY_LOOKBACK:]

    # Run feature classes — each returns a DataFrame with indicators added as columns.
    ti = TechnicalIndicators(Timeframe.D1, enable_hilbert=False)
    mf = MomentumFeatures(Timeframe.D1)
    vf = VolatilityFeatures(Timeframe.D1)

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
        logger.warning(f"[technical] {symbol}: Supertrend failed: {exc}")

    try:
        ichi_df = IchimokuCloud().compute(hi, lo, cl)
        result["tenkan_sen"] = _safe_float(ichi_df["tenkan_sen"].iloc[-1])
        result["kijun_sen"] = _safe_float(ichi_df["kijun_sen"].iloc[-1])
        result["cloud_bullish"] = int(ichi_df["cloud_bullish"].iloc[-1])
        result["price_above_cloud"] = int(ichi_df["price_above_cloud"].iloc[-1])
        result["price_below_cloud"] = int(ichi_df["price_below_cloud"].iloc[-1])
        result["tenkan_above_kijun"] = int(ichi_df["tenkan_above_kijun"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Ichimoku failed: {exc}")

    try:
        hma_df = HullMovingAverage(period=20).compute(cl)
        result["hma"] = _safe_float(hma_df["hma"].iloc[-1])
        result["hma_uptrend"] = int(hma_df["hma_uptrend"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: HMA failed: {exc}")

    try:
        pct_r_df = PercentRExhaustion(short=14, long=112).compute(hi, lo, cl)
        result["pct_r_short"] = _safe_float(pct_r_df["pct_r_short"].iloc[-1])
        result["pct_r_long"] = _safe_float(pct_r_df["pct_r_long"].iloc[-1])
        result["exhaustion_top"] = int(pct_r_df["exhaustion_top"].iloc[-1])
        result["exhaustion_bottom"] = int(pct_r_df["exhaustion_bottom"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: %R Exhaustion failed: {exc}")

    try:
        wvf_df = WilliamsVIXFix(lookback=22, bb_period=20, bb_dev=2.0).compute(
            hi, lo, cl
        )
        result["wvf"] = _safe_float(wvf_df["wvf"].iloc[-1])
        result["wvf_extreme"] = int(wvf_df["wvf_extreme"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Williams VIX Fix failed: {exc}")

    # --- ICT Smart Money Concepts ---
    op = df["open"] if "open" in df.columns else cl  # fallback if open unavailable
    try:
        fvg_df = FairValueGapDetector(min_gap_atr_multiple=0.1).compute(hi, lo, cl)
        result["bullish_fvg"] = int(fvg_df["bullish_fvg"].iloc[-1])
        result["bearish_fvg"] = int(fvg_df["bearish_fvg"].iloc[-1])
        result["fvg_top"] = _safe_float(fvg_df["fvg_top"].iloc[-1])
        result["fvg_bottom"] = _safe_float(fvg_df["fvg_bottom"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: FVG failed: {exc}")

    try:
        ob_df = OrderBlockDetector(impulse_atr_multiple=1.5).compute(op, hi, lo, cl)
        result["bullish_ob"] = int(ob_df["bullish_ob"].iloc[-1])
        result["bearish_ob"] = int(ob_df["bearish_ob"].iloc[-1])
        result["ob_high"] = _safe_float(ob_df["ob_high"].iloc[-1])
        result["ob_low"] = _safe_float(ob_df["ob_low"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Order Block failed: {exc}")

    try:
        sa_df = StructureAnalysis(swing_period=5).compute(hi, lo, cl)
        result["bos_bullish"] = int(sa_df["bos_bullish"].iloc[-1])
        result["bos_bearish"] = int(sa_df["bos_bearish"].iloc[-1])
        result["choch_bullish"] = int(sa_df["choch_bullish"].iloc[-1])
        result["choch_bearish"] = int(sa_df["choch_bearish"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Structure Analysis failed: {exc}")

    try:
        ehl_df = EqualHighsLows(lookback=20).compute(hi, lo, cl)
        result["equal_highs"] = int(ehl_df["equal_highs"].iloc[-1])
        result["equal_lows"] = int(ehl_df["equal_lows"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Equal H/L failed: {exc}")

    try:
        ote_df = OTELevels(swing_period=20).compute(hi, lo, cl)
        result["price_in_ote"] = int(ote_df["price_in_ote"].iloc[-1])
        result["ote_upper"] = _safe_float(ote_df["ote_upper"].iloc[-1])
        result["ote_lower"] = _safe_float(ote_df["ote_lower"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: OTE Levels failed: {exc}")

    try:
        kz_df = ICTKillZones().compute(df.index)
        result["in_any_kz"] = int(kz_df["in_any_kz"].iloc[-1])
        result["in_london_kz"] = int(kz_df["in_london_kz"].iloc[-1])
        result["in_ny_am_kz"] = int(kz_df["in_ny_am_kz"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Kill Zones failed: {exc}")

    try:
        po3_df = ICTPowerOfThree().compute(op, hi, lo, cl)
        result["po3_tight_range"] = int(po3_df["tight_range"].iloc[-1])
        result["po3_manipulation_up"] = int(po3_df["manipulation_up"].iloc[-1])
        result["po3_manipulation_down"] = int(po3_df["manipulation_down"].iloc[-1])
        result["po3_distribution_up"] = int(po3_df["distribution_up"].iloc[-1])
        result["po3_distribution_down"] = int(po3_df["distribution_down"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Power of Three failed: {exc}")

    try:
        bb_df = BreakerBlockDetector(impulse_atr_multiple=1.5).compute(op, hi, lo, cl)
        result["bullish_breaker"] = int(bb_df["bullish_breaker"].iloc[-1])
        result["bearish_breaker"] = int(bb_df["bearish_breaker"].iloc[-1])
        result["breaker_high"] = _safe_float(bb_df["breaker_high"].iloc[-1])
        result["breaker_low"] = _safe_float(bb_df["breaker_low"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Breaker Block failed: {exc}")

    try:
        mmxm_df = MMXMCycle().compute(hi, lo, cl)
        result["mmxm_phase"] = int(mmxm_df["mmxm_phase"].iloc[-1])
        result["mmxm_label"] = mmxm_df["mmxm_label"].iloc[-1]
        result["in_consolidation"] = int(mmxm_df["in_consolidation"].iloc[-1])
        result["in_manipulation"] = int(mmxm_df["in_manipulation"].iloc[-1])
        result["in_expansion"] = int(mmxm_df["in_expansion"].iloc[-1])
        result["in_retracement"] = int(mmxm_df["in_retracement"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: MMXM Cycle failed: {exc}")

    try:
        sb_df = SilverBullet().compute(hi, lo, cl)
        result["sb_bullish"] = int(sb_df["sb_bullish"].iloc[-1])
        result["sb_bearish"] = int(sb_df["sb_bearish"].iloc[-1])
        result["sb_fvg_top"] = _safe_float(sb_df["sb_fvg_top"].iloc[-1])
        result["sb_fvg_bot"] = _safe_float(sb_df["sb_fvg_bot"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: SilverBullet failed: {exc}")

    try:
        lrsi_df = LaguerreRSI(gamma=0.5).compute(cl)
        result["lrsi"] = _safe_float(lrsi_df["lrsi"].iloc[-1])
        result["lma"] = _safe_float(lrsi_df["lma"].iloc[-1])
        result["lrsi_overbought"] = int(lrsi_df["lrsi_ob"].iloc[-1])
        result["lrsi_oversold"] = int(lrsi_df["lrsi_os"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Laguerre RSI failed: {exc}")

    try:
        dm_df = DualMomentum(abs_lookback=252, skip_period=21).compute(cl)
        result["momentum_12m1m"] = _safe_float(dm_df["momentum_12m1m"].iloc[-1])
        result["abs_momentum_signal"] = int(dm_df["abs_momentum_signal"].iloc[-1])
        result["momentum_6m"] = _safe_float(dm_df["momentum_6m"].iloc[-1])
        result["momentum_3m"] = _safe_float(dm_df["momentum_3m"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: DualMomentum failed: {exc}")

    # --- Order flow approximations (require volume column) ---
    if "volume" in df.columns:
        vol = df["volume"]
        try:
            cvd_df = CumulativeVolumeDelta(lookback=20).compute(op, hi, lo, cl, vol)
            result["cvd"] = _safe_float(cvd_df["cvd"].iloc[-1])
            result["cvd_divergence"] = int(cvd_df["cvd_divergence"].iloc[-1])
            result["bar_delta"] = _safe_float(cvd_df["bar_delta"].iloc[-1])
        except Exception as exc:
            logger.warning(f"[technical] {symbol}: CVD failed: {exc}")

        try:
            hawkes_df = HawkesIntensity().compute(hi, lo, cl, vol)
            result["hawkes_intensity"] = _safe_float(hawkes_df["intensity"].iloc[-1])
            result["hawkes_excited"] = int(hawkes_df["excited"].iloc[-1])
            result["hawkes_event"] = int(hawkes_df["event"].iloc[-1])
        except Exception as exc:
            logger.warning(f"[technical] {symbol}: Hawkes Intensity failed: {exc}")

        try:
            kc_df = Koncorde().compute(hi, lo, cl, vol)
            result["koncorde_green"] = _safe_float(kc_df["green_line"].iloc[-1])
            result["koncorde_blue"] = _safe_float(kc_df["blue_line"].iloc[-1])
            result["koncorde_agreement"] = int(kc_df["agreement"].iloc[-1])
            result["koncorde_divergence"] = int(kc_df["divergence"].iloc[-1])
        except Exception as exc:
            logger.warning(f"[technical] {symbol}: Koncorde failed: {exc}")

        try:
            op = df["open"]
            fp_df = FootprintApproximation().compute(op, hi, lo, cl, vol)
            result["fp_bar_delta"] = _safe_float(fp_df["bar_delta"].iloc[-1])
            result["fp_delta_pct"] = _safe_float(fp_df["delta_pct"].iloc[-1])
            result["fp_imbalanced_bull"] = int(fp_df["imbalanced_bull"].iloc[-1])
            result["fp_imbalanced_bear"] = int(fp_df["imbalanced_bear"].iloc[-1])
            result["fp_stacked_bull"] = int(fp_df["stacked_bull"].iloc[-1])
            result["fp_stacked_bear"] = int(fp_df["stacked_bear"].iloc[-1])
            result["fp_poc_price"] = _safe_float(fp_df["poc_price"].iloc[-1])
        except Exception as exc:
            logger.warning(f"[technical] {symbol}: FootprintApproximation failed: {exc}")

        try:
            vpin_df = VPIN(n_buckets=50, window=50).compute(op, hi, lo, cl, vol)
            result["vpin"] = _safe_float(vpin_df["vpin"].iloc[-1])
            result["vpin_high"] = int(vpin_df["vpin_high"].iloc[-1])
        except Exception as exc:
            logger.warning(f"[technical] {symbol}: VPIN failed: {exc}")

    # --- Overnight gap + volume spike (institutional intent signal) ---
    if "open" in df.columns:
        op_series = df["open"]
        try:
            vol_series = df["volume"] if "volume" in df.columns else None
            gap_df = OvernightGapPersistence(
                min_gap_pct=0.2, volume_spike_mult=2.0
            ).compute(op_series, cl, volume=vol_series)
            result["gap_pct"] = _safe_float(gap_df["gap_pct"].iloc[-1])
            result["gap_up"] = int(gap_df["gap_up"].iloc[-1])
            result["gap_down"] = int(gap_df["gap_down"].iloc[-1])
            result["gap_persisted"] = int(gap_df["gap_persisted"].iloc[-1])
            result["gap_filled_pct"] = _safe_float(gap_df["gap_filled_pct"].iloc[-1])
            if "institutional_gap" in gap_df.columns:
                result["volume_spike"] = int(gap_df["volume_spike"].iloc[-1])
                result["institutional_gap"] = int(gap_df["institutional_gap"].iloc[-1])
        except Exception as exc:
            logger.warning(f"[technical] {symbol}: OvernightGapPersistence failed: {exc}")

    # --- Statistical features (hedge fund grade) ---
    try:
        if "open" in df.columns:
            yz_df = YangZhangVolatility(period=22).compute(
                df["open"], df["high"], df["low"], cl
            )
            result["yang_zhang_vol"] = _safe_float(yz_df["yang_zhang_vol"].iloc[-1])
            result["parkinson_vol"] = _safe_float(yz_df["parkinson_vol"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: YangZhang failed: {exc}")

    try:
        vr_df = VarianceRatioTest(lags=[2, 5, 10], window=126).compute(cl)
        result["vr_5"] = _safe_float(vr_df["vr_5"].iloc[-1])
        result["vr_10"] = _safe_float(vr_df["vr_10"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: VarianceRatio failed: {exc}")

    try:
        ent_df = EntropyFeatures(window=63).compute(cl)
        result["shannon_entropy"] = _safe_float(ent_df["shannon_entropy"].iloc[-1])
        result["entropy_regime"] = _safe_float(ent_df["entropy_regime"].iloc[-1])
    except Exception as exc:
        logger.warning(f"[technical] {symbol}: Entropy failed: {exc}")

    # Note: HurstExponent (window=252) is too slow for real-time collector
    # (~0.5s per symbol). Use it in the FeatureFactory for ML training only.

    # Weekly MTF alignment
    weekly_df = store.load_ohlcv(symbol, Timeframe.W1)
    if weekly_df is not None and len(weekly_df) >= _MIN_WEEKLY_BARS:
        wti = TechnicalIndicators(Timeframe.W1, enable_hilbert=False)
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
        "rsi_14": _f("rsi_14", "RSI"),
        "macd_hist": _f("macd_hist", "MACD_Hist"),
        "macd_line": _f("macd_line", "MACD"),
        "macd_signal": _f("macd_signal", "MACD_Signal"),
        "stoch_k": _f("stoch_k", "SlowK"),
        "stoch_d": _f("stoch_d", "SlowD"),
        # Trend strength
        "adx_14": _f("adx_14", "ADX"),
        "plus_di": _f("plus_di", "plus_DI"),
        "minus_di": _f("minus_di", "minus_DI"),
        # Moving averages
        "sma_20": _f("sma_20", "SMA_20"),
        "sma_50": _f("sma_50", "SMA_50"),
        "sma_200": _f("sma_200", "SMA_200"),
        "ema_20": _f("ema_20", "EMA_20"),
        # Price context
        "close": _f("close"),
        # Volatility
        "atr_14": _f("atr_14", "ATR"),
        "atr_pct": _f("atr_pct"),  # ATR as % of price
        "bb_upper": _f("bb_upper", "BBands_Upper"),
        "bb_lower": _f("bb_lower", "BBands_Lower"),
        "bb_middle": _f("bb_middle", "BBands_Middle"),
        "bb_pct": _f("bb_pct"),  # (close - lower) / (upper - lower)
        "bb_width": _f("bb_width"),  # (upper - lower) / middle
        "realized_vol_20": _f("realized_vol_20"),
        "vol_zscore": _f("vol_zscore"),  # current vol vs rolling avg
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
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None
