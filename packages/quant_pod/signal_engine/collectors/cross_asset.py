# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Cross-asset collector — inter-market signals for risk regime detection.

Loads OHLCV from the local DataStore for SPY, QQQ, IWM, TLT, and GLD.
No network call — all data must be pre-loaded via the data pipeline.
Returns {} if SPY data is missing (required baseline).  Other ETFs are
optional and degrade gracefully.
"""

import asyncio
from typing import Any

from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.storage import DataStore
from quantcore.features.smart_money import SMTDivergence


_MIN_BARS = 10  # need at least 5 + buffer for return computation
_CROSS_ASSET_SYMBOLS = ("SPY", "QQQ", "IWM", "TLT", "GLD")


async def collect_cross_asset(symbol: str, store: DataStore) -> dict[str, Any]:
    """Compute cross-asset risk regime signals. Returns {} on failure."""
    try:
        return await asyncio.to_thread(_collect_cross_asset_sync, symbol, store)
    except Exception as exc:
        logger.debug(f"[cross_asset] {symbol}: {exc} — returning empty")
        return {}


def _collect_cross_asset_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    # SPY is the required baseline — without it we cannot compute anything.
    spy_df = store.load_ohlcv("SPY", Timeframe.D1)
    if spy_df is None or len(spy_df) < _MIN_BARS:
        logger.debug("[cross_asset] SPY data missing or insufficient — skipping")
        return {}

    spy_close = spy_df["close"].values
    spy_ret_5d = _return_nday(spy_close, 5)

    if spy_ret_5d is None:
        return {}

    result: dict[str, Any] = {
        "spy_regime_5d": _round_or_none(spy_ret_5d),
    }

    # QQQ vs SPY — tech premium
    qqq_ret = _load_return(store, "QQQ", 5)
    result["qqq_vs_spy_5d"] = _round_or_none(_sub(qqq_ret, spy_ret_5d))

    # IWM vs SPY — small-cap premium
    iwm_ret = _load_return(store, "IWM", 5)
    result["iwm_vs_spy_5d"] = _round_or_none(_sub(iwm_ret, spy_ret_5d))

    # TLT — bond direction
    tlt_ret = _load_return(store, "TLT", 5)
    result["tlt_return_5d"] = _round_or_none(tlt_ret)

    # GLD — gold / fear
    gld_ret = _load_return(store, "GLD", 5)
    result["gld_return_5d"] = _round_or_none(gld_ret)

    # Composite risk-on score
    result["risk_on_score"] = _compute_risk_on_score(spy_ret_5d, tlt_ret, gld_ret)

    # Cross-asset regime classification
    result["cross_asset_regime"] = _classify_regime(
        spy_ret_5d, tlt_ret, gld_ret
    )

    # --- SMT Divergence: symbol vs SPY (or SPY vs QQQ when symbol IS SPY) ---
    try:
        if symbol.upper() != "SPY":
            sym_df = store.load_ohlcv(symbol, Timeframe.D1)
        else:
            sym_df = store.load_ohlcv("QQQ", Timeframe.D1)

        if sym_df is not None and len(sym_df) >= 20 and len(spy_df) >= 20:
            # Align on common index (inner join by position — both are daily)
            n = min(len(sym_df), len(spy_df))
            smt_df = SMTDivergence(swing_period=5).compute(
                high_a=sym_df["high"].iloc[-n:].reset_index(drop=True),
                low_a=sym_df["low"].iloc[-n:].reset_index(drop=True),
                high_b=spy_df["high"].iloc[-n:].reset_index(drop=True),
                low_b=spy_df["low"].iloc[-n:].reset_index(drop=True),
            )
            result["smt_bearish"] = int(smt_df["bearish_smt"].iloc[-1])
            result["smt_bullish"] = int(smt_df["bullish_smt"].iloc[-1])
            result["smt_strength"] = _safe_float(smt_df["smt_strength"].iloc[-1])
    except Exception as exc:
        logger.debug(f"[cross_asset] {symbol}: SMTDivergence failed: {exc}")

    return result


def _load_return(store: DataStore, ticker: str, n: int) -> float | None:
    """Load OHLCV and compute n-day return. None if data unavailable."""
    df = store.load_ohlcv(ticker, Timeframe.D1)
    if df is None or len(df) < n + 1:
        return None
    return _return_nday(df["close"].values, n)


def _return_nday(close: Any, n: int) -> float | None:
    """Compute n-day simple return from a close price array."""
    if len(close) < n + 1:
        return None
    prior = _safe_float(close[-(n + 1)])
    latest = _safe_float(close[-1])
    if prior is None or latest is None or prior <= 0:
        return None
    return (latest - prior) / prior


def _compute_risk_on_score(
    spy_ret: float | None, tlt_ret: float | None, gld_ret: float | None
) -> float:
    """Composite risk-on score in [0, 1].

    High when: equities up, bonds down, gold down (classic risk-on).
    Low when: equities down, bonds up, gold up (classic risk-off).

    Each component contributes equally.  Missing components scored as neutral (0.5).
    """
    scores: list[float] = []

    # Equities up = risk-on
    if spy_ret is not None:
        scores.append(_sigmoid_score(spy_ret, scale=50.0))
    else:
        scores.append(0.5)

    # Bonds down = risk-on (inverse)
    if tlt_ret is not None:
        scores.append(_sigmoid_score(-tlt_ret, scale=50.0))
    else:
        scores.append(0.5)

    # Gold down = risk-on (inverse)
    if gld_ret is not None:
        scores.append(_sigmoid_score(-gld_ret, scale=50.0))
    else:
        scores.append(0.5)

    avg = sum(scores) / len(scores)
    return round(max(0.0, min(1.0, avg)), 3)


def _sigmoid_score(x: float, scale: float = 50.0) -> float:
    """Map a return value to [0, 1] using a sigmoid-like function.

    scale controls sensitivity — higher values make the function steeper.
    At scale=50, a 2% return maps to ~0.73.
    """
    import math

    try:
        return 1.0 / (1.0 + math.exp(-scale * x))
    except OverflowError:
        return 1.0 if x > 0 else 0.0


def _classify_regime(
    spy_ret: float | None, tlt_ret: float | None, gld_ret: float | None
) -> str:
    """Classify cross-asset regime into risk_on / risk_off / mixed."""
    if spy_ret is None:
        return "mixed"

    # Strong equity signals dominate
    equity_positive = spy_ret > 0.005
    equity_negative = spy_ret < -0.005

    bonds_up = tlt_ret is not None and tlt_ret > 0.005
    bonds_down = tlt_ret is not None and tlt_ret < -0.005
    gold_up = gld_ret is not None and gld_ret > 0.005
    gold_down = gld_ret is not None and gld_ret < -0.005

    # Classic risk-on: equities up, bonds flat/down, gold flat/down
    if equity_positive and not bonds_up and not gold_up:
        return "risk_on"

    # Classic risk-off: equities down, bonds up or gold up
    if equity_negative and (bonds_up or gold_up):
        return "risk_off"

    # Equities down without safe-haven bid — still risk-off
    if equity_negative:
        return "risk_off"

    # Equities up but safe havens also up — mixed / uncertain
    if equity_positive and (bonds_up or gold_up):
        return "mixed"

    return "mixed"


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _round_or_none(v: float | None, decimals: int = 6) -> float | None:
    return round(v, decimals) if v is not None else None


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return None
