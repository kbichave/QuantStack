# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Risk collector — replaces risk_limits_ic.

Computes VaR, ATR-based stop levels, and liquidity metrics entirely from
locally stored OHLCV. No network call. No LLM.
"""

import asyncio
from typing import Any

import numpy as np
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore

_LOOKBACK = 252  # 1 year for VaR
_MIN_BARS = 30
_TRADING_DAYS_PER_YEAR = 252


async def collect_risk(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Compute risk metrics for *symbol* from locally stored daily OHLCV.

    Returns a dict with keys:
        var_95              : float — 1-day 95% VaR as % of price (positive = loss)
        var_99              : float — 1-day 99% VaR as % of price
        atr_stop_long       : float — price level for ATR-based stop (2× ATR below close)
        atr_stop_short      : float — price level for ATR-based stop (2× ATR above close)
        realized_vol_ann    : float — annualized realized volatility (30-day)
        liquidity_score     : float [0, 1] — based on ADV (>1M=1.0, <100k=0.0)
        max_drawdown_90d    : float — max drawdown over last 90 days (%)
    """
    try:
        return await asyncio.to_thread(_collect_risk_sync, symbol, store)
    except Exception as exc:
        logger.warning(f"[risk] {symbol}: {exc}")
        return {}


def _collect_risk_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    df = store.load_ohlcv(symbol, Timeframe.D1)
    if df is None or len(df) < _MIN_BARS:
        return {}

    df = df.iloc[-_LOOKBACK:].copy()
    close = df["close"].values
    current_price = float(close[-1])

    # Daily log returns
    returns = np.diff(np.log(close))

    # VaR (historical simulation)
    var_95 = float(-np.percentile(returns[-252:], 5)) if len(returns) >= 20 else 0.02
    var_99 = float(-np.percentile(returns[-252:], 1)) if len(returns) >= 20 else 0.04

    # ATR (14-day) for stop levels
    high = df["high"].values
    low = df["low"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])),
    )
    atr_14 = float(tr[-14:].mean()) if len(tr) >= 14 else float(tr.mean())

    # Realized vol (30-day, annualized)
    rv_30 = (
        float(returns[-30:].std() * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if len(returns) >= 30
        else 0.20
    )

    # ADV-based liquidity score
    adv = float(df["volume"].iloc[-20:].mean())
    liquidity_score = min(1.0, max(0.0, (adv - 100_000) / (1_000_000 - 100_000)))

    # Max drawdown over last 90 days
    recent_close = close[-90:] if len(close) >= 90 else close
    roll_max = np.maximum.accumulate(recent_close)
    drawdowns = (recent_close - roll_max) / roll_max
    max_dd_90 = float(drawdowns.min())  # negative value; e.g. -0.05 = 5% drawdown

    return {
        "var_95": round(var_95 * 100, 3),  # convert to %
        "var_99": round(var_99 * 100, 3),
        "atr_14": round(atr_14, 4),
        "atr_stop_long": round(current_price - 2 * atr_14, 4),
        "atr_stop_short": round(current_price + 2 * atr_14, 4),
        "realized_vol_ann": round(rv_30, 4),
        "liquidity_score": round(liquidity_score, 3),
        "max_drawdown_90d": round(max_dd_90 * 100, 3),  # as %
        "adv_20": round(adv, 0),
    }
