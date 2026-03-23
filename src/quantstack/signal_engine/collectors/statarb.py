# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Statistical arbitrage collector — pairs cointegration and mean-reversion z-score.

Computes spread z-score, ADF cointegration test, and mean-reversion half-life
for known ETF pairs.  All data comes from the local DataStore — no network
call in the live trading path.  Returns {} when the symbol is not in any
known pair or data is insufficient.  Never raises.
"""

import asyncio
from typing import Any

import numpy as np
from loguru import logger
from statsmodels.tsa.stattools import adfuller

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore

# Known pairs for stat-arb analysis. Includes sector ETF pairs, broad market
# pairs, and commodity/equity pairs (GLD/GDX). Key can appear on either side —
# the lookup normalises both directions.
ARBITRAGE_PAIRS: dict[str, str] = {
    "XLK": "QQQ",
    "QQQ": "XLK",
    "XLF": "KBE",
    "KBE": "XLF",
    "XLE": "OIH",
    "OIH": "XLE",
    "SPY": "IWM",
    "IWM": "SPY",
    "GLD": "GDX",
    "GDX": "GLD",
    "TLT": "IEF",
    "IEF": "TLT",
}

_LOOKBACK = 252  # 1 year for ADF / half-life estimation
_ZSCORE_WINDOW = 60  # 60-day rolling mean/std for z-score
_MIN_BARS = 60


async def collect_statarb(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Compute stat-arb spread metrics for *symbol* vs its ETF pair.

    Returns a dict with keys:
        pair_symbol      : str | None — the paired ETF
        spread_zscore    : float | None — z-score of current spread vs 60-day mean
        half_life_days   : float | None — estimated mean-reversion half-life
        is_cointegrated  : bool — ADF test p-value < 0.05
        adf_pvalue       : float | None
        statarb_signal   : str — "long_spread" (z < -2), "short_spread" (z > 2), "neutral"

    Returns {} if symbol is not in a known pair or data is insufficient.
    """
    try:
        return await asyncio.to_thread(_collect_statarb_sync, symbol, store)
    except Exception as exc:
        logger.debug(f"[statarb] {symbol}: {exc} — returning empty")
        return {}


def _collect_statarb_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    """Synchronous stat-arb computation — called via asyncio.to_thread."""
    pair = ARBITRAGE_PAIRS.get(symbol.upper())
    if pair is None:
        return {}

    # Load OHLCV for both legs
    df_a = store.load_ohlcv(symbol, Timeframe.D1)
    df_b = store.load_ohlcv(pair, Timeframe.D1)

    if df_a is None or df_b is None:
        return {}
    if len(df_a) < _MIN_BARS or len(df_b) < _MIN_BARS:
        return {}

    # Align on date index — inner join to keep only overlapping dates
    close_a = df_a[["close"]].rename(columns={"close": "a"}).iloc[-_LOOKBACK:]
    close_b = df_b[["close"]].rename(columns={"close": "b"}).iloc[-_LOOKBACK:]
    merged = close_a.join(close_b, how="inner")

    if len(merged) < _MIN_BARS:
        return {}

    a = merged["a"].values
    b = merged["b"].values

    # --- Spread: log-price spread (a - hedge_ratio * b) ---
    log_a = np.log(a)
    log_b = np.log(b)

    # OLS hedge ratio: regress log_a on log_b
    hedge_ratio = _ols_slope(log_b, log_a)
    spread = log_a - hedge_ratio * log_b

    # --- Z-score (rolling 60-day window) ---
    window = min(_ZSCORE_WINDOW, len(spread))
    recent_spread = spread[-window:]
    spread_mean = float(np.mean(recent_spread))
    spread_std = float(np.std(recent_spread, ddof=1))

    if spread_std == 0:
        zscore = 0.0
    else:
        zscore = float((spread[-1] - spread_mean) / spread_std)

    # --- ADF test for cointegration ---
    adf_pvalue = _run_adf(spread)
    is_cointegrated = adf_pvalue is not None and adf_pvalue < 0.05

    # --- Half-life of mean reversion ---
    half_life = _compute_half_life(spread)

    # --- Signal classification ---
    if zscore < -2.0:
        signal = "long_spread"
    elif zscore > 2.0:
        signal = "short_spread"
    else:
        signal = "neutral"

    return {
        "pair_symbol": pair,
        "spread_zscore": round(zscore, 4),
        "half_life_days": round(half_life, 2) if half_life is not None else None,
        "is_cointegrated": is_cointegrated,
        "adf_pvalue": round(adf_pvalue, 6) if adf_pvalue is not None else None,
        "statarb_signal": signal,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS slope: y = alpha + beta * x. Returns beta."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 1.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _run_adf(spread: np.ndarray) -> float | None:
    """Run Augmented Dickey-Fuller test. Returns p-value or None on failure."""
    try:
        result = adfuller(spread, maxlag=None, autolag="AIC")
        return float(result[1])  # p-value
    except Exception as exc:
        logger.debug(f"[statarb] ADF test failed: {exc}")
        return None


def _compute_half_life(spread: np.ndarray) -> float | None:
    """Estimate mean-reversion half-life via OLS on lagged spread.

    Model: delta_spread = alpha + beta * spread_lag + epsilon
    Half-life = -log(2) / beta

    Returns None if beta >= 0 (not mean-reverting) or insufficient data.
    """
    if len(spread) < 3:
        return None

    spread_lag = spread[:-1]
    delta_spread = np.diff(spread)

    # OLS: delta_spread = alpha + beta * spread_lag
    beta = _ols_slope(spread_lag, delta_spread)

    if beta >= 0:
        # Not mean-reverting
        return None

    half_life = -np.log(2) / beta
    # Sanity: half-life should be positive and reasonable (< 252 trading days)
    if half_life <= 0 or half_life > 252:
        return None

    return float(half_life)
