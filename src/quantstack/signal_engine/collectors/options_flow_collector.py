# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Options flow collector — async wrapper for the options_flow module.

Integrates the dealer-positioning signal computation (GEX, gamma flip,
DEX, max pain, IV skew, VRP, charm, vanna) into the SignalEngine
collector interface.

The collector fetches the current spot price from DataStore, then calls
the existing `collect_options_flow()` synchronous function via
asyncio.to_thread.

Returns {} if:
- Alpaca credentials are not set (options chain requires subscription)
- No OHLCV data available for the symbol (can't determine spot price)
- Any error during chain fetch or signal computation
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.signal_engine.collectors.options_flow import collect_options_flow


async def collect_options_flow_async(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Async collector interface for options flow signals.

    Returns a dict with keys prefixed with 'opt_':
        opt_gex              : float — net Gamma Exposure (positive = mean-reverting regime)
        opt_gamma_flip       : float — strike where GEX crosses zero (key S/R level)
        opt_above_gamma_flip : int (0|1) — whether spot is above gamma flip
        opt_dex              : float — net Delta Exposure (directional bias)
        opt_max_pain         : float — max pain strike for nearest expiry
        opt_iv_skew          : float — OTM put IV minus OTM call IV
        opt_iv_skew_zscore   : float — skew z-score vs historical window
        opt_vrp              : float — ATM IV minus 30-day realized vol
        opt_charm            : float — aggregate delta decay rate
        opt_vanna            : float — aggregate dDelta/dVol
        opt_ehd              : float — Expected Hedging Demand
        opt_os_ratio         : float — options volume / OI (informed flow proxy)
        opt_avemoney         : float — dollar-weighted moneyness
        opt_n_contracts      : int — contracts in chain
        opt_call_oi          : int — total call open interest
        opt_put_oi           : int — total put open interest

    Returns {} if data unavailable or credentials missing.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_sync, symbol, store),
            timeout=15.0,  # options chain fetch can be slow
        )
    except asyncio.TimeoutError:
        logger.debug(f"[options_flow] {symbol}: timeout after 15s")
        return {}
    except Exception as exc:
        logger.debug(f"[options_flow] {symbol}: {exc}")
        return {}


def _collect_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    """Synchronous options flow collection."""
    # Get spot price from latest OHLCV
    df = store.load_ohlcv(symbol, Timeframe.D1)
    if df is None or df.empty:
        return {}

    spot = float(df["close"].iloc[-1])
    if spot <= 0:
        return {}

    # Compute 30-day realized vol for VRP calculation
    realized_vol_30d = None
    if len(df) >= 30:
        returns = df["close"].pct_change().dropna()
        if len(returns) >= 20:
            realized_vol_30d = float(returns.tail(30).std() * (252**0.5))

    # Load historical IV skew observations for z-score (if available)
    historical_skews = _load_historical_skews(symbol, store)

    return collect_options_flow(
        symbol=symbol,
        spot=spot,
        realized_vol_30d=realized_vol_30d,
    )


def _load_historical_skews(symbol: str, store: DataStore) -> list[float] | None:
    """Load cached historical IV skew observations for z-score computation."""
    # IV skew history would be stored from previous collector runs.
    # For now, return None — the z-score field will be None.
    # Future: persist daily skew snapshots to PostgreSQL and load last 30 here.
    return None
