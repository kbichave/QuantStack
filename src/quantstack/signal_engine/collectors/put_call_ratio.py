# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Put-call ratio collector — derives PCR signal from options_chains volume data.

Uses the ``load_options_volume_summary`` store method (aggregates put/call volume
by date) to compute a contrarian signal:
  - High PCR (>80th percentile) → excess fear → contrarian bullish
  - Low PCR (<20th percentile) → excess greed → contrarian bearish

Design invariants:
- Returns {} when options data is missing, stale, or below the minimum
  volume threshold (500 contracts).
- Guards division by zero (call_volume == 0 rows are excluded).
- Never raises — all errors are caught and logged.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantstack.data.storage import DataStore
from quantstack.signal_engine.staleness import check_freshness

_MIN_VOLUME_THRESHOLD = 500
_LOOKBACK_DAYS = 90
_MIN_HISTORY_DAYS = 20


async def collect_put_call_ratio(symbol: str, store: DataStore) -> dict[str, Any]:
    """Compute put-call ratio signal from options chain volume data."""
    if not check_freshness(symbol, "options_chains", max_days=3):
        return {}
    try:
        return await asyncio.to_thread(
            _collect_put_call_ratio_sync, symbol, store
        )
    except Exception as exc:
        logger.warning(
            f"[put_call_ratio] {symbol}: {type(exc).__name__} — returning empty"
        )
        return {}


def _collect_put_call_ratio_sync(
    symbol: str, store: DataStore
) -> dict[str, Any]:
    """Synchronous PCR computation."""
    end = date.today()
    start = end - timedelta(days=_LOOKBACK_DAYS)

    if not hasattr(store, "load_options_volume_summary"):
        logger.debug("[put_call_ratio] store missing load_options_volume_summary")
        return {}

    df = store.load_options_volume_summary(symbol, start, end)

    if df.empty:
        return {}

    # Check minimum volume threshold on the most recent day
    last_day = df.iloc[-1]
    total_vol = int(last_day["put_volume"]) + int(last_day["call_volume"])
    if total_vol < _MIN_VOLUME_THRESHOLD:
        return {}

    # Exclude rows where call_volume == 0 to guard division by zero
    df = df[df["call_volume"] > 0].copy()
    if df.empty:
        return {}

    df["pcr"] = df["put_volume"] / df["call_volume"]

    # 10-day SMA of PCR
    df["pcr_10d_sma"] = df["pcr"].rolling(10, min_periods=1).mean()

    today_pcr = float(df["pcr"].iloc[-1])
    today_sma = float(df["pcr_10d_sma"].iloc[-1])

    # Percentile-based contrarian signal (80th/20th of history)
    if len(df) < _MIN_HISTORY_DAYS:
        pcr_signal = 0
    else:
        pct = float((df["pcr_10d_sma"] <= today_sma).mean())
        if pct > 0.80:
            pcr_signal = 1  # contrarian bullish (excess fear)
        elif pct < 0.20:
            pcr_signal = -1  # contrarian bearish (excess greed)
        else:
            pcr_signal = 0

    # 30-day percentile
    last_30 = df["pcr"].tail(30)
    pcr_pct_30d = float((last_30 <= today_pcr).mean())

    return {
        "pcr_raw": round(today_pcr, 4),
        "pcr_10d_sma": round(today_sma, 4),
        "pcr_signal": pcr_signal,
        "pcr_percentile_30d": round(pcr_pct_30d, 4),
        "put_volume_total": int(last_day["put_volume"]),
        "call_volume_total": int(last_day["call_volume"]),
    }
