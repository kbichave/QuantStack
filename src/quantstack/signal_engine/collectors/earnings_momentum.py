# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Earnings momentum collector — surprise streaks, drift detection, and
forward-looking earnings timing.

Reads from the ``earnings_calendar`` table (via ``store.load_earnings_calendar``)
and computes:
  - Consecutive beats/misses from the most recent quarter
  - Average surprise_pct over last 4 quarters
  - Post-earnings announcement drift (PEAD) detection
  - Days-to-next-earnings estimate from quarterly cadence
  - Composite momentum score in [-1, 1]

Design invariants:
- Returns {} when earnings data is missing or all surprise_pct are NULL.
- Never raises — all errors are caught and logged.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from quantstack.data.storage import DataStore
from quantstack.signal_engine.staleness import check_freshness

_DRIFT_SURPRISE_THRESHOLD = 5.0  # |surprise_pct| > 5% triggers drift
_DRIFT_WINDOW_DAYS = 30  # drift active for 30 days after report
_MAX_QUARTERS = 12


async def collect_earnings_momentum(
    symbol: str, store: DataStore
) -> dict[str, Any]:
    """Compute earnings momentum signal from earnings calendar data."""
    if not check_freshness(symbol, "earnings_history", max_days=90):
        return {}
    try:
        return await asyncio.to_thread(
            _collect_earnings_momentum_sync, symbol, store
        )
    except Exception as exc:
        logger.warning(
            f"[earnings_momentum] {symbol}: {type(exc).__name__} — returning empty"
        )
        return {}


def _collect_earnings_momentum_sync(
    symbol: str, store: DataStore
) -> dict[str, Any]:
    """Synchronous earnings momentum computation."""
    if not hasattr(store, "load_earnings_calendar"):
        logger.debug("[earnings_momentum] store missing load_earnings_calendar")
        return {}

    df = store.load_earnings_calendar(symbol=symbol)
    if df is None or df.empty:
        return {}

    # Ensure we have surprise_pct; drop rows where it's null
    if "surprise_pct" not in df.columns:
        return {}

    df = df.dropna(subset=["surprise_pct"]).copy()
    if df.empty:
        return {}

    # Ensure report_date is datetime for computation
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"])
    else:
        return {}

    # Sort ascending by report_date, take last N quarters
    df = df.sort_values("report_date").tail(_MAX_QUARTERS).reset_index(drop=True)

    # --- Consecutive beats / misses ---
    consecutive_beats = 0
    consecutive_misses = 0
    # Walk backwards from most recent
    for i in range(len(df) - 1, -1, -1):
        sp = float(df.loc[i, "surprise_pct"])
        if sp > 0:
            if consecutive_misses > 0:
                break
            consecutive_beats += 1
        elif sp < 0:
            if consecutive_beats > 0:
                break
            consecutive_misses += 1
        else:
            break  # exact meet — streak ends

    # --- Average surprise_pct over last 4 quarters ---
    last_4 = df.tail(4)
    avg_surprise_pct_4q = float(last_4["surprise_pct"].mean())

    # --- Drift detection ---
    last_row = df.iloc[-1]
    last_surprise_pct = float(last_row["surprise_pct"])
    last_report_date = pd.Timestamp(last_row["report_date"])
    days_since_last = (pd.Timestamp(date.today()) - last_report_date).days

    drift_active = (
        abs(last_surprise_pct) > _DRIFT_SURPRISE_THRESHOLD
        and days_since_last <= _DRIFT_WINDOW_DAYS
    )

    # --- Days to next earnings (estimate from quarterly cadence) ---
    days_to_next: int | None = None
    if len(df) >= 2:
        intervals = df["report_date"].diff().dropna().dt.days
        avg_interval = float(intervals.mean())
        if avg_interval > 0:
            est_next = last_report_date + timedelta(days=avg_interval)
            days_to_next = max(0, (est_next - pd.Timestamp(date.today())).days)
    # Single quarter: can't estimate
    if len(df) < 2:
        days_to_next = None

    # --- Composite momentum score [-1, 1] ---
    # 50% streak component + 50% magnitude component
    max_streak = max(consecutive_beats, consecutive_misses)
    streak_direction = 1 if consecutive_beats > 0 else (-1 if consecutive_misses > 0 else 0)
    # Normalize streak: 4+ consecutive is max signal
    streak_component = streak_direction * min(max_streak / 4.0, 1.0)

    # Magnitude: avg_surprise_pct clamped to [-20, 20], normalized to [-1, 1]
    clamped_surprise = max(-20.0, min(20.0, avg_surprise_pct_4q))
    magnitude_component = clamped_surprise / 20.0

    earnings_momentum_score = round(
        max(-1.0, min(1.0, 0.5 * streak_component + 0.5 * magnitude_component)),
        4,
    )

    return {
        "consecutive_beats": consecutive_beats,
        "consecutive_misses": consecutive_misses,
        "avg_surprise_pct_4q": round(avg_surprise_pct_4q, 4),
        "drift_active": drift_active,
        "days_since_last_earnings": days_since_last,
        "days_to_next_earnings": days_to_next,
        "last_surprise_pct": round(last_surprise_pct, 4),
        "earnings_momentum_score": earnings_momentum_score,
    }
