# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Job Postings Collector — hiring surges and mass layoffs as leading indicators.

Queries the ``job_posting_metrics`` table for monthly open-position counts and
computes a year-over-year growth signal.

Signal rationale:
- Hiring is a leading indicator of revenue expectations.  Companies hire 3–6
  months ahead of expected demand.  Mass layoffs signal margin pressure or
  declining business lines.
- Academic evidence (Belo, Lin & Bazdresch 2014 — "Labor Hiring, Investment,
  and Stock Return Predictability") shows abnormal hiring growth predicts
  positive returns with a 3–6 month lead.
- This is a slow-moving signal: monthly granularity, 3–6 month lead time.
  Confidence is moderate because hiring decisions reflect management
  expectations, which can be wrong.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantstack.db import pg_conn

_COLLECTOR_TIMEOUT = 8.0  # seconds
_MIN_MONTHS_REQUIRED = 6  # need at least 6 months for meaningful YoY comparison

# YoY growth thresholds for signal generation
_HIRING_SURGE_THRESHOLD = 0.20   # +20% YoY → bullish signal
_MASS_LAYOFF_THRESHOLD = -0.20   # -20% YoY → bearish signal

_POSTINGS_SQL = """
SELECT
    month_start,        -- date, first day of the month
    open_positions      -- integer, total job openings
FROM job_posting_metrics
WHERE symbol = %s
  AND month_start >= %s
ORDER BY month_start ASC
"""


async def collect_job_posting_signals(
    symbol: str,
    bars_df: Any,
    regime: dict | None = None,
) -> dict | None:
    """Compute hiring-trend signal for *symbol*.

    Returns a signal dict or None when insufficient data.
    Never raises — returns None on any failure.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_sync, symbol),
            timeout=_COLLECTOR_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[job_postings] %s: timed out after %.0fs", symbol, _COLLECTOR_TIMEOUT
        )
        return None
    except Exception as exc:
        logger.warning("[job_postings] %s: %s: %s", symbol, type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_sync(symbol: str) -> dict | None:
    monthly_openings = _fetch_posting_data(symbol)
    if len(monthly_openings) < _MIN_MONTHS_REQUIRED:
        return None

    signal_value, confidence = _compute_hiring_signal(monthly_openings)

    # Latest data point for the response
    _, current_openings = monthly_openings[-1]

    # YoY growth for reporting
    yoy_growth = _yoy_growth(monthly_openings)

    return {
        "signal_value": signal_value,
        "confidence": confidence,
        "yoy_growth_pct": round(yoy_growth * 100, 2) if yoy_growth is not None else None,
        "current_openings": current_openings,
    }


def _fetch_posting_data(symbol: str) -> list[tuple[date, int]]:
    """Query job_posting_metrics for the last 18 months.

    Returns chronologically-ordered list of (month_start, open_positions).
    We fetch 18 months to ensure a full 12-month lookback for YoY comparison.
    """
    cutoff = date.today() - timedelta(days=548)  # ~18 months
    try:
        with pg_conn() as conn:
            rows = conn.execute(_POSTINGS_SQL, [symbol, cutoff])
            if rows is None:
                return []
            return [(r[0], int(r[1])) for r in rows]
    except Exception as exc:
        logger.warning("[job_postings] DB query failed: %s", exc)
        return []


def _yoy_growth(monthly_openings: list[tuple[date, int]]) -> float | None:
    """Compute year-over-year growth from the monthly openings series.

    Compares the latest month to the same month one year prior (or the
    closest available month within 11-13 months).
    """
    if len(monthly_openings) < 2:
        return None

    latest_date, latest_count = monthly_openings[-1]
    target_date = latest_date.replace(year=latest_date.year - 1)

    # Find the closest data point to 12 months ago
    best_match: tuple[date, int] | None = None
    best_delta = timedelta(days=999)
    for m_date, m_count in monthly_openings[:-1]:
        delta = abs(m_date - target_date)
        if delta < best_delta:
            best_delta = delta
            best_match = (m_date, m_count)

    if best_match is None or best_match[1] == 0:
        return None

    # Only use if within 60 days of the target (roughly same quarter)
    if best_delta > timedelta(days=60):
        return None

    return (latest_count - best_match[1]) / best_match[1]


def _compute_hiring_signal(
    monthly_openings: list[tuple[date, int]],
) -> tuple[float, float]:
    """Compute directional signal and confidence from hiring trends.

    Signal mapping (YoY growth → signal_value):
      - growth > +20%  → bullish: linearly scaled from +0.3 to +0.7
      - growth < -20%  → bearish: linearly scaled from -0.3 to -0.7
      - between ±20%   → neutral: linearly scaled from -0.3 to +0.3
      - Clamped to [-1, 1]

    Confidence scaling rationale:
      - 3–6 month lead time means signal is slow but directionally reliable.
      - Base confidence: 0.3 (hiring is a noisy proxy for future revenue).
      - Scales up to 0.5 with data completeness (12+ months of history).
      - Additional +0.1 when the YoY move is extreme (> 40% either direction),
        since extreme moves are harder to reverse.
      - Cap at 0.6 — slow signals are never high-conviction for timing.
    """
    yoy = _yoy_growth(monthly_openings)
    if yoy is None:
        return 0.0, 0.0

    # Signal value: map YoY growth to [-1, 1] with threshold zones
    if yoy > _HIRING_SURGE_THRESHOLD:
        # Bullish: scale 20% → +0.3, 60%+ → +0.7
        excess = (yoy - _HIRING_SURGE_THRESHOLD)
        signal_value = 0.3 + min(excess / 0.40, 1.0) * 0.4
    elif yoy < _MASS_LAYOFF_THRESHOLD:
        # Bearish: scale -20% → -0.3, -60%+ → -0.7
        excess = abs(yoy) - abs(_MASS_LAYOFF_THRESHOLD)
        signal_value = -(0.3 + min(excess / 0.40, 1.0) * 0.4)
    else:
        # Neutral zone: linear interpolation within ±20% → ±0.3
        signal_value = (yoy / _HIRING_SURGE_THRESHOLD) * 0.3

    signal_value = max(-1.0, min(1.0, signal_value))

    # Confidence: base 0.3, scales with data completeness
    data_months = len(monthly_openings)
    completeness = min(data_months / 12.0, 1.0)
    confidence = 0.3 + 0.2 * completeness

    # Boost for extreme moves (hard to fake or reverse quickly)
    if abs(yoy) > 0.40:
        confidence = min(0.6, confidence + 0.1)

    return round(signal_value, 4), round(confidence, 4)
