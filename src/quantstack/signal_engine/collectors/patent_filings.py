# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Patent Filings Collector — patent acceleration as R&D strength indicator.

Queries the ``patent_filings`` table for the last 24 months of filing counts
and computes a filing-acceleration signal.

Signal rationale:
- Patent filing velocity is a proxy for R&D investment and innovation
  pipeline strength.  Cohen, Diether & Malloy (2013 — "Misvaluing
  Innovation") show that patent-based measures predict returns because
  markets systematically undervalue R&D output.
- Filing acceleration (current-year count / prior-year count) captures
  changes in innovation intensity.  Steady filing rates are noise;
  acceleration or deceleration is the signal.
- Very long lead time (6–12 months before market impact) and high noise
  (patent quality varies wildly).  Confidence is therefore capped at 0.5.
  This signal is best used as a slow confirming factor, not a timing tool.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantstack.db import pg_conn

_COLLECTOR_TIMEOUT = 8.0  # seconds
_MIN_MONTHS_REQUIRED = 12  # Need at least 12 months for YoY acceleration

_FILINGS_SQL = """
SELECT
    filing_month,   -- date, first day of the month
    filing_count    -- integer
FROM patent_filings
WHERE symbol = %s
  AND filing_month >= %s
ORDER BY filing_month ASC
"""


async def collect_patent_signals(
    symbol: str,
    bars_df: Any,
    regime: dict | None = None,
) -> dict | None:
    """Compute patent-filing-acceleration signal for *symbol*.

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
            "[patent_filings] %s: timed out after %.0fs", symbol, _COLLECTOR_TIMEOUT
        )
        return None
    except Exception as exc:
        logger.warning(
            "[patent_filings] %s: %s: %s", symbol, type(exc).__name__, exc
        )
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_sync(symbol: str) -> dict | None:
    monthly_filings = _fetch_filing_data(symbol)
    if len(monthly_filings) < _MIN_MONTHS_REQUIRED:
        return None

    signal_value, confidence = _compute_patent_signal(monthly_filings)

    # Compute acceleration and recent count for the response
    recent_12 = sum(count for _, count in monthly_filings[-12:])
    prior_12 = sum(count for _, count in monthly_filings[:-12]) if len(monthly_filings) > 12 else 0
    acceleration = recent_12 / prior_12 if prior_12 > 0 else 0.0

    return {
        "signal_value": signal_value,
        "confidence": confidence,
        "filing_acceleration": round(acceleration, 3),
        "recent_count": recent_12,
    }


def _fetch_filing_data(symbol: str) -> list[tuple[date, int]]:
    """Query patent_filings for the last 24 months.

    Returns chronologically-ordered list of (filing_month, filing_count).
    """
    cutoff = date.today() - timedelta(days=730)  # ~24 months
    try:
        with pg_conn() as conn:
            rows = conn.execute(_FILINGS_SQL, [symbol, cutoff])
            if rows is None:
                return []
            return [(r[0], int(r[1])) for r in rows]
    except Exception as exc:
        logger.warning("[patent_filings] DB query failed: %s", exc)
        return []


def _compute_patent_signal(
    monthly_filings: list[tuple[date, int]],
) -> tuple[float, float]:
    """Compute filing-acceleration signal and confidence.

    Signal: derived from the ratio of recent-12-month filings to prior-12-month
    filings (the "acceleration ratio").
      - ratio > 1.0 → R&D accelerating → bullish
      - ratio < 1.0 → R&D decelerating → bearish
      - ratio ≈ 1.0 → steady state → neutral

    Mapping (acceleration → signal_value):
      - ratio 1.5+ → +0.5 to +0.7 (strong acceleration)
      - ratio 1.0–1.5 → 0.0 to +0.5 (mild acceleration)
      - ratio 0.5–1.0 → -0.5 to 0.0 (mild deceleration)
      - ratio < 0.5 → -0.5 to -0.7 (severe deceleration)
      - Clamped to [-1, 1]

    Confidence scaling rationale:
      - Very noisy signal: patent quality varies, filing strategy differs by
        company, and the 6–12 month lead time introduces large uncertainty.
      - Base: 0.2 with minimum data.
      - Scales up to 0.4 with data completeness (full 24 months).
      - Boosted +0.1 when acceleration is extreme (> 2x or < 0.5x).
      - Hard cap at 0.5 — this is always a supplementary signal.
    """
    if len(monthly_filings) < _MIN_MONTHS_REQUIRED:
        return 0.0, 0.0

    # Split into recent 12 months and prior period
    recent_12 = sum(count for _, count in monthly_filings[-12:])
    prior_months = monthly_filings[:-12]

    if not prior_months:
        # Only 12 months of data — can compute trends but not acceleration
        return 0.0, 0.0

    # Annualize prior period to make comparison fair
    prior_total = sum(count for _, count in prior_months)
    prior_months_count = len(prior_months)
    prior_annualized = (prior_total / prior_months_count) * 12 if prior_months_count > 0 else 0

    if prior_annualized < 1:
        # Near-zero prior filings — can't compute meaningful ratio.
        # If recent filings exist, treat as weak bullish (new filer).
        if recent_12 > 0:
            return 0.2, 0.15
        return 0.0, 0.0

    acceleration = recent_12 / prior_annualized

    # Map acceleration to signal value
    if acceleration >= 1.5:
        # Strong acceleration: 1.5 → +0.5, 2.0+ → +0.7
        excess = min((acceleration - 1.5) / 0.5, 1.0)
        signal_value = 0.5 + excess * 0.2
    elif acceleration >= 1.0:
        # Mild acceleration: 1.0 → 0.0, 1.5 → +0.5
        signal_value = (acceleration - 1.0) / 0.5 * 0.5
    elif acceleration >= 0.5:
        # Mild deceleration: 1.0 → 0.0, 0.5 → -0.5
        signal_value = -((1.0 - acceleration) / 0.5 * 0.5)
    else:
        # Severe deceleration: 0.5 → -0.5, 0.0 → -0.7
        excess = min((0.5 - acceleration) / 0.5, 1.0)
        signal_value = -(0.5 + excess * 0.2)

    signal_value = max(-1.0, min(1.0, signal_value))

    # Confidence: base 0.2, scales with data completeness
    data_months = len(monthly_filings)
    completeness = min(data_months / 24.0, 1.0)
    confidence = 0.2 + 0.2 * completeness

    # Boost for extreme acceleration/deceleration
    if acceleration > 2.0 or acceleration < 0.5:
        confidence = min(0.5, confidence + 0.1)

    return round(signal_value, 4), round(confidence, 4)
