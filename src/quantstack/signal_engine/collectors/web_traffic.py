# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Web Traffic Collector — monthly unique visitor deltas as leading indicator.

Queries the ``web_traffic_metrics`` table for historical monthly unique-visitor
counts and computes a z-scored month-over-month growth signal.

Signal rationale:
- Web traffic to a company's primary domain is a proxy for customer demand
  and brand momentum.  Academic work (Da, Engelberg & Gao 2011 — "In Search
  of Attention") demonstrates Google search volume predicts short-term returns.
  Direct site traffic is a higher-quality version of the same thesis.
- This is inherently a noisy signal: traffic spikes can reflect marketing
  campaigns, viral events, or seasonality rather than fundamental demand.
- Confidence is therefore set to a low base (0.4) and only increases with
  data completeness.  Even at maximum, it stays below 0.6.
"""

from __future__ import annotations

import asyncio
import statistics
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantstack.db import pg_conn

_COLLECTOR_TIMEOUT = 8.0  # seconds
_MIN_MONTHS_REQUIRED = 3

_TRAFFIC_SQL = """
SELECT
    month_start,       -- date, first day of the month
    unique_visitors    -- bigint
FROM web_traffic_metrics
WHERE symbol = %s
  AND month_start >= %s
ORDER BY month_start ASC
"""


async def collect_web_traffic_signals(
    symbol: str,
    bars_df: Any,
    regime: dict | None = None,
) -> dict | None:
    """Compute web-traffic-based signal for *symbol*.

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
            "[web_traffic] %s: timed out after %.0fs", symbol, _COLLECTOR_TIMEOUT
        )
        return None
    except Exception as exc:
        logger.warning("[web_traffic] %s: %s: %s", symbol, type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_sync(symbol: str) -> dict | None:
    monthly_visitors = _fetch_traffic_data(symbol)
    if len(monthly_visitors) < _MIN_MONTHS_REQUIRED:
        return None

    signal_value, confidence = _compute_traffic_signal(monthly_visitors)

    current = monthly_visitors[-1]
    prev = monthly_visitors[-2]
    mom_growth = (current - prev) / prev if prev > 0 else 0.0

    return {
        "signal_value": signal_value,
        "confidence": confidence,
        "mom_growth_pct": round(mom_growth * 100, 2),
        "visitors_millions": round(current / 1_000_000, 2),
    }


def _fetch_traffic_data(symbol: str) -> list[float]:
    """Query web_traffic_metrics for the last 12 months of visitor counts.

    Returns a chronologically-ordered list of monthly unique visitor counts.
    """
    cutoff = date.today() - timedelta(days=365)
    try:
        with pg_conn() as conn:
            rows = conn.execute(_TRAFFIC_SQL, [symbol, cutoff])
            if rows is None:
                return []
            return [float(r[1]) for r in rows]
    except Exception as exc:
        logger.warning("[web_traffic] DB query failed: %s", exc)
        return []


def _compute_traffic_signal(monthly_visitors: list[float]) -> tuple[float, float]:
    """Compute z-scored MoM growth signal and confidence.

    Signal: z-score of the latest MoM growth rate against the trailing
    12-month history of MoM growth rates.  Clamped to [-1, 1].
      - z > 0 → traffic accelerating faster than normal → bullish
      - z < 0 → traffic decelerating → bearish

    Confidence scaling rationale:
      - Base: 0.4 (web traffic is inherently noisy — bots, campaigns, seasonality)
      - Scaled by data completeness: if we have 12 months of history, we get
        the full 0.4.  Fewer months → proportionally lower.
      - Boosted slightly (+0.1) when the z-score magnitude is large (> 1.5),
        indicating a statistically unusual move.
      - Hard cap at 0.55 — this signal type is never high-conviction on its own.
    """
    if len(monthly_visitors) < _MIN_MONTHS_REQUIRED:
        return 0.0, 0.0

    # Compute MoM growth rates for each consecutive pair
    mom_rates: list[float] = []
    for i in range(1, len(monthly_visitors)):
        prev = monthly_visitors[i - 1]
        curr = monthly_visitors[i]
        if prev > 0:
            mom_rates.append((curr - prev) / prev)

    if len(mom_rates) < 2:
        return 0.0, 0.0

    latest_growth = mom_rates[-1]

    # Z-score the latest growth vs history
    mean_growth = statistics.mean(mom_rates)
    stdev_growth = statistics.stdev(mom_rates) if len(mom_rates) >= 3 else 0.0

    if stdev_growth < 1e-9:
        # All months had identical growth — no signal
        z_score = 0.0
    else:
        z_score = (latest_growth - mean_growth) / stdev_growth

    # Clamp signal to [-1, 1]
    signal_value = max(-1.0, min(1.0, z_score / 3.0))

    # Confidence: base 0.4, scaled by data completeness
    data_completeness = min(len(monthly_visitors) / 12.0, 1.0)
    confidence = 0.4 * data_completeness

    # Boost for statistically unusual moves
    if abs(z_score) > 1.5:
        confidence = min(0.55, confidence + 0.1)

    return round(signal_value, 4), round(confidence, 4)
