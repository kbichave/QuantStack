"""Short interest collector.

Computes short interest ratio, days-to-cover, and squeeze candidate detection.
Data source: FINRA short interest (bi-monthly, ~2-week reporting lag) or
Alpaca market data if available.

Limitation: 2-week lag means this is useful for swing/investment thesis
timeframes, not intraday timing.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from quantstack.data.storage import DataStore


async def collect_short_interest(symbol: str, store: DataStore) -> dict[str, Any]:
    """Fetch short interest data for a symbol.

    Returns dict with short_interest_ratio, days_to_cover, squeeze_candidate.
    Returns {} on failure (collector contract).
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_sync, symbol),
            timeout=10.0,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.debug(f"[short_interest] {symbol}: {type(exc).__name__}")
        return {}


def _collect_sync(symbol: str) -> dict[str, Any]:
    """Synchronous collection — placeholder for FINRA/Alpaca integration."""
    return {}


# ---------------------------------------------------------------------------
# Computation Functions (pure, testable)
# ---------------------------------------------------------------------------


def compute_short_interest_metrics(
    si_shares: int,
    float_shares: int,
    avg_daily_volume: int,
    si_change_mom: float = 0.0,
) -> dict[str, Any]:
    """Compute short interest metrics from raw data.

    Returns dict with short_interest_ratio, days_to_cover, squeeze_candidate.
    Handles edge cases (zero volume, zero float) without crashing.
    """
    # Short interest ratio
    if float_shares > 0:
        si_ratio = si_shares / float_shares
    else:
        si_ratio = None

    # Days to cover
    if avg_daily_volume > 0:
        dtc = si_shares / avg_daily_volume
    else:
        dtc = None

    # Squeeze candidate: SI > 20% and DTC > 5
    squeeze = False
    if si_ratio is not None and dtc is not None:
        squeeze = si_ratio > 0.20 and dtc > 5

    return {
        "short_interest_ratio": si_ratio,
        "days_to_cover": dtc,
        "si_change_mom": si_change_mom,
        "squeeze_candidate": squeeze,
    }
