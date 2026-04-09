# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Congressional Trades Collector — tracks US congressional stock trades.

Reads STOCK Act disclosure data from the ``congressional_trades`` table and
produces a directional signal based on net buy/sell activity.  Congressional
trades are disclosed 30–45 days after execution, so this is a medium-lag
signal — useful for confirming conviction rather than timing entries.

Signal rationale:
- Members of Congress have asymmetric information access (committee
  assignments, briefings, pending legislation).  Academic evidence
  (Ziobrowski et al. 2004, Eggers & Hainmueller 2013) shows their
  portfolios outperform benchmarks by 6–10% annually.
- Net buy skew across multiple members is a stronger signal than any
  single transaction.
- Confidence is intentionally capped at 0.7 because disclosure lag
  means the information edge may already be priced in by the time we
  see the filing.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger

from quantstack.db import pg_conn

_COLLECTOR_TIMEOUT = 8.0  # seconds
_DEFAULT_LOOKBACK_DAYS = 30

# Traders with historically documented outperformance get a weight boost.
# This set would be maintained from periodic accuracy audits stored in DB.
_HIGH_ACCURACY_BOOST = 1.3

_TRADES_SQL = """
SELECT
    trader_name,
    transaction_type,   -- 'buy' | 'sell'
    amount_lower,       -- reported range lower bound (USD)
    amount_upper,       -- reported range upper bound (USD)
    disclosed_at,
    traded_at,
    historical_accuracy -- NULL if not yet scored
FROM congressional_trades
WHERE symbol = %s
  AND traded_at >= %s
ORDER BY traded_at DESC
"""


async def collect_congressional_signals(
    symbol: str,
    bars_df: Any,
    regime: dict | None = None,
) -> dict | None:
    """Fetch and score recent congressional trading activity for *symbol*.

    Returns a signal dict or None when no data is available.
    Never raises — returns None on any failure.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_sync, symbol, _DEFAULT_LOOKBACK_DAYS),
            timeout=_COLLECTOR_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[congressional] %s: timed out after %.0fs", symbol, _COLLECTOR_TIMEOUT
        )
        return None
    except Exception as exc:
        logger.warning("[congressional] %s: %s: %s", symbol, type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_sync(symbol: str, lookback_days: int) -> dict | None:
    trades = _fetch_congressional_trades(symbol, lookback_days)
    if not trades:
        return None

    signal_value, confidence = _compute_signal(trades)

    buy_trades = [t for t in trades if t["transaction_type"] == "buy"]
    sell_trades = [t for t in trades if t["transaction_type"] == "sell"]

    # Notable traders: unique names, limited to top 5 by recency
    seen: set[str] = set()
    notable: list[str] = []
    for t in trades:
        name = t["trader_name"]
        if name not in seen:
            seen.add(name)
            notable.append(name)
        if len(notable) >= 5:
            break

    return {
        "signal_value": signal_value,
        "confidence": confidence,
        "net_buys": len(buy_trades),
        "net_sells": len(sell_trades),
        "notable_traders": notable,
    }


def _fetch_congressional_trades(
    symbol: str, lookback_days: int = _DEFAULT_LOOKBACK_DAYS
) -> list[dict]:
    """Query congressional_trades table for recent activity on *symbol*."""
    cutoff = date.today() - timedelta(days=lookback_days)
    try:
        with pg_conn() as conn:
            rows = conn.execute(_TRADES_SQL, [symbol, cutoff])
            if rows is None:
                return []
            return [
                {
                    "trader_name": r[0],
                    "transaction_type": r[1],
                    "amount_lower": r[2],
                    "amount_upper": r[3],
                    "disclosed_at": r[4],
                    "traded_at": r[5],
                    "historical_accuracy": r[6],
                }
                for r in rows
            ]
    except Exception as exc:
        logger.warning("[congressional] DB query failed: %s", exc)
        return []


def _compute_signal(trades: list[dict]) -> tuple[float, float]:
    """Compute directional signal and confidence from congressional trades.

    Signal: weighted net buy ratio in [-1, 1].
      - Each trade contributes +1 (buy) or -1 (sell), optionally scaled by
        the trader's historical accuracy score.
      - Final value = weighted_sum / total_weight, clamped to [-1, 1].

    Confidence: based on trade count.
      - 1 trade  → 0.15  (anecdotal)
      - 3 trades → 0.35  (moderate)
      - 5+ trades→ 0.55  (solid cluster)
      - Cap at 0.7: disclosure lag means edge may be partially priced in.
    """
    weighted_sum = 0.0
    total_weight = 0.0

    for t in trades:
        direction = 1.0 if t["transaction_type"] == "buy" else -1.0

        # Weight by historical accuracy when available
        accuracy = t.get("historical_accuracy")
        weight = _HIGH_ACCURACY_BOOST if accuracy and accuracy > 0.55 else 1.0

        weighted_sum += direction * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0, 0.0

    signal_value = max(-1.0, min(1.0, weighted_sum / total_weight))

    # Confidence scales with trade count — more independent observations
    # increase our certainty that the signal is real, not noise.
    # Capped at 0.7 because of the 30-45 day disclosure lag.
    trade_count = len(trades)
    if trade_count >= 5:
        confidence = 0.55
    elif trade_count >= 3:
        confidence = 0.35
    else:
        confidence = 0.15

    # Boost confidence slightly when signal is strongly directional
    # (all buys or all sells suggests coordinated conviction)
    abs_signal = abs(signal_value)
    if abs_signal > 0.8:
        confidence = min(0.7, confidence + 0.15)

    return round(signal_value, 4), round(confidence, 4)
