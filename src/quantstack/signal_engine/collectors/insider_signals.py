"""Insider trading signal collector.

Analyzes SEC Form 4 insider trading activity and produces a directional
signal score. Detects: cluster buys, C-suite buys, unusual transaction sizes.

Data source: SEC EDGAR Form 4 filings (free, machine-readable XML).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from quantstack.data.storage import DataStore
from quantstack.signal_engine.staleness import check_freshness

_CSUITE_TITLES = {"CEO", "CFO", "COO", "President", "Chief Executive Officer",
                  "Chief Financial Officer", "Chief Operating Officer"}
_EXERCISE_CODES = {"A", "M"}  # Award/exercise — exclude from sell analysis


async def collect_insider_signals(symbol: str, store: DataStore) -> dict[str, Any]:
    """Fetch and analyze insider trading for a symbol.

    Returns dict with insider_signal_score, signal_types, transactions, confidence.
    Returns {} on failure (collector contract).
    """
    if not check_freshness(symbol, "insider_trades", max_days=30):
        return {}
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_sync, symbol),
            timeout=10.0,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning(f"[insider_signals] {symbol}: {type(exc).__name__}")
        return {}


def _collect_sync(symbol: str) -> dict[str, Any]:
    """Synchronous collection — placeholder for SEC EDGAR integration."""
    # In production, this fetches from SEC EDGAR.
    # For now, return empty dict (no external calls in unit test context).
    return {}


# ---------------------------------------------------------------------------
# Signal Detection Functions (pure, testable)
# ---------------------------------------------------------------------------


def detect_cluster_buy(
    transactions: list[dict],
    window_days: int = 30,
    min_insiders: int = 3,
) -> bool:
    """Detect cluster buying: 3+ distinct insiders purchasing within window."""
    buys = [t for t in transactions if t.get("transaction_type") == "P"]
    if not buys:
        return False

    now = datetime.utcnow()
    recent_buyers = set()
    for t in buys:
        try:
            txn_date = datetime.fromisoformat(t["date"])
        except (ValueError, KeyError):
            continue
        if (now - txn_date).days <= window_days:
            recent_buyers.add(t.get("insider_name", ""))

    return len(recent_buyers) >= min_insiders


def detect_csuite_buy(
    transactions: list[dict],
    min_value: float = 100_000,
) -> bool:
    """Detect C-suite insider buying above threshold value."""
    for t in transactions:
        if t.get("transaction_type") != "P":
            continue
        title = t.get("title", "")
        if any(cs.lower() in title.lower() for cs in _CSUITE_TITLES):
            value = t.get("value", 0) or (t.get("shares", 0) * t.get("price", 0))
            if value >= min_value:
                return True
    return False


def detect_unusual_size(
    transactions: list[dict],
    multiplier: float = 10,
) -> bool:
    """Detect if any transaction is >multiplier x average for that insider."""
    # Group by insider
    insider_txns: dict[str, list[int]] = {}
    for t in transactions:
        name = t.get("insider_name", "unknown")
        shares = abs(t.get("shares", 0))
        insider_txns.setdefault(name, []).append(shares)

    for name, share_list in insider_txns.items():
        if len(share_list) < 2:
            continue
        avg = sum(share_list[:-1]) / len(share_list[:-1])
        if avg > 0 and share_list[-1] > avg * multiplier:
            return True

    return False


def compute_insider_score(transactions: list[dict]) -> float:
    """Compute insider signal score in [-1, +1].

    +1 = heavy buying, -1 = heavy selling. Excludes option exercises (code A/M).
    """
    if not transactions:
        return 0.0

    buy_value = 0.0
    sell_value = 0.0

    for t in transactions:
        txn_type = t.get("transaction_type", "")

        # Exclude option exercises from sell-side analysis
        if txn_type in _EXERCISE_CODES:
            continue

        value = abs(t.get("value", 0) or (t.get("shares", 0) * t.get("price", 0)))
        if txn_type == "P":
            buy_value += value
        elif txn_type == "S":
            sell_value += value

    total = buy_value + sell_value
    if total == 0:
        return 0.0

    # Score: (buys - sells) / total, in [-1, +1]
    score = (buy_value - sell_value) / total
    return max(-1.0, min(1.0, score))
