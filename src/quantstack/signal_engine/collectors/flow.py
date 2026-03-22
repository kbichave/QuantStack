# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Flow collector — insider trades and institutional ownership signals.

Fetches insider trading activity and institutional ownership changes from
FinancialDatasets.ai.  Returns {} if no API key is configured — flow
signals are supplementary, never blocking.
"""

import asyncio
import os
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantstack.data.adapters.financial_datasets_client import FinancialDatasetsClient
from quantstack.data.storage import DataStore


_TIMEOUT_SECONDS = 10.0
_INSIDER_LOOKBACK_DAYS = 90
_INSIDER_LIMIT = 50
_INSTITUTIONAL_LIMIT = 10  # most recent filings


async def collect_flow(symbol: str, store: DataStore) -> dict[str, Any]:
    """Collect insider + institutional ownership flow signals. Returns {} on failure."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_flow_sync, symbol, store),
            timeout=_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.debug(f"[flow] {symbol}: {type(exc).__name__} — returning empty")
        return {}


def _collect_flow_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
    if not api_key:
        logger.debug("[flow] FINANCIAL_DATASETS_API_KEY not set — skipping")
        return {}

    with FinancialDatasetsClient(api_key=api_key) as client:
        insider_resp = client.get_insider_trades(symbol, limit=_INSIDER_LIMIT)
        institutional_resp = client.get_institutional_ownership_by_ticker(
            symbol, limit=_INSTITUTIONAL_LIMIT
        )

    result: dict[str, Any] = {}

    # --- Insider trades ---
    insider_signals = _process_insider_trades(insider_resp)
    result.update(insider_signals)

    # --- Institutional ownership ---
    institutional_signals = _process_institutional_ownership(institutional_resp)
    result.update(institutional_signals)

    # --- Composite flow signal ---
    result["flow_signal"] = _compute_composite_signal(
        result.get("insider_direction", "neutral"),
        result.get("institutional_direction", "stable"),
    )

    return result


def _process_insider_trades(resp: dict[str, Any] | None) -> dict[str, Any]:
    """Extract net insider buying/selling over the lookback window."""
    if resp is None:
        return {
            "insider_net_90d": 0.0,
            "insider_direction": "neutral",
            "insider_trade_count": 0,
        }

    trades = resp.get("insider_trades") or resp.get("trades") or []
    if not isinstance(trades, list):
        return {
            "insider_net_90d": 0.0,
            "insider_direction": "neutral",
            "insider_trade_count": 0,
        }

    cutoff = date.today() - timedelta(days=_INSIDER_LOOKBACK_DAYS)
    net_shares = 0.0
    count = 0

    for trade in trades:
        if not isinstance(trade, dict):
            continue

        # Filter by date if available
        trade_date_raw = trade.get("filing_date") or trade.get("date")
        if trade_date_raw:
            try:
                trade_date = date.fromisoformat(str(trade_date_raw)[:10])
                if trade_date < cutoff:
                    continue
            except ValueError:
                pass  # include if date unparseable — better than silently dropping

        shares = _safe_float(trade.get("shares") or trade.get("quantity") or 0)
        if shares is None:
            continue

        # Transaction type: "buy" / "purchase" add shares; "sell" / "sale" subtract
        tx_type = str(
            trade.get("transaction_type")
            or trade.get("type")
            or trade.get("acquisition_or_disposal")
            or ""
        ).lower()

        if any(kw in tx_type for kw in ("buy", "purchase", "acquisition", "a")):
            net_shares += shares
            count += 1
        elif any(kw in tx_type for kw in ("sell", "sale", "disposal", "d")):
            net_shares -= shares
            count += 1

    if count == 0:
        direction = "neutral"
    elif net_shares > 0:
        direction = "buying"
    elif net_shares < 0:
        direction = "selling"
    else:
        direction = "neutral"

    return {
        "insider_net_90d": round(net_shares, 0),
        "insider_direction": direction,
        "insider_trade_count": count,
    }


def _process_institutional_ownership(resp: dict[str, Any] | None) -> dict[str, Any]:
    """Compute quarter-over-quarter institutional ownership change."""
    if resp is None:
        return {
            "institutional_change_pct": None,
            "institutional_direction": "unknown",
        }

    ownership = (
        resp.get("institutional_ownership")
        or resp.get("ownership")
        or resp.get("holders")
        or []
    )
    if not isinstance(ownership, list) or len(ownership) < 2:
        # Need at least 2 filings to compute change
        if isinstance(ownership, list) and len(ownership) == 1:
            return {
                "institutional_change_pct": 0.0,
                "institutional_direction": "stable",
            }
        return {
            "institutional_change_pct": None,
            "institutional_direction": "unknown",
        }

    # Filings are returned most-recent-first (assumption from API ordering).
    # Sum total shares held across all institutions per filing period.
    latest_total = _sum_shares(ownership[0])
    prior_total = _sum_shares(ownership[1])

    if prior_total is None or prior_total <= 0 or latest_total is None:
        return {
            "institutional_change_pct": None,
            "institutional_direction": "unknown",
        }

    change_pct = round((latest_total - prior_total) / prior_total * 100, 2)

    if change_pct > 1.0:
        direction = "accumulating"
    elif change_pct < -1.0:
        direction = "distributing"
    else:
        direction = "stable"

    return {
        "institutional_change_pct": change_pct,
        "institutional_direction": direction,
    }


def _sum_shares(filing: Any) -> float | None:
    """Sum shares from a single institutional ownership filing entry."""
    if isinstance(filing, dict):
        # Single filing may have total shares directly
        shares = _safe_float(
            filing.get("shares")
            or filing.get("shares_held")
            or filing.get("total_shares")
        )
        if shares is not None:
            return shares
        # Or it may be a list of holders
        holders = filing.get("holders") or filing.get("positions") or []
        if isinstance(holders, list):
            total = 0.0
            for h in holders:
                s = _safe_float(h.get("shares") or h.get("shares_held") or 0)
                if s is not None:
                    total += s
            return total if total > 0 else None
    return None


def _compute_composite_signal(
    insider_direction: str, institutional_direction: str
) -> float:
    """Combine insider and institutional signals into [-1, 1] composite.

    Insider weight: 0.4  (smaller sample, noisier, but high-conviction)
    Institutional weight: 0.6  (larger, more systematic)
    """
    insider_score = {"buying": 1.0, "selling": -1.0, "neutral": 0.0}.get(
        insider_direction, 0.0
    )
    inst_score = {
        "accumulating": 1.0,
        "distributing": -1.0,
        "stable": 0.0,
        "unknown": 0.0,
    }.get(institutional_direction, 0.0)

    composite = round(0.4 * insider_score + 0.6 * inst_score, 2)
    return max(-1.0, min(1.0, composite))


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return None
