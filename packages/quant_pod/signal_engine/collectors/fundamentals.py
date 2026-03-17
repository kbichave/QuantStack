# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Fundamentals collector — replaces fundamentals_ic.

Reads only from the local QuantCore fundamentals store — no network call in
the live trading path.  Returns {} (not a failure) when data isn't loaded yet.
Fundamentals are refreshed separately (e.g., nightly) via data fetchers.
"""

import asyncio
from typing import Any

from loguru import logger

from quantcore.data.storage import DataStore


async def collect_fundamentals(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Load stored fundamental metrics for *symbol*.

    Returns a dict with keys (subset of what's available):
        pe_ratio        : float | None
        eps_ttm         : float | None
        revenue_growth  : float | None
        gross_margin    : float | None
        debt_to_equity  : float | None
        beta            : float | None
        market_cap      : float | None
        fundamentals_age_days : int | None — days since last fundamental update
    """
    try:
        return await asyncio.to_thread(_collect_fundamentals_sync, symbol, store)
    except Exception as exc:
        logger.debug(f"[fundamentals] {symbol}: {exc} — skipping (non-critical)")
        return {}


def _collect_fundamentals_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    """Load fundamentals from local store. Returns {} if not yet fetched."""
    try:
        # DataStore exposes get_fundamentals if FundamentalsSchemaMixin is loaded.
        # Returns None if no fundamentals row exists for this symbol.
        row = store.get_fundamentals(symbol) if hasattr(store, "get_fundamentals") else None
    except Exception as exc:
        logger.debug(f"[fundamentals] {symbol}: store.get_fundamentals raised: {exc}")
        return {}

    if row is None:
        return {}

    def _safe_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
            return None if f != f else f  # NaN guard
        except (TypeError, ValueError):
            return None

    return {
        "pe_ratio":           _safe_float(row.get("pe_ratio")),
        "eps_ttm":            _safe_float(row.get("eps_ttm")),
        "revenue_growth":     _safe_float(row.get("revenue_growth")),
        "gross_margin":       _safe_float(row.get("gross_margin")),
        "debt_to_equity":     _safe_float(row.get("debt_to_equity")),
        "beta":               _safe_float(row.get("beta")),
        "market_cap":         _safe_float(row.get("market_cap")),
        "fundamentals_age_days": row.get("age_days"),
    }
