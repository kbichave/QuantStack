# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Commodity signals collector — gold/silver/copper ratios, sector rotation,
USD strength, and commodity regime classification.

Reads from the ``macro_indicators`` table (GOLD, SILVER, COPPER, EURUSD,
USDJPY) and computes cross-commodity signals useful for regime detection
and sector allocation.

This is a *global* collector — the symbol argument is accepted for API
consistency but the output applies market-wide.

Design invariants:
- Returns {} when data is missing or stale (>2 days).
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

_STALENESS_DAYS = 2
_LOOKBACK_DAYS = 60


async def collect_commodity_signals(
    symbol: str, store: DataStore
) -> dict[str, Any]:
    """Compute commodity signals from macro indicator data."""
    if not check_freshness(symbol, "macro_indicators", max_days=45):
        return {}
    try:
        return await asyncio.to_thread(
            _collect_commodity_sync, symbol, store
        )
    except Exception as exc:
        logger.warning(
            f"[commodity] {symbol}: {type(exc).__name__} — returning empty"
        )
        return {}


def _collect_commodity_sync(
    symbol: str, store: DataStore
) -> dict[str, Any]:
    """Synchronous commodity signal computation."""
    if not hasattr(store, "load_macro_indicator"):
        logger.debug("[commodity] store missing load_macro_indicator")
        return {}

    start = (date.today() - timedelta(days=_LOOKBACK_DAYS)).isoformat()

    # Load commodity price series
    gold_df = store.load_macro_indicator("GOLD", start_date=start)
    silver_df = store.load_macro_indicator("SILVER", start_date=start)
    copper_df = store.load_macro_indicator("COPPER", start_date=start)

    if gold_df.empty:
        return {}

    # Staleness check — if GOLD most recent > 2 days ago, data is stale
    gold_df["date"] = pd.to_datetime(gold_df["date"])
    latest_gold_date = gold_df["date"].max()
    days_stale = (pd.Timestamp(date.today()) - latest_gold_date).days
    if days_stale > _STALENESS_DAYS:
        logger.debug(
            f"[commodity] Gold data stale ({days_stale} days) — returning empty"
        )
        return {}

    result: dict[str, Any] = {}

    gold_price = float(gold_df["value"].iloc[-1])
    result["gold_price"] = round(gold_price, 2)

    # --- Gold/Silver ratio ---
    if not silver_df.empty:
        silver_price = float(silver_df["value"].iloc[-1])
        if silver_price > 0:
            result["gold_silver_ratio"] = round(gold_price / silver_price, 4)
        else:
            result["gold_silver_ratio"] = None
    else:
        result["gold_silver_ratio"] = None

    # --- Copper/Gold ratio ---
    if not copper_df.empty:
        copper_price = float(copper_df["value"].iloc[-1])
        if gold_price > 0:
            result["copper_gold_ratio"] = round(copper_price / gold_price, 6)
        else:
            result["copper_gold_ratio"] = None
    else:
        result["copper_gold_ratio"] = None

    # --- Momentum (5d and 20d returns) ---
    gold_5d_ret = _compute_return(gold_df, 5)
    gold_20d_ret = _compute_return(gold_df, 20)
    result["gold_5d_return"] = gold_5d_ret
    result["gold_20d_return"] = gold_20d_ret

    copper_5d_ret = _compute_return(copper_df, 5) if not copper_df.empty else None
    result["copper_5d_return"] = copper_5d_ret

    # --- Sector rotation signal ---
    # copper up + gold down → favor_cyclicals
    # gold up + copper down → favor_defensives
    # both up → inflationary
    # both down → neutral
    result["sector_rotation_signal"] = _classify_rotation(
        gold_5d_ret, copper_5d_ret
    )

    # --- USD strength proxy from forex ---
    eurusd_df = store.load_macro_indicator("EURUSD", start_date=start)
    usdjpy_df = store.load_macro_indicator("USDJPY", start_date=start)
    result["usd_strength_proxy"] = _compute_usd_strength(eurusd_df, usdjpy_df)

    # --- Risk-off composite score ---
    # High gold + falling copper + strong USD = risk-off
    risk_off_score = _compute_risk_off_score(
        gold_5d_ret, copper_5d_ret, result.get("usd_strength_proxy")
    )
    result["risk_off_score"] = risk_off_score

    # --- Commodity regime ---
    result["commodity_regime"] = _classify_regime(risk_off_score)

    return result


def _compute_return(df: pd.DataFrame, periods: int) -> float | None:
    """Compute percentage return over *periods* rows."""
    if df.empty or len(df) < periods + 1:
        return None
    try:
        current = float(df["value"].iloc[-1])
        prior = float(df["value"].iloc[-(periods + 1)])
        if prior == 0:
            return None
        return round((current - prior) / prior * 100, 4)
    except (IndexError, ValueError, TypeError):
        return None


def _classify_rotation(
    gold_ret: float | None, copper_ret: float | None
) -> str:
    """Classify sector rotation signal from gold and copper momentum."""
    if gold_ret is None or copper_ret is None:
        return "unknown"

    gold_up = gold_ret > 0
    copper_up = copper_ret > 0

    if copper_up and not gold_up:
        return "favor_cyclicals"
    if gold_up and not copper_up:
        return "favor_defensives"
    if gold_up and copper_up:
        return "inflationary"
    # both down
    return "neutral"


def _compute_usd_strength(
    eurusd_df: pd.DataFrame, usdjpy_df: pd.DataFrame
) -> float | None:
    """Compute USD strength proxy from EUR/USD and USD/JPY.

    USD strength = (-EURUSD_return + USDJPY_return) / 2
    Positive = USD strengthening.
    """
    eur_ret = _compute_return(eurusd_df, 5) if not eurusd_df.empty else None
    jpy_ret = _compute_return(usdjpy_df, 5) if not usdjpy_df.empty else None

    if eur_ret is None and jpy_ret is None:
        return None
    if eur_ret is None:
        return round(jpy_ret, 4)  # type: ignore[arg-type]
    if jpy_ret is None:
        return round(-eur_ret, 4)

    return round((-eur_ret + jpy_ret) / 2, 4)


def _compute_risk_off_score(
    gold_ret: float | None,
    copper_ret: float | None,
    usd_strength: float | None,
) -> float | None:
    """Composite risk-off score: higher = more risk-off.

    Components (each 0-1):
    - Gold rising → risk-off
    - Copper falling → risk-off
    - USD strengthening → risk-off
    """
    components: list[float] = []

    if gold_ret is not None:
        # Gold up = risk-off; clamp to [-5, 5], normalize to [0, 1]
        gold_component = max(-5, min(5, gold_ret)) / 10 + 0.5
        components.append(gold_component)

    if copper_ret is not None:
        # Copper down = risk-off; invert and normalize
        copper_component = max(-5, min(5, -copper_ret)) / 10 + 0.5
        components.append(copper_component)

    if usd_strength is not None:
        # USD up = risk-off; normalize
        usd_component = max(-5, min(5, usd_strength)) / 10 + 0.5
        components.append(usd_component)

    if not components:
        return None

    return round(sum(components) / len(components), 4)


def _classify_regime(risk_off_score: float | None) -> str:
    """Classify commodity regime from risk-off score."""
    if risk_off_score is None:
        return "unknown"
    if risk_off_score > 0.65:
        return "risk_off"
    if risk_off_score < 0.35:
        return "risk_on"
    return "neutral"
