# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Sector collector — relative strength, breadth, and rotation signals.

Loads OHLCV from the local DataStore for sector ETFs and SPY.  No network
call — all data must be pre-loaded via the data pipeline.  Returns {} if
SPY data is missing (required baseline).
"""

import asyncio
from typing import Any

from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore


_MIN_BARS = 25  # need at least 20 + a few for safety


# ── Symbol-to-sector mapping ────────────────────────────────────────────────
# Maps ~50 of the most liquid US equities to their sector ETF proxy.
# Symbols not in this map will use SPY as a fallback (market return).

SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI")

SYMBOL_TO_SECTOR_ETF: dict[str, str] = {
    # Technology — XLK
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "GOOG": "XLK",
    "GOOGL": "XLK",
    "META": "XLK",
    "AVGO": "XLK",
    "ADBE": "XLK",
    "CRM": "XLK",
    "AMD": "XLK",
    "INTC": "XLK",
    "CSCO": "XLK",
    "ORCL": "XLK",
    "QCOM": "XLK",
    "TXN": "XLK",
    "MU": "XLK",
    "AMAT": "XLK",
    "NOW": "XLK",
    "PLTR": "XLK",
    # Financials — XLF
    "JPM": "XLF",
    "BAC": "XLF",
    "WFC": "XLF",
    "GS": "XLF",
    "MS": "XLF",
    "C": "XLF",
    "BLK": "XLF",
    "SCHW": "XLF",
    "AXP": "XLF",
    "V": "XLF",
    "MA": "XLF",
    # Energy — XLE
    "XOM": "XLE",
    "CVX": "XLE",
    "COP": "XLE",
    "SLB": "XLE",
    "EOG": "XLE",
    "OXY": "XLE",
    # Healthcare — XLV
    "UNH": "XLV",
    "JNJ": "XLV",
    "LLY": "XLV",
    "PFE": "XLV",
    "ABBV": "XLV",
    "MRK": "XLV",
    "TMO": "XLV",
    "ABT": "XLV",
    # Consumer Discretionary — XLY
    "AMZN": "XLY",
    "TSLA": "XLY",
    "HD": "XLY",
    "MCD": "XLY",
    "NKE": "XLY",
    "LOW": "XLY",
    "SBUX": "XLY",
    # Consumer Staples — XLP
    "PG": "XLP",
    "KO": "XLP",
    "PEP": "XLP",
    "COST": "XLP",
    "WMT": "XLP",
    # Industrials — XLI
    "CAT": "XLI",
    "BA": "XLI",
    "HON": "XLI",
    "UPS": "XLI",
    "RTX": "XLI",
    "DE": "XLI",
    "GE": "XLI",
}


async def collect_sector(symbol: str, store: DataStore) -> dict[str, Any]:
    """Compute sector relative strength and rotation signals. Returns {} on failure."""
    try:
        return await asyncio.to_thread(_collect_sector_sync, symbol, store)
    except Exception as exc:
        logger.warning(f"[sector] {symbol}: {exc} — returning empty")
        return {}


def _collect_sector_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    # SPY is the required baseline — without it, nothing is meaningful.
    spy_df = store.load_ohlcv("SPY", Timeframe.D1)
    if spy_df is None or len(spy_df) < _MIN_BARS:
        logger.debug("[sector] SPY data missing or insufficient — skipping")
        return {}

    spy_close = spy_df["close"].values
    spy_ret_5d = _return_nday(spy_close, 5)
    spy_ret_20d = _return_nday(spy_close, 20)

    if spy_ret_5d is None:
        return {}

    # Determine the symbol's sector ETF
    sector_etf = SYMBOL_TO_SECTOR_ETF.get(symbol)
    result: dict[str, Any] = {
        "symbol_sector": sector_etf if sector_etf else "unknown",
    }

    # Sector relative strength for the symbol's own sector
    if sector_etf:
        etf_df = store.load_ohlcv(sector_etf, Timeframe.D1)
        if etf_df is not None and len(etf_df) >= _MIN_BARS:
            etf_close = etf_df["close"].values
            etf_ret_5d = _return_nday(etf_close, 5)
            etf_ret_20d = _return_nday(etf_close, 20)
            result["sector_rs_5d"] = _round_or_none(_sub(etf_ret_5d, spy_ret_5d))
            result["sector_rs_20d"] = _round_or_none(_sub(etf_ret_20d, spy_ret_20d))
            result["sector_trend"] = _classify_sector_trend(
                result["sector_rs_5d"], result["sector_rs_20d"]
            )
        else:
            result["sector_rs_5d"] = None
            result["sector_rs_20d"] = None
            result["sector_trend"] = "unknown"
    else:
        result["sector_rs_5d"] = None
        result["sector_rs_20d"] = None
        result["sector_trend"] = "unknown"

    # Breadth: how many sector ETFs have positive 5-day returns
    positive_count = 0
    sector_returns_5d: dict[str, float | None] = {}
    for etf in SECTOR_ETFS:
        etf_df = store.load_ohlcv(etf, Timeframe.D1)
        if etf_df is not None and len(etf_df) >= _MIN_BARS:
            ret = _return_nday(etf_df["close"].values, 5)
            sector_returns_5d[etf] = ret
            if ret is not None and ret > 0:
                positive_count += 1
        else:
            sector_returns_5d[etf] = None

    result["breadth_positive_sectors"] = positive_count

    # Rotation signal
    result["rotation_signal"] = _classify_rotation(sector_returns_5d, spy_ret_5d)

    return result


def _return_nday(close: Any, n: int) -> float | None:
    """Compute n-day simple return from a close price array."""
    if len(close) < n + 1:
        return None
    prior = _safe_float(close[-(n + 1)])
    latest = _safe_float(close[-1])
    if prior is None or latest is None or prior <= 0:
        return None
    return (latest - prior) / prior


def _classify_sector_trend(rs_5d: float | None, rs_20d: float | None) -> str:
    """Classify sector as leading, lagging, or inline vs SPY."""
    if rs_5d is None or rs_20d is None:
        return "unknown"
    # Leading if outperforming on both windows
    if rs_5d > 0.005 and rs_20d > 0.005:
        return "leading"
    if rs_5d < -0.005 and rs_20d < -0.005:
        return "lagging"
    return "inline"


def _classify_rotation(
    sector_returns: dict[str, float | None], spy_ret_5d: float | None
) -> str:
    """Classify market rotation pattern from sector ETF returns."""
    growth_etfs = ("XLK", "XLY")  # growth proxies
    value_etfs = ("XLF", "XLE", "XLI")  # value proxies
    defensive_etfs = ("XLV", "XLP")  # defensive proxies

    growth_avg = _avg_returns(sector_returns, growth_etfs)
    value_avg = _avg_returns(sector_returns, value_etfs)
    defensive_avg = _avg_returns(sector_returns, defensive_etfs)

    if growth_avg is None or value_avg is None or defensive_avg is None:
        return "mixed"

    available = [r for r in sector_returns.values() if r is not None]
    positive = sum(1 for r in available if r > 0)

    # Broad rally: most sectors positive
    if len(available) >= 5 and positive >= 6:
        return "broad_rally"

    # Narrow rally: few sectors carrying the market
    if len(available) >= 5 and positive <= 2:
        return "narrow_rally"

    # Defensive shift: defensive outperforming both growth and value
    if defensive_avg > growth_avg and defensive_avg > value_avg and defensive_avg > 0:
        return "defensive_shift"

    # Growth vs value rotation
    spread = growth_avg - value_avg
    if spread > 0.01:
        return "value_to_growth"
    if spread < -0.01:
        return "growth_to_value"

    return "mixed"


def _avg_returns(
    sector_returns: dict[str, float | None], etfs: tuple[str, ...]
) -> float | None:
    """Average returns for a subset of ETFs. None if no data."""
    vals = [sector_returns.get(e) for e in etfs if sector_returns.get(e) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _round_or_none(v: float | None, decimals: int = 6) -> float | None:
    return round(v, decimals) if v is not None else None


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return None
