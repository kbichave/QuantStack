# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
L2 Microstructure signals for SignalEngine.

Computes institutional-grade microstructure signals when real-time Level 2
order book data is available via IBKR. All signals degrade gracefully to
None when IBKR is unavailable or L2 data cannot be fetched.

Signals implemented
-------------------
- OBI (Order Book Imbalance): (ΣBidQty_N - ΣAskQty_N) / Total_N
  Positive = bid-side pressure = bullish. Range [-1, 1].

- Micro-Price (Weighted Mid): Ask × (BidQty/Total) + Bid × (AskQty/Total)
  Accounts for queue priority — more accurate than (bid+ask)/2.

- Quoted Spread: (Ask - Bid) / Mid × 10_000 (in bps).
  Widening spread = low liquidity regime = reduce position sizing.

- Spread Percentile: rolling percentile of quoted spread over last 20 snapshots.
  High percentile = unusually wide spread right now.

- Kyle's Lambda (OHLCV approximation): OLS of |Δprice| ~ λ × |volume| over
  a rolling window. True Kyle's Lambda needs signed trades (tick data), but this
  OHLCV proxy captures the same price-impact relationship at daily resolution.
  Built without L2 — always available.

Failure modes
-------------
- IBKR not installed / not reachable → returns empty dict (all None)
- L2 feed not subscribed for the symbol → returns OBI=None, spread=None
- Kyle's Lambda requires OHLCV → always computable; returns NaN for early bars
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Kyle's Lambda (OHLCV-based proxy, always available)
# ---------------------------------------------------------------------------


def kyle_lambda_ohlcv(
    close: pd.Series,
    volume: pd.Series,
    period: int = 22,
) -> pd.Series:
    """
    OHLCV proxy for Kyle's Lambda (price impact coefficient).

    True Lambda = OLS of signed_price_change ~ λ × signed_volume.
    Approximation: rolling OLS of |Δclose| ~ λ × volume.

    High lambda = large price move per unit volume = low liquidity / high impact.
    Low lambda = deep book, price moves little per trade.

    Parameters
    ----------
    close, volume : pd.Series
    period : int
        Rolling window for OLS. Default 22 (1 month daily).

    Returns
    -------
    pd.Series — Kyle's Lambda estimate (units: $/share per share traded)
    """
    dP = close.diff().abs()
    V = volume.replace(0, np.nan)

    lambdas = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        dp_w = dP.iloc[i - period : i].values
        v_w = V.iloc[i - period : i].values
        mask = ~(np.isnan(dp_w) | np.isnan(v_w))
        if mask.sum() < 5:
            continue
        x = v_w[mask]
        y = dp_w[mask]
        # OLS: lambda = cov(x,y) / var(x)
        vx = x - x.mean()
        denom = (vx**2).sum()
        if denom > 0:
            lambdas[i] = (vx * (y - y.mean())).sum() / denom

    return pd.Series(lambdas, index=close.index, name="kyle_lambda")


def kyle_lambda_df(
    close: pd.Series,
    volume: pd.Series,
    period: int = 22,
) -> pd.DataFrame:
    """
    Compute Kyle's Lambda with derived regime signals.

    Returns
    -------
    pd.DataFrame with columns:
        kyle_lambda         – raw Lambda estimate
        kyle_lambda_zscore  – rolling z-score (window=period)
        high_impact         – 1 when lambda z-score > 1.5 (illiquid, high impact)
    """
    lam = kyle_lambda_ohlcv(close, volume, period)
    roll_mean = lam.rolling(period).mean()
    roll_std = lam.rolling(period).std()
    z = (lam - roll_mean) / roll_std.replace(0, np.nan)
    high_impact = (z > 1.5).astype(int)
    return pd.DataFrame(
        {"kyle_lambda": lam, "kyle_lambda_zscore": z, "high_impact": high_impact},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# IBKR L2 book signals
# ---------------------------------------------------------------------------


def _fetch_ibkr_book(symbol: str, n_levels: int = 5) -> dict | None:
    """
    Fetch L2 order book from IBKR via ib_insync.

    Returns a dict with keys:
        bids: list of (price, size) tuples, best first
        asks: list of (price, size) tuples, best first

    Returns None if IBKR is unavailable or L2 data fails.
    """
    try:
        from ib_insync import IB, Stock

        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        port = int(os.environ.get("IBKR_PORT", "4001"))
        client_id = int(
            os.environ.get("IBKR_L2_CLIENT_ID", "2")
        )  # separate client from main

        ib = IB()
        ib.connect(host, port, clientId=client_id, timeout=5)
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)

        ticker = ib.reqMktDepth(contract, numRows=n_levels)
        ib.sleep(0.5)  # wait for snapshot

        bids = [(d.price, d.size) for d in ticker.domBids[:n_levels]]
        asks = [(d.price, d.size) for d in ticker.domAsks[:n_levels]]

        ib.cancelMktDepth(contract)
        ib.disconnect()
        return {"bids": bids, "asks": asks}

    except Exception as exc:
        logger.debug(f"[l2] {symbol}: IBKR L2 fetch failed — {exc}")
        return None


def compute_book_signals(book: dict, n_levels: int = 5) -> dict[str, float | None]:
    """
    Compute OBI, micro-price, and spread from a raw order book snapshot.

    Parameters
    ----------
    book : dict with keys 'bids' and 'asks' (list of (price, size) tuples)
    n_levels : int
        Top N levels to use. Default 5.

    Returns
    -------
    dict with keys: obi, micro_price, quoted_spread_bps
    """
    result: dict[str, float | None] = {
        "obi": None,
        "micro_price": None,
        "quoted_spread_bps": None,
    }

    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        return result

    bid_sizes = np.array([b[1] for b in bids[:n_levels]], dtype=float)
    ask_sizes = np.array([a[1] for a in asks[:n_levels]], dtype=float)
    bid_prices = np.array([b[0] for b in bids[:n_levels]], dtype=float)
    ask_prices = np.array([a[0] for a in asks[:n_levels]], dtype=float)

    total_bid = bid_sizes.sum()
    total_ask = ask_sizes.sum()
    total = total_bid + total_ask

    if total > 0:
        result["obi"] = round((total_bid - total_ask) / total, 4)

    best_bid = bid_prices[0] if len(bid_prices) > 0 else None
    best_ask = ask_prices[0] if len(ask_prices) > 0 else None

    if best_bid and best_ask and total > 0:
        # Micro-price: weighted mid accounting for queue imbalance
        result["micro_price"] = round(
            best_ask * (total_bid / total) + best_bid * (total_ask / total), 4
        )
        mid = (best_bid + best_ask) / 2
        if mid > 0:
            result["quoted_spread_bps"] = round((best_ask - best_bid) / mid * 10_000, 2)

    return result


# ---------------------------------------------------------------------------
# SignalEngine integration
# ---------------------------------------------------------------------------


class SpreadHistory:
    """Rolling spread history for percentile computation (stateless per call)."""

    @staticmethod
    def percentile(spread_bps: float, history: list[float]) -> float | None:
        if not history or spread_bps is None:
            return None
        rank = sum(1 for h in history if h <= spread_bps)
        return round(rank / len(history), 4)


def collect_l2_microstructure(
    symbol: str,
    close: pd.Series | None = None,
    volume: pd.Series | None = None,
    n_levels: int = 5,
    kyle_period: int = 22,
) -> dict[str, Any]:
    """
    Collect L2 microstructure signals for SignalEngine.

    Always computes Kyle's Lambda from OHLCV (if provided).
    Attempts IBKR L2 book fetch; silently skips on any failure.

    Returns
    -------
    dict with prefixed keys: l2_obi, l2_micro_price, l2_spread_bps,
                             l2_kyle_lambda, l2_kyle_zscore, l2_high_impact
    """
    result: dict[str, Any] = {}

    # Kyle's Lambda — always available from OHLCV
    if close is not None and volume is not None and len(close) >= kyle_period:
        try:
            kl_df = kyle_lambda_df(close, volume, period=kyle_period)
            last = kl_df.iloc[-1]
            result["l2_kyle_lambda"] = (
                float(last["kyle_lambda"])
                if not np.isnan(last["kyle_lambda"])
                else None
            )
            result["l2_kyle_zscore"] = (
                float(last["kyle_lambda_zscore"])
                if not np.isnan(last["kyle_lambda_zscore"])
                else None
            )
            result["l2_high_impact"] = int(last["high_impact"])
        except Exception as exc:
            logger.debug(f"[l2] {symbol}: Kyle's Lambda failed — {exc}")

    # IBKR L2 book — gated on connectivity
    book = _fetch_ibkr_book(symbol, n_levels=n_levels)
    if book is not None:
        book_signals = compute_book_signals(book, n_levels=n_levels)
        result["l2_obi"] = book_signals.get("obi")
        result["l2_micro_price"] = book_signals.get("micro_price")
        result["l2_spread_bps"] = book_signals.get("quoted_spread_bps")
    else:
        result["l2_obi"] = None
        result["l2_micro_price"] = None
        result["l2_spread_bps"] = None

    return result
