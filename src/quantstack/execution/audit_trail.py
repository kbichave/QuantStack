# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Best Execution Audit Trail — NBBO capture and price improvement tracking.

Every fill (immediate or child leg) gets an ``execution_audit`` row recording
the NBBO at fill time, the execution venue, the algo that was selected, and
the price improvement (or cost) relative to the NBBO midpoint.

Design choices:
  - **AuditRecorder.record() never raises.** Audit failure must never block
    a fill from being recorded. The entire body is wrapped in try/except.
  - **NBBOFetcher is injectable** for testing. Production uses Alpaca IEX
    (15-min delayed but free/unlimited). Falls back to (None, None) if
    the Alpaca SDK is unavailable or the quote request fails.
  - **Price improvement is signed:** positive = favorable (bought below
    midpoint or sold above midpoint), negative = adverse.

Usage:
    recorder = AuditRecorder()  # production: fetches live NBBO
    recorder.record(conn, order_id="abc", symbol="AAPL", side="buy",
                    fill_price=149.98, fill_venue="paper",
                    algo_selected="immediate")

    # In tests, inject a fake fetcher:
    recorder = AuditRecorder(nbbo_fetcher=FakeNBBOFetcher(bid=149.95, ask=150.05))
"""

from __future__ import annotations

import time
from typing import Protocol

from loguru import logger

from quantstack.db import PgConnection

# Optional Alpaca import — not a hard dependency.
try:
    from alpaca.data.live import StockDataStream  # noqa: F401
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.data.historical import StockHistoricalDataClient

    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False


# ---------------------------------------------------------------------------
# NBBO fetcher protocol + implementations
# ---------------------------------------------------------------------------


class NBBOFetcherProtocol(Protocol):
    """Structural interface for NBBO quote fetchers."""

    def fetch(self, symbol: str) -> tuple[float | None, float | None]: ...  # noqa: E704


class NBBOFetcher:
    """Fetches NBBO quotes from Alpaca IEX.

    Returns (bid, ask) or (None, None) if unavailable.
    Never raises -- returns None pair on any failure.
    """

    def __init__(self) -> None:
        self._client = None
        if _HAS_ALPACA:
            import os

            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            if api_key and secret_key:
                try:
                    self._client = StockHistoricalDataClient(api_key, secret_key)
                except Exception as exc:
                    logger.debug(f"[AUDIT] Alpaca client init failed: {exc}")

    def fetch(self, symbol: str) -> tuple[float | None, float | None]:
        """Return (bid, ask) or (None, None) if unavailable."""
        if self._client is None:
            return (None, None)
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._client.get_stock_latest_quote(request)
            quote = quotes.get(symbol)
            if quote is None:
                return (None, None)
            return (float(quote.bid_price), float(quote.ask_price))
        except Exception as exc:
            logger.debug(f"[AUDIT] NBBO fetch failed for {symbol}: {exc}")
            return (None, None)


# ---------------------------------------------------------------------------
# Audit recorder
# ---------------------------------------------------------------------------


def _compute_price_improvement_bps(
    side: str,
    fill_price: float,
    midpoint: float,
) -> float:
    """Compute price improvement in basis points relative to NBBO midpoint.

    Positive = favorable execution (bought below mid, sold above mid).
    Negative = adverse execution.
    """
    if midpoint <= 0:
        return 0.0
    if side == "buy":
        return (midpoint - fill_price) / midpoint * 10_000
    # sell
    return (fill_price - midpoint) / midpoint * 10_000


class AuditRecorder:
    """Records best-execution audit rows for every fill.

    Captures the NBBO at fill time, computes price improvement relative to
    the midpoint, and persists to the ``execution_audit`` table.

    The entire ``record()`` method is wrapped in try/except -- audit failures
    must never propagate to callers or block fill recording.
    """

    def __init__(self, nbbo_fetcher: NBBOFetcherProtocol | None = None) -> None:
        self._fetcher: NBBOFetcherProtocol = nbbo_fetcher or NBBOFetcher()

    def record(
        self,
        conn: PgConnection,
        order_id: str,
        symbol: str,
        side: str,
        fill_price: float,
        fill_venue: str | None,
        algo_selected: str,
        algo_rationale: str = "",
        fill_leg_id: int | None = None,
    ) -> None:
        """Capture NBBO, compute price improvement, persist audit row.

        Catches all exceptions -- audit failure must never block fills.
        """
        try:
            bid, ask = self._fetcher.fetch(symbol)

            midpoint: float | None = None
            price_improvement_bps: float | None = None

            if bid is not None and ask is not None:
                midpoint = (bid + ask) / 2.0
                price_improvement_bps = _compute_price_improvement_bps(
                    side=side,
                    fill_price=fill_price,
                    midpoint=midpoint,
                )

            timestamp_ns = time.time_ns()

            conn.execute(
                """
                INSERT INTO execution_audit (
                    order_id, fill_leg_id, nbbo_bid, nbbo_ask,
                    nbbo_midpoint, fill_price, fill_venue,
                    price_improvement_bps, algo_selected,
                    algo_rationale, timestamp_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    order_id,
                    fill_leg_id,
                    bid,
                    ask,
                    midpoint,
                    fill_price,
                    fill_venue,
                    price_improvement_bps,
                    algo_selected,
                    algo_rationale,
                    timestamp_ns,
                ],
            )

            if price_improvement_bps is not None:
                direction = "favorable" if price_improvement_bps >= 0 else "adverse"
                logger.debug(
                    f"[AUDIT] {symbol} {side} @ ${fill_price:.4f} "
                    f"mid=${midpoint:.4f} PI={price_improvement_bps:+.1f}bps ({direction})"
                )
            else:
                logger.debug(
                    f"[AUDIT] {symbol} {side} @ ${fill_price:.4f} "
                    f"(NBBO unavailable)"
                )

        except Exception as exc:
            # Audit failure must NEVER propagate.
            logger.warning(
                f"[AUDIT] Failed to record audit for order {order_id}: {exc}"
            )
