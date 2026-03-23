"""
OrderBookReconstructor — maintains a live limit order book from L2Update events.

Relationship to OrderBook
--------------------------
``OrderBook`` (order_book.py) is a simulation-oriented LOB used by the
backtesting matching engine.  It tracks individual ``Order`` objects with
order IDs and price-time priority queues.

``OrderBookReconstructor`` is real-time oriented: it maintains a *price-level*
view (not order-by-order) because that is the granularity delivered by market
data providers (Polygon Level 2, IBKR reqMktDepth, etc.).

Design
------
Internal state: two dicts, ``bids`` and ``asks``, mapping ``price → size``.
On each L2Update:
  - "add" / "modify": ``book[side][price] = size``
  - "delete" or ``size == 0``: ``del book[side][price]``

Snapshot handling:
    When ``L2Update.is_snapshot is True``, the first message of the batch
    clears the book before applying.  Subsequent snapshot messages in the
    same batch (``is_snapshot=True``) are applied without clearing.  The
    transition back to incremental updates is automatic.

``BookSnapshot`` output
-----------------------
After each ``L2Update`` the reconstructor produces a ``BookSnapshot``
that is emitted to registered callbacks.  The snapshot contains:

  best_bid / best_ask       — Top-of-book prices
  spread / spread_bps       — Bid-ask spread
  mid                       — Midpoint price
  imbalance                 — (bid_vol − ask_vol) / (bid_vol + ask_vol) for top N levels
  bid_depth / ask_depth     — [(price, size)] sorted best → worst, up to depth_levels
  timestamp_ns              — Nanoseconds from the triggering L2Update
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from quantstack.data.streaming.tick_models import L2Update

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


@dataclass
class BookSnapshot:
    """Point-in-time view of the limit order book after an L2Update.

    ``bid_depth`` is sorted descending by price (best bid first).
    ``ask_depth`` is sorted ascending by price (best ask first).
    """

    symbol: str
    timestamp_ns: int

    best_bid: float | None
    best_ask: float | None
    spread: float | None  # ask − bid  (None if one side is empty)
    spread_bps: float | None  # spread / mid × 10 000
    mid: float | None  # (bid + ask) / 2

    # Order imbalance at top-N levels: (bid_vol − ask_vol) / total_vol ∈ [−1, 1]
    imbalance: float

    bid_depth: list[tuple]  # [(price, size), ...] best → worst
    ask_depth: list[tuple]  # [(price, size), ...] best → worst


BookSnapshotCallback = Callable[[BookSnapshot], Awaitable[None]]


# ---------------------------------------------------------------------------
# Reconstructor
# ---------------------------------------------------------------------------


class OrderBookReconstructor:
    """Builds a price-level order book from L2Update events.

    Args:
        depth_levels: Number of levels to include in ``BookSnapshot.bid_depth``
                      and ``BookSnapshot.ask_depth``.
        imbalance_levels: Number of levels to use for imbalance computation.
    """

    def __init__(
        self,
        depth_levels: int = 10,
        imbalance_levels: int = 5,
    ) -> None:
        self._depth = depth_levels
        self._imb_lvl = imbalance_levels

        # symbol → {price: size}
        self._bids: dict[str, dict[float, float]] = {}
        self._asks: dict[str, dict[float, float]] = {}

        # Whether the next snapshot batch should clear the book first
        self._pending_clear: dict[str, bool] = {}

        self._callbacks: list[BookSnapshotCallback] = []

    # ── Callback registration ─────────────────────────────────────────────────

    def add_callback(self, callback: BookSnapshotCallback) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: BookSnapshotCallback) -> None:
        self._callbacks = [c for c in self._callbacks if c is not callback]

    # ── L2Callback interface ──────────────────────────────────────────────────

    async def on_l2_update(self, update: L2Update) -> None:
        """Process one L2Update event and emit a BookSnapshot."""
        sym = update.symbol

        # Initialise book for new symbol
        if sym not in self._bids:
            self._bids[sym] = {}
            self._asks[sym] = {}
            self._pending_clear[sym] = True

        # Snapshot: clear the book on the first message of a snapshot batch
        if update.is_snapshot and self._pending_clear.get(sym, False):
            self._bids[sym].clear()
            self._asks[sym].clear()
            self._pending_clear[sym] = False

        # Transition back to incremental mode after snapshot
        if not update.is_snapshot:
            self._pending_clear[sym] = False

        # Apply the update
        book = self._bids[sym] if update.side == "bid" else self._asks[sym]
        if update.size <= 0.0 or update.action == "delete":
            book.pop(update.price, None)
        else:
            book[update.price] = update.size

        # Build and emit snapshot
        snapshot = self._build_snapshot(sym, update.timestamp_ns)
        if snapshot and self._callbacks:
            await asyncio.gather(
                *(cb(snapshot) for cb in self._callbacks),
                return_exceptions=True,
            )

    # ── Snapshot builder ──────────────────────────────────────────────────────

    def _build_snapshot(self, symbol: str, timestamp_ns: int) -> BookSnapshot | None:
        bids = self._bids.get(symbol, {})
        asks = self._asks.get(symbol, {})

        # Sort: bids descending, asks ascending
        bid_sorted = sorted(bids.items(), key=lambda kv: kv[0], reverse=True)
        ask_sorted = sorted(asks.items(), key=lambda kv: kv[0])

        best_bid = bid_sorted[0][0] if bid_sorted else None
        best_ask = ask_sorted[0][0] if ask_sorted else None

        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            mid = (best_bid + best_ask) / 2.0
            spread_bps = spread / mid * 10_000 if mid != 0.0 else None
        else:
            spread = mid = spread_bps = None

        # Imbalance at top-N levels
        bid_vol = sum(s for _, s in bid_sorted[: self._imb_lvl])
        ask_vol = sum(s for _, s in ask_sorted[: self._imb_lvl])
        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

        return BookSnapshot(
            symbol=symbol,
            timestamp_ns=timestamp_ns,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_bps=spread_bps,
            mid=mid,
            imbalance=imbalance,
            bid_depth=bid_sorted[: self._depth],
            ask_depth=ask_sorted[: self._depth],
        )

    # ── Read API ──────────────────────────────────────────────────────────────

    def get_snapshot(self, symbol: str) -> BookSnapshot | None:
        """Return a current snapshot without waiting for a new update."""
        if symbol not in self._bids:
            return None
        return self._build_snapshot(symbol, time.time_ns())

    def best_bid(self, symbol: str) -> float | None:
        bids = self._bids.get(symbol, {})
        return max(bids.keys()) if bids else None

    def best_ask(self, symbol: str) -> float | None:
        asks = self._asks.get(symbol, {})
        return min(asks.keys()) if asks else None

    def mid(self, symbol: str) -> float | None:
        bb, ba = self.best_bid(symbol), self.best_ask(symbol)
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None
