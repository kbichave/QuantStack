"""
FillTracker — live position map and daily P&L ledger.

Responsibilities
----------------
1. Maintain an in-memory position map keyed by symbol: how many shares are
   held, at what average cost, and what the current mark-to-market value is.

2. Track daily realised P&L per symbol and in aggregate.

3. Accept fill notifications from the execution loop so the risk gate can
   read up-to-date exposure without hitting the broker REST API on every
   order check.

4. Support current-price injection so the paper-trade / live engine can push
   fresh quotes without the tracker needing to own a data connection.

Thread safety
-------------
All mutating methods acquire ``_lock`` (threading.RLock).  The execution
loop calls update_fill() from a single thread but the risk gate (and any
monitoring code) may read from a different thread.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class LivePosition:
    """Current state of one symbol held in the portfolio."""

    symbol: str
    quantity: float          # positive = long, negative = short
    avg_cost: float          # volume-weighted average entry price
    current_price: float     # last known mark-to-market price
    realised_pnl: float = 0.0   # cumulative closed P&L (today)
    open_pnl: float = 0.0       # mark-to-market open P&L

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def total_pnl(self) -> float:
        return self.realised_pnl + self.open_pnl

    def refresh_open_pnl(self) -> None:
        self.open_pnl = (self.current_price - self.avg_cost) * self.quantity


@dataclass
class FillEvent:
    """A single fill notification from the broker."""

    order_id: str
    symbol: str
    side: str           # "buy" | "sell"
    filled_qty: float
    avg_fill_price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FillTracker:
    """Maintains live positions and daily P&L from fill notifications.

    Args:
        starting_cash: Starting cash balance (informational; not enforced here).
    """

    def __init__(self, starting_cash: float = 0.0) -> None:
        self._lock = threading.RLock()
        self._positions: Dict[str, LivePosition] = {}
        self._fills: List[FillEvent] = []
        self._starting_cash = starting_cash
        self._day_start = date.today()

    # ── Fill ingestion ────────────────────────────────────────────────────────

    def update_fill(self, fill: FillEvent) -> None:
        """Record a fill and update the position map.

        Handles both opening and partially/fully closing positions.
        Uses FIFO-style average cost updating:
          - Buy adds to position at weighted avg cost.
          - Sell reduces position and realises P&L based on current avg cost.
        """
        with self._lock:
            self._fills.append(fill)
            sym = fill.symbol
            qty = fill.filled_qty if fill.side.lower() == "buy" else -fill.filled_qty

            if sym not in self._positions:
                self._positions[sym] = LivePosition(
                    symbol        = sym,
                    quantity      = 0.0,
                    avg_cost      = 0.0,
                    current_price = fill.avg_fill_price,
                )

            pos = self._positions[sym]
            prev_qty = pos.quantity

            if prev_qty == 0:
                # Opening a new position
                pos.quantity = qty
                pos.avg_cost = fill.avg_fill_price
            elif (prev_qty > 0 and qty > 0) or (prev_qty < 0 and qty < 0):
                # Adding to an existing position — update weighted avg cost
                total_cost = prev_qty * pos.avg_cost + qty * fill.avg_fill_price
                pos.quantity += qty
                pos.avg_cost = total_cost / pos.quantity if pos.quantity != 0 else 0.0
            else:
                # Reducing or flipping position
                close_qty = min(abs(qty), abs(prev_qty))
                realised = close_qty * (fill.avg_fill_price - pos.avg_cost) * (1 if prev_qty > 0 else -1)
                pos.realised_pnl += realised
                pos.quantity += qty
                if abs(pos.quantity) < 1e-9:
                    pos.quantity  = 0.0
                    pos.avg_cost  = 0.0
                elif (prev_qty > 0) != (pos.quantity > 0):
                    # Flipped side: residual quantity is at fill price
                    pos.avg_cost = fill.avg_fill_price

            pos.current_price = fill.avg_fill_price
            pos.refresh_open_pnl()

            logger.debug(
                f"[FillTracker] {fill.side.upper()} {fill.filled_qty} {sym} "
                f"@ {fill.avg_fill_price:.4f} | pos={pos.quantity:.2f} "
                f"avg={pos.avg_cost:.4f} rpnl={pos.realised_pnl:.2f}"
            )

    # ── Price updates ─────────────────────────────────────────────────────────

    def update_price(self, symbol: str, price: float) -> None:
        """Push a fresh market price and refresh open P&L for a symbol."""
        with self._lock:
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.current_price = price
                pos.refresh_open_pnl()

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Bulk price update from a quote snapshot."""
        with self._lock:
            for sym, px in prices.items():
                if sym in self._positions:
                    pos = self._positions[sym]
                    pos.current_price = px
                    pos.refresh_open_pnl()

    # ── Queries (read-only — safe to call without lock from risk gate) ─────────

    def get_position(self, symbol: str) -> Optional[LivePosition]:
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, LivePosition]:
        with self._lock:
            return dict(self._positions)

    def get_open_positions(self) -> Dict[str, LivePosition]:
        """Return only positions with non-zero quantity."""
        with self._lock:
            return {s: p for s, p in self._positions.items() if abs(p.quantity) > 1e-9}

    def net_exposure(self) -> float:
        """Sum of |market_value| across all open positions."""
        with self._lock:
            return sum(abs(p.market_value) for p in self._positions.values())

    def daily_realised_pnl(self) -> float:
        """Total realised P&L across all symbols today."""
        with self._lock:
            return sum(p.realised_pnl for p in self._positions.values())

    def daily_total_pnl(self) -> float:
        """Realised + unrealised P&L across all open positions."""
        with self._lock:
            return sum(p.total_pnl for p in self._positions.values())

    def position_count(self) -> int:
        """Number of symbols with non-zero quantity."""
        with self._lock:
            return sum(1 for p in self._positions.values() if abs(p.quantity) > 1e-9)

    def fill_count(self) -> int:
        with self._lock:
            return len(self._fills)

    def recent_fills(self, n: int = 20) -> List[FillEvent]:
        with self._lock:
            return list(self._fills[-n:])

    # ── Day reset ─────────────────────────────────────────────────────────────

    def reset_daily_pnl(self) -> None:
        """Zero out today's realised P&L counters.  Call at market open."""
        with self._lock:
            for pos in self._positions.values():
                pos.realised_pnl = 0.0
            self._fills.clear()
            self._day_start = date.today()
            logger.info("[FillTracker] Daily P&L counters reset")
