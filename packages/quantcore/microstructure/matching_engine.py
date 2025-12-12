"""
Matching Engine for Order Book.

Price-time priority matching with support for multiple order types.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from quantcore.microstructure.order_book import OrderBook, Order, OrderType, Side


@dataclass
class Fill:
    """Trade execution fill."""

    price: float
    quantity: float
    aggressor_id: int
    passive_id: int
    side: Side


@dataclass
class ExecutionReport:
    """Report of order execution."""

    order_id: int
    fills: List[Fill]
    total_filled: float
    avg_price: float
    remaining: float
    status: str


class MatchingEngine:
    """
    Order matching engine with price-time priority.

    Example:
        engine = MatchingEngine()
        engine.submit_order(Order(1, Side.BID, 100.0, 10))
        engine.submit_order(Order(2, Side.ASK, 101.0, 10))
        fills = engine.submit_order(Order(3, Side.BID, 101.0, 5, OrderType.MARKET))
    """

    def __init__(self):
        self.book = OrderBook()
        self.order_counter = 0
        self.fills: List[Fill] = []

    def submit_order(self, order: Order) -> ExecutionReport:
        """Submit order to matching engine."""
        if order.order_id == 0:
            self.order_counter += 1
            order.order_id = self.order_counter

        fills = []

        if order.order_type == OrderType.MARKET:
            fills = self._execute_market_order(order)
        else:
            fills = self._process_limit_order(order)

        self.fills.extend(fills)

        total_filled = sum(f.quantity for f in fills)
        avg_price = (
            sum(f.price * f.quantity for f in fills) / total_filled
            if total_filled > 0
            else 0
        )

        status = (
            "filled"
            if total_filled >= order.quantity
            else "partial" if total_filled > 0 else "resting"
        )

        return ExecutionReport(
            order_id=order.order_id,
            fills=fills,
            total_filled=total_filled,
            avg_price=avg_price,
            remaining=order.remaining,
            status=status,
        )

    def _execute_market_order(self, order: Order) -> List[Fill]:
        """Execute market order against resting orders."""
        fills = []
        levels = self.book.asks if order.side == Side.BID else self.book.bids
        prices = (
            sorted(levels.keys())
            if order.side == Side.BID
            else sorted(levels.keys(), reverse=True)
        )

        for price in prices:
            if order.remaining <= 0:
                break

            level = levels[price]
            for passive in level.orders[:]:
                if order.remaining <= 0:
                    break

                fill_qty = min(order.remaining, passive.remaining)
                order.remaining -= fill_qty
                passive.remaining -= fill_qty

                fills.append(
                    Fill(
                        price=price,
                        quantity=fill_qty,
                        aggressor_id=order.order_id,
                        passive_id=passive.order_id,
                        side=order.side,
                    )
                )

                if passive.remaining <= 0:
                    level.orders.remove(passive)
                    del self.book.orders[passive.order_id]

            if not level.orders:
                del levels[price]

        return fills

    def _process_limit_order(self, order: Order) -> List[Fill]:
        """Process limit order (may cross or rest)."""
        fills = []

        if order.side == Side.BID:
            while (
                order.remaining > 0
                and self.book.best_ask
                and order.price >= self.book.best_ask
            ):
                fills.extend(self._match_at_price(order, self.book.best_ask, Side.ASK))
        else:
            while (
                order.remaining > 0
                and self.book.best_bid
                and order.price <= self.book.best_bid
            ):
                fills.extend(self._match_at_price(order, self.book.best_bid, Side.BID))

        if order.remaining > 0:
            self.book.add_order(order)

        return fills

    def _match_at_price(
        self, aggressor: Order, price: float, passive_side: Side
    ) -> List[Fill]:
        """Match aggressor order against passive orders at price."""
        fills = []
        levels = self.book.bids if passive_side == Side.BID else self.book.asks

        if price not in levels:
            return fills

        level = levels[price]

        for passive in level.orders[:]:
            if aggressor.remaining <= 0:
                break

            fill_qty = min(aggressor.remaining, passive.remaining)
            aggressor.remaining -= fill_qty
            passive.remaining -= fill_qty

            fills.append(
                Fill(
                    price=price,
                    quantity=fill_qty,
                    aggressor_id=aggressor.order_id,
                    passive_id=passive.order_id,
                    side=aggressor.side,
                )
            )

            if passive.remaining <= 0:
                level.orders.remove(passive)
                del self.book.orders[passive.order_id]

        if not level.orders:
            del levels[price]

        return fills

    def get_vwap(self, n_fills: int = 100) -> float:
        """Get VWAP of recent fills."""
        recent = self.fills[-n_fills:]
        if not recent:
            return 0.0

        total_value = sum(f.price * f.quantity for f in recent)
        total_qty = sum(f.quantity for f in recent)

        return total_value / total_qty if total_qty > 0 else 0.0
