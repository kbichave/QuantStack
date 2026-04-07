"""Execution models for bracket orders and stop-loss enforcement.

BracketIntent: immutable specification for a bracket order (entry + SL + optional TP).
BracketLeg: persisted record of a single leg's state in the bracket_legs table.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class BracketIntent(BaseModel):
    """Specification for a bracket order submission.

    stop_price is REQUIRED — this is enforced at the type level.
    No order can exist without a stop-loss.
    """

    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    entry_type: str = "market"  # "market" or "limit"
    entry_price: float | None = None
    stop_price: float  # REQUIRED — not Optional
    target_price: float | None = None
    strategy_id: str = ""
    client_order_id: str = ""


class BracketLeg(BaseModel):
    """Persisted record of a single bracket order leg."""

    parent_order_id: str
    leg_type: str  # "entry", "stop_loss", "take_profit"
    broker_order_id: str = ""
    status: str = "pending"
    price: float = 0.0
    quantity: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
