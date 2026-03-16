"""
Broker-agnostic unified execution models.

These dataclasses are the common currency exchanged between ``BrokerInterface``
implementations (Alpaca, IBKR, E*Trade) and the strategy layer.  No provider-
specific fields.  Provider clients translate to/from these models.

Design
------
- Dataclasses (not Pydantic) to stay consistent with the rest of quantcore.
- All monetary values in USD.
- Quantities in shares / contracts (not lots).
- Timestamps are UTC-aware datetimes.

Serialisation
-------------
``asdict()`` from dataclasses works out of the box.  datetime fields require
``.isoformat()`` before JSON serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Account & Balance
# ---------------------------------------------------------------------------


@dataclass
class UnifiedAccount:
    """Top-level account descriptor."""

    account_id: str
    account_type: str  # "margin", "cash", "ira", etc.
    currency: str = "USD"
    status: str = "active"


@dataclass
class UnifiedBalance:
    """Account balance and buying power."""

    account_id: str
    cash: float  # settled cash
    buying_power: float  # available for new positions
    portfolio_value: float  # total market value of positions + cash
    day_trade_buying_power: float | None = None  # PDT accounts
    maintenance_margin: float | None = None
    currency: str = "USD"
    as_of: datetime | None = None


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@dataclass
class UnifiedPosition:
    """A currently held position."""

    account_id: str
    symbol: str
    quantity: float  # shares held (negative = short)
    avg_entry_price: float
    current_price: float
    market_value: float  # quantity × current_price
    unrealised_pnl: float
    unrealised_pnl_pct: float
    side: str  # "long" | "short"


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


@dataclass
class UnifiedQuote:
    """Real-time or delayed best-bid/offer snapshot."""

    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: float | None = None
    ask_size: float | None = None
    volume: float | None = None
    timestamp: datetime | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@dataclass
class UnifiedOrder:
    """Order submission request.

    Use ``UnifiedOrderResult`` for the broker's response after placement.
    """

    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    order_type: str = "market"  # "market" | "limit" | "stop" | "stop_limit"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "day"  # "day" | "gtc" | "ioc" | "fok"
    extended_hours: bool = False
    client_order_id: str | None = None


@dataclass
class UnifiedOrderPreview:
    """Cost/commission estimate before placing an order."""

    symbol: str
    side: str
    quantity: float
    estimated_fill_price: float
    estimated_commission: float
    estimated_total_cost: float  # including commission
    currency: str = "USD"
    warnings: list[str] = field(default_factory=list)


@dataclass
class UnifiedOrderResult:
    """Broker's response after an order is placed or queried."""

    order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: float | None
    stop_price: float | None
    status: str  # "pending" | "open" | "filled" | "cancelled" | "rejected"
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    commission: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    reject_reason: str | None = None
