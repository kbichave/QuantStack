"""
Tick-level data models for HFT / microstructure analysis.

Design notes
------------
Nanosecond timestamps
    Stored as ``int`` (Python arbitrary-precision int) rather than ``datetime``
    or ``float``.  Float64 has only 15-16 significant digits; a nanosecond Unix
    timestamp in 2025 is ~1.74e18 ns which saturates float precision at the
    microsecond level.  Integer ns avoids all rounding.  Convert to ``datetime``
    only when displaying or writing to a non-ns-capable sink.

Side encoding
    String literals "buy" | "sell" | "unknown" rather than an Enum.  Tick data
    arrives from multiple providers with different aggressor encodings; keeping
    it as a string prevents import cycles and simplifies JSON serialisation.

L2Update semantics
    action="add"    — new price level or size increase at a level
    action="modify" — size at an existing level has changed
    action="delete" — level removed (size = 0)

    Providers differ on whether they send full-book snapshots (then incremental
    updates) or only changes.  ``OrderBookReconstructor`` handles both via the
    ``is_snapshot`` flag on the first batch.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Trade tick
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TradeTick:
    """A single executed trade (last-sale print).

    Attributes:
        symbol:       Ticker symbol.
        timestamp_ns: Execution timestamp in UTC nanoseconds since Unix epoch.
        price:        Trade price.
        size:         Trade size (shares / contracts).
        side:         Aggressor side: "buy" | "sell" | "unknown".
        exchange:     Exchange or venue code (e.g. "N" for NYSE, "Q" for NASDAQ).
        trade_id:     Provider-assigned trade identifier (optional).
        conditions:   Trade condition codes e.g. ["@", "I"] (optional).
    """

    symbol: str
    timestamp_ns: int
    price: float
    size: float
    side: str = "unknown"  # "buy" | "sell" | "unknown"
    exchange: str | None = None
    trade_id: str | None = None
    conditions: list | None = None

    @property
    def timestamp_s(self) -> float:
        """Float seconds since Unix epoch (loses ns precision for display only)."""
        return self.timestamp_ns / 1_000_000_000


# ---------------------------------------------------------------------------
# Quote tick (NBBO update)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QuoteTick:
    """A best-bid / best-offer (NBBO) quote update.

    Attributes:
        symbol:       Ticker symbol.
        timestamp_ns: Quote timestamp in UTC nanoseconds since Unix epoch.
        bid:          National best bid price.
        ask:          National best ask price.
        bid_size:     Size at best bid (shares).
        ask_size:     Size at best ask (shares).
        bid_exchange: Exchange posting the best bid (optional).
        ask_exchange: Exchange posting the best ask (optional).
    """

    symbol: str
    timestamp_ns: int
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    bid_exchange: str | None = None
    ask_exchange: str | None = None

    @property
    def spread(self) -> float:
        """Bid-ask spread in price units."""
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        """Midpoint price."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        """Spread in basis points relative to mid."""
        return self.spread / self.mid * 10_000 if self.mid != 0.0 else 0.0


# ---------------------------------------------------------------------------
# Level-2 order book update
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class L2Update:
    """An incremental update to a limit order book (Level 2 data).

    Attributes:
        symbol:       Ticker symbol.
        timestamp_ns: Update timestamp in UTC nanoseconds since Unix epoch.
        side:         "bid" | "ask".
        price:        Price level.
        size:         New total size at this level (0 = level removed).
        action:       "add" | "modify" | "delete".
        is_snapshot:  True if this message is part of a full-book snapshot
                      (rather than an incremental update).  The reconstructor
                      clears the book before processing the first snapshot batch.
        level_count:  Number of levels in this snapshot message batch (optional).
    """

    symbol: str
    timestamp_ns: int
    side: str  # "bid" | "ask"
    price: float
    size: float
    action: str = "modify"  # "add" | "modify" | "delete"
    is_snapshot: bool = False
    level_count: int | None = None
