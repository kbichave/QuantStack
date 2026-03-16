"""
Real-time streaming infrastructure for live bar and tick data.

Architecture
------------
Bar-level (1-minute / 5-second aggregated):
    StreamingAdapter (base.py)
        └── AlpacaStreamingAdapter  — Alpaca StockDataStream WebSocket
        └── PolygonStreamingAdapter — Polygon AM.* WebSocket
        └── IBKRStreamingAdapter    — ib_insync reqRealTimeBars (5s → 1m aggregation)

Tick-level (individual trades + NBBO quotes + L2 depth):
    TickStreamingAdapter (tick_base.py)
        └── PolygonTickAdapter — Polygon T.* trades + Q.* quotes WebSocket
        └── IBKRTickAdapter    — ib_insync reqTickByTickData + reqMktDepth

Bar pipeline:
    StreamingAdapter (or TickAggregator wrapping a TickStreamingAdapter)
        → BarPublisher  (fan-out to N async subscribers)
            → LiveBarStore              (write-through DuckDB + in-memory deque)
            → IncrementalFeatureEngine  (rolling O(1) feature updates)
"""

from quantcore.data.streaming.base import StreamingAdapter, BarEvent
from quantcore.data.streaming.publisher import BarPublisher
from quantcore.data.streaming.live_store import LiveBarStore
from quantcore.data.streaming.incremental_features import (
    IncrementalFeatureEngine,
    IncrementalFeatures,
)
from quantcore.data.streaming.tick_models import TradeTick, QuoteTick, L2Update
from quantcore.data.streaming.tick_base import TickStreamingAdapter
from quantcore.data.streaming.tick_aggregator import TickAggregator

__all__ = [
    # Bar streaming
    "StreamingAdapter",
    "BarEvent",
    "BarPublisher",
    "LiveBarStore",
    "IncrementalFeatureEngine",
    "IncrementalFeatures",
    # Tick streaming
    "TickStreamingAdapter",
    "TradeTick",
    "QuoteTick",
    "L2Update",
    "TickAggregator",
]
