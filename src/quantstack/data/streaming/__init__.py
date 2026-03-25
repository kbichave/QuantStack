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
            → LiveBarStore              (write-through PostgreSQL + in-memory deque)
            → IncrementalFeatureEngine  (rolling O(1) feature updates)
"""

from quantstack.data.streaming.base import BarEvent, StreamingAdapter
from quantstack.data.streaming.incremental_features import (
    IncrementalFeatureEngine,
    IncrementalFeatures,
)
from quantstack.data.streaming.live_store import LiveBarStore
from quantstack.data.streaming.publisher import BarPublisher
from quantstack.data.streaming.tick_aggregator import TickAggregator
from quantstack.data.streaming.stream_manager import StreamManager, get_stream_manager
from quantstack.data.streaming.tick_base import TickStreamingAdapter
from quantstack.data.streaming.tick_models import L2Update, QuoteTick, TradeTick

__all__ = [
    # Bar streaming
    "StreamingAdapter",
    "BarEvent",
    "BarPublisher",
    "LiveBarStore",
    "IncrementalFeatureEngine",
    "IncrementalFeatures",
    # Orchestration
    "StreamManager",
    "get_stream_manager",
    # Tick streaming
    "TickStreamingAdapter",
    "TradeTick",
    "QuoteTick",
    "L2Update",
    "TickAggregator",
]
