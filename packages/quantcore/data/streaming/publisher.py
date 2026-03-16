"""
BarPublisher — asyncio fan-out from a single bar source to N subscribers.

Why a publisher instead of direct callbacks:
  - Decouples producers (StreamingAdapters) from consumers (feature engines,
    strategy loops, live stores).
  - Each subscriber gets its own asyncio.Queue so a slow consumer cannot
    block a fast producer or other consumers.
  - Oldest-drop policy on full queues: real-time systems must never block
    on backpressure from a lagging subscriber.

Usage
-----
    publisher = BarPublisher()

    # Subscriber A: live feature engine
    queue_a = publisher.subscribe("feature_engine")
    asyncio.create_task(consume(queue_a))

    # Wire to a streaming adapter
    adapter.add_callback(publisher.on_bar)
    await adapter.subscribe(["SPY"], Timeframe.M1)

    # ... in consume():
    async def consume(q: asyncio.Queue[BarEvent]) -> None:
        while True:
            bar = await q.get()
            # process bar
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

from loguru import logger

from quantcore.data.streaming.base import BarEvent

_DEFAULT_QUEUE_DEPTH = 500


class BarPublisher:
    """Fan-out hub: one async bar callback → N subscriber queues.

    Args:
        max_queue_depth: Maximum bars each subscriber queue can hold.
                         When full, the *oldest* bar is dropped (not the
                         newest) to keep queues from growing unboundedly.
    """

    def __init__(self, max_queue_depth: int = _DEFAULT_QUEUE_DEPTH) -> None:
        self._max_depth = max_queue_depth
        self._queues: Dict[str, asyncio.Queue[Optional[BarEvent]]] = {}

    # ── Subscriber management ─────────────────────────────────────────────────

    def subscribe(self, subscriber_id: str) -> asyncio.Queue[Optional[BarEvent]]:
        """Register a subscriber and return its bar queue.

        Passing ``None`` to the queue signals shutdown (see ``shutdown()``).

        Args:
            subscriber_id: Unique label for this subscriber (used in logs).
        Returns:
            asyncio.Queue that will receive BarEvent objects.
        """
        if subscriber_id in self._queues:
            logger.warning(
                f"[Publisher] Subscriber '{subscriber_id}' already registered — "
                "returning existing queue"
            )
            return self._queues[subscriber_id]

        q: asyncio.Queue[Optional[BarEvent]] = asyncio.Queue(
            maxsize=self._max_depth
        )
        self._queues[subscriber_id] = q
        logger.debug(f"[Publisher] Subscriber '{subscriber_id}' registered")
        return q

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber."""
        self._queues.pop(subscriber_id, None)
        logger.debug(f"[Publisher] Subscriber '{subscriber_id}' removed")

    # ── Bar dispatch ──────────────────────────────────────────────────────────

    async def on_bar(self, bar: BarEvent) -> None:
        """BarCallback compatible method — publish ``bar`` to all subscribers.

        Compatible with ``StreamingAdapter.add_callback(publisher.on_bar)``.
        """
        for sub_id, q in self._queues.items():
            if q.full():
                # Drop the oldest bar to make room (non-blocking)
                try:
                    q.get_nowait()
                    logger.debug(
                        f"[Publisher] Queue '{sub_id}' full — dropped oldest bar "
                        f"({bar.symbol} @ {bar.timestamp})"
                    )
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(bar)
            except asyncio.QueueFull:
                pass  # race condition between full check and put — safe to skip

    # ── Graceful shutdown ─────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Signal all subscribers to stop by sending None to each queue."""
        for q in self._queues.values():
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        logger.info("[Publisher] Shutdown signal sent to all subscribers")
