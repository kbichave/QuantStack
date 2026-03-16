"""
AsyncExecutionLoop — asyncio wiring loop that converts feature signals into orders.

Architecture
------------

    IncrementalFeatureEngine
            │ FeaturesCallback (async)
            ▼
    AsyncExecutionLoop.on_features()
            │
            ├── [signal evaluation]  →  order candidate or None
            │
            ├── PreTradeRiskGate.check()  →  RiskGateError → skip + log
            │
            ├── SmartOrderRouter.route()  →  UnifiedOrderResult
            │
            └── RiskGate.record_submission()

The loop is intentionally signal-agnostic: it accepts a user-supplied
``SignalEvaluator`` callable that decides whether to buy/sell/hold based on
an ``IncrementalFeatures`` snapshot.  This keeps trading logic out of the
infrastructure layer.

Signal evaluator contract
-------------------------
    async def evaluate(features: IncrementalFeatures) -> Optional[UnifiedOrder]:
        ...

Return ``None`` to skip (hold).  Return a ``UnifiedOrder`` to submit.

Lifecycle
---------
    loop = AsyncExecutionLoop(...)
    await loop.start()          # registers callbacks, begins processing
    ...
    await loop.stop()           # graceful shutdown, cancels tasks

Thread model
------------
The loop runs entirely inside a single asyncio task.  Broker calls
(place_order) are synchronous; the loop wraps them in
``asyncio.get_event_loop().run_in_executor(None, ...)`` so they don't
block the event loop.

Observability
-------------
``loop.stats()`` returns a dict with cumulative signal, order, fill, and
rejection counts for monitoring or MCP tool exposure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from quantcore.data.streaming.incremental_features import IncrementalFeatures
from quantcore.execution.risk_gate import PreTradeRiskGate, RiskGateError
from quantcore.execution.smart_order_router import SmartOrderRouter, SmartOrderRouterError
from quantcore.execution.unified_models import UnifiedOrder

SignalEvaluator = Callable[[IncrementalFeatures], Awaitable[UnifiedOrder | None]]


@dataclass
class _LoopStats:
    signals_received: int = 0
    signals_warm: int = 0
    orders_attempted: int = 0
    orders_placed: int = 0
    risk_rejections: int = 0
    router_errors: int = 0
    exceptions: int = 0
    symbols_seen: set = field(default_factory=set)


class AsyncExecutionLoop:
    """asyncio execution loop: features → risk gate → broker.

    Args:
        signal_evaluator: Async callable mapping IncrementalFeatures → UnifiedOrder|None.
        risk_gate:        PreTradeRiskGate instance with configured limits.
        router:           SmartOrderRouter with broker connections.
        account_id:       Broker account to route orders to.
        asset_class:      Asset class hint passed to the router (default "equity").
        price_fn:         Optional callable to get current price for a symbol;
                          used by the risk gate when the features.close is stale.
                          If None, features.close is used as the price estimate.
    """

    def __init__(
        self,
        signal_evaluator: SignalEvaluator,
        risk_gate: PreTradeRiskGate,
        router: SmartOrderRouter,
        account_id: str = "",
        asset_class: str = "equity",
        price_fn: Callable[[str], float] | None = None,
    ) -> None:
        self._evaluator = signal_evaluator
        self._risk_gate = risk_gate
        self._router = router
        self._account_id = account_id
        self._asset_class = asset_class
        self._price_fn = price_fn
        self._stats = _LoopStats()
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Mark the loop as running.  Callbacks registered via on_features() are now live."""
        self._running = True
        logger.info(
            f"[ExecLoop] Started — account={self._account_id} asset_class={self._asset_class}"
        )

    async def stop(self) -> None:
        """Gracefully stop: cancel pending tasks and mark as stopped."""
        self._running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info(
            f"[ExecLoop] Stopped — {self._stats.orders_placed} orders placed, "
            f"{self._stats.risk_rejections} risk rejections"
        )

    # ── Feature callback ──────────────────────────────────────────────────────

    async def on_features(self, features: IncrementalFeatures) -> None:
        """Called by IncrementalFeatureEngine for every new bar.

        This is the FeaturesCallback entry point.  The method is non-blocking;
        actual order work is dispatched as a fire-and-forget task so the
        feature engine's callback chain is not stalled.
        """
        if not self._running:
            return

        self._stats.signals_received += 1
        self._stats.symbols_seen.add(features.symbol)

        if not features.is_warm:
            return  # suppress signals during warmup — indicators not yet valid

        self._stats.signals_warm += 1
        task = asyncio.create_task(
            self._process_signal(features),
            name=f"exec_{features.symbol}_{features.timestamp.isoformat()}",
        )
        self._tasks.append(task)
        task.add_done_callback(self._tasks.remove)

    # ── Signal processing (runs in a task) ────────────────────────────────────

    async def _process_signal(self, features: IncrementalFeatures) -> None:
        try:
            order = await self._evaluator(features)
        except Exception as exc:
            logger.exception(f"[ExecLoop] SignalEvaluator raised for {features.symbol}: {exc}")
            self._stats.exceptions += 1
            return

        if order is None:
            return  # hold signal

        self._stats.orders_attempted += 1
        current_price = self._price_fn(features.symbol) if self._price_fn else features.close

        # Pre-trade risk gate (synchronous — fast)
        try:
            self._risk_gate.check(order, current_price)
        except RiskGateError as exc:
            logger.warning(f"[ExecLoop] Risk gate blocked {order.symbol}: {exc}")
            self._stats.risk_rejections += 1
            return

        # Broker submission (blocking I/O — run off the event loop)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._router.route(self._account_id, order, self._asset_class),
            )
            self._risk_gate.record_submission()
            self._stats.orders_placed += 1
            logger.info(
                f"[ExecLoop] Placed: {order.side.upper()} {order.quantity} "
                f"{order.symbol} → order_id={result.order_id} "
                f"status={result.status}"
            )
        except SmartOrderRouterError as exc:
            logger.error(f"[ExecLoop] Router failed for {order.symbol}: {exc}")
            self._stats.router_errors += 1
        except Exception as exc:
            logger.exception(f"[ExecLoop] Unexpected error submitting {order.symbol}: {exc}")
            self._stats.exceptions += 1

    # ── Observability ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return cumulative loop statistics for monitoring."""
        s = self._stats
        return {
            "running": self._running,
            "signals_received": s.signals_received,
            "signals_warm": s.signals_warm,
            "orders_attempted": s.orders_attempted,
            "orders_placed": s.orders_placed,
            "risk_rejections": s.risk_rejections,
            "router_errors": s.router_errors,
            "exceptions": s.exceptions,
            "symbols_tracked": len(s.symbols_seen),
            "risk_gate": self._risk_gate.status(),
        }
