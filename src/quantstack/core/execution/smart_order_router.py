"""
SmartOrderRouter — routes orders to the appropriate broker based on asset class
and availability.

Routing logic
-------------
1. Equities:         Alpaca first (REST, faster), IBKR fallback.
2. Futures/FX/FI:   IBKR only (Alpaca doesn't support these asset classes).
3. Paper mode:       If paper=True, routes ALL orders to Alpaca paper endpoint
                     regardless of asset class.
4. Broker unavailable: If the primary broker's check_auth() fails or raises,
                     the router falls back to the secondary broker and logs a
                     warning.  If both fail, raises SmartOrderRouterError.

Why not a pure registry
-----------------------
Broker routing has business logic that a simple dict can't express: asset-class
affinity, auth-health-based fallback, and paper-mode overrides.  Encoding that
in a dedicated class keeps the execution loop clean.

Thread safety
-------------
``_health_cache`` is not protected by a lock because it is only read/written
from the single execution thread.  If multiple threads call route() concurrently,
the worst case is redundant check_auth() calls — not data corruption.
"""

from __future__ import annotations

import time

from loguru import logger

from quantstack.core.execution.broker import BrokerError, BrokerInterface
from quantstack.core.execution.fill_tracker import FillEvent, FillTracker
from quantstack.core.execution.unified_models import UnifiedOrder, UnifiedOrderResult


class SmartOrderRouterError(RuntimeError):
    """Raised when no broker can service the order."""


# Asset classes that only IBKR can handle (Alpaca equities-only)
_IBKR_ONLY_TYPES = {"futures", "fx", "forex", "fixed_income", "crypto"}

# Health cache TTL — re-check auth every 5 minutes
_HEALTH_TTL_S = 300.0


class SmartOrderRouter:
    """Routes orders to Alpaca or IBKR based on asset class and availability.

    Args:
        alpaca_broker:  AlpacaBrokerClient (or None if not configured).
        ibkr_broker:    IBKRBrokerClient (or None if not configured / gateway offline).
        fill_tracker:   FillTracker to receive fill notifications after placement.
        paper:          If True, force all orders to Alpaca paper endpoint.
    """

    def __init__(
        self,
        alpaca_broker: BrokerInterface | None = None,
        ibkr_broker: BrokerInterface | None = None,
        fill_tracker: FillTracker | None = None,
        paper: bool = True,
    ) -> None:
        self._alpaca = alpaca_broker
        self._ibkr = ibkr_broker
        self._tracker = fill_tracker
        self._paper = paper
        # {broker_name: (is_healthy: bool, checked_at: float)}
        self._health: dict[str, tuple[bool, float]] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def route(
        self,
        account_id: str,
        order: UnifiedOrder,
        asset_class: str = "equity",
    ) -> UnifiedOrderResult:
        """Submit order to the appropriate broker.

        Args:
            account_id:  Broker account ID (passed through to broker.place_order).
            order:       The validated order to submit.
            asset_class: Asset class hint — "equity", "futures", "fx", etc.

        Returns:
            UnifiedOrderResult from the broker that filled the order.

        Raises:
            SmartOrderRouterError: If no healthy broker can service the order.
        """
        brokers = self._select_brokers(asset_class)
        last_error: Exception | None = None

        for name, broker in brokers:
            if not self._is_healthy(name, broker):
                logger.warning(f"[SOR] {name} unhealthy — skipping")
                continue
            try:
                result = broker.place_order(account_id, order)
                logger.info(
                    f"[SOR] Placed via {name}: {order.side.upper()} "
                    f"{order.quantity} {order.symbol} → order_id={result.order_id}"
                )
                self._record_fill(order, result)
                return result
            except BrokerError as exc:
                logger.warning(f"[SOR] {name} rejected order: {exc}")
                self._mark_unhealthy(name)
                last_error = exc

        raise SmartOrderRouterError(
            f"No broker could service {order.side} {order.quantity} {order.symbol}. "
            f"Last error: {last_error}"
        )

    def cancel(
        self, account_id: str, order_id: str, asset_class: str = "equity"
    ) -> bool:
        """Cancel an open order.  Tries brokers in the same priority order."""
        brokers = self._select_brokers(asset_class)
        for name, broker in brokers:
            try:
                ok = broker.cancel_order(account_id, order_id)
                if ok:
                    logger.info(f"[SOR] Cancelled order {order_id} via {name}")
                return ok
            except BrokerError:
                continue
        return False

    def available_brokers(self) -> list[str]:
        """Return names of brokers that are currently configured."""
        out = []
        if self._alpaca:
            out.append("alpaca")
        if self._ibkr:
            out.append("ibkr")
        return out

    # ── Routing logic ─────────────────────────────────────────────────────────

    def _select_brokers(self, asset_class: str) -> list[tuple[str, BrokerInterface]]:
        """Return (name, broker) pairs in priority order for this asset class."""
        ac = asset_class.lower()
        if self._paper or ac == "equity":
            # Alpaca first, IBKR fallback
            brokers = []
            if self._alpaca:
                brokers.append(("alpaca", self._alpaca))
            if self._ibkr:
                brokers.append(("ibkr", self._ibkr))
            return brokers

        if ac in _IBKR_ONLY_TYPES:
            if self._ibkr:
                return [("ibkr", self._ibkr)]
            raise SmartOrderRouterError(
                f"Asset class '{ac}' requires IBKR, but ibkr_broker is not configured"
            )

        # Unknown asset class — try all brokers
        brokers = []
        if self._alpaca:
            brokers.append(("alpaca", self._alpaca))
        if self._ibkr:
            brokers.append(("ibkr", self._ibkr))
        return brokers

    # ── Health tracking ───────────────────────────────────────────────────────

    def _is_healthy(self, name: str, broker: BrokerInterface) -> bool:
        """Return True if the broker passed its last auth check within the TTL."""
        entry = self._health.get(name)
        now = time.monotonic()
        if entry is None or (now - entry[1]) > _HEALTH_TTL_S:
            healthy = broker.check_auth()
            self._health[name] = (healthy, now)
            return healthy
        return entry[0]

    def _mark_unhealthy(self, name: str) -> None:
        self._health[name] = (False, time.monotonic())

    # ── Fill notification ─────────────────────────────────────────────────────

    def _record_fill(self, order: UnifiedOrder, result: UnifiedOrderResult) -> None:
        if self._tracker is None or result.filled_qty <= 0:
            return
        fill = FillEvent(
            order_id=result.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_qty=result.filled_qty,
            avg_fill_price=result.avg_fill_price or 0.0,
        )
        self._tracker.update_fill(fill)
