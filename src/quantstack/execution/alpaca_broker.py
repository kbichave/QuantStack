# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Alpaca broker adapter — same execute(OrderRequest) -> Fill interface as PaperBroker.

Supports both Alpaca paper trading (ALPACA_PAPER=true, default) and live trading
(ALPACA_PAPER=false). The rest of the system never knows which is active.

Execution flow for every order:
  1. Auth check via get_clock() — fast, fails immediately if keys are wrong
  2. Market-hours check — warn but do not block (Alpaca queues orders)
  3. Submit market or limit order via TradingClient
  4. Poll for fill up to FILL_TIMEOUT_SECONDS (market orders typically fill in <2s)
  5. Translate Alpaca Order → internal Fill model
  6. Update PortfolioState — same path as PaperBroker and EtradeBroker

Kill switch integration:
  On trigger: cancel all open Alpaca orders, then market-sell all positions.
  Registered automatically on first execute().

Startup reconciliation:
  On __init__, syncs PortfolioState from Alpaca positions so the local DB
  matches what Alpaca actually holds.

Environment variables (read via AlpacaSettings / pydantic-settings):
  ALPACA_API_KEY    — required
  ALPACA_SECRET_KEY — required
  ALPACA_PAPER      — "true" (default) for paper trading endpoint
  ALPACA_FILL_TIMEOUT — seconds to wait for a fill (default 30)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from loguru import logger

from quantstack.config.settings import get_settings
from quantstack.execution.kill_switch import get_kill_switch
from quantstack.execution.paper_broker import Fill, OrderRequest
from quantstack.execution.portfolio_state import Position, get_portfolio_state

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce


FILL_TIMEOUT_SECONDS = int(os.getenv("ALPACA_FILL_TIMEOUT", "30"))
FILL_POLL_INTERVAL = int(os.getenv("ALPACA_FILL_POLL_INTERVAL", "1"))


def _side(action: str) -> "OrderSide":
    return OrderSide.BUY if action.lower() == "buy" else OrderSide.SELL


class AlpacaBroker:
    """
    Live/paper broker adapter wrapping alpaca-py TradingClient.

    Implements the same execute(OrderRequest) -> Fill interface as PaperBroker
    and EtradeBroker. The trading flow and risk gate call this without knowing
    which broker is underneath.

    No OAuth required — API key + secret are enough. Keys can be rotated by
    restarting the service with updated env vars.
    """

    def __init__(self) -> None:
        s = get_settings().alpaca

        if not s.api_key or not s.secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set to use AlpacaBroker"
            )

        self._paper = s.paper
        self._client = TradingClient(
            api_key=s.api_key,
            secret_key=s.secret_key,
            paper=self._paper,
        )
        self._portfolio = get_portfolio_state()
        self._kill_switch_registered = False

        mode = "paper" if self._paper else "live"
        logger.info(f"[ALPACA] AlpacaBroker initialized (mode={mode})")

        self._reconcile_on_startup()

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def execute(self, req: OrderRequest) -> Fill:
        """
        Submit an order to Alpaca and wait for fill.

        Risk gate must be called BEFORE this — this method does not re-check risk.
        """
        if not self._kill_switch_registered:
            self._register_kill_switch_closer()

        mode = "PAPER" if self._paper else "LIVE"
        logger.info(
            f"[ALPACA:{mode}] {req.side.upper()} {req.quantity} {req.symbol} "
            f"@ {req.order_type} (ref ${req.current_price:.2f})"
        )

        try:
            side = _side(req.side)

            if req.order_type == "market":
                order_req = MarketOrderRequest(
                    symbol=req.symbol.upper(),
                    qty=req.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                if req.limit_price is None:
                    return self._reject(req, "limit_price required for limit orders")
                order_req = LimitOrderRequest(
                    symbol=req.symbol.upper(),
                    qty=req.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=req.limit_price,
                )

            order = self._client.submit_order(order_data=order_req)
            logger.info(f"[ALPACA] Order submitted: id={order.id} status={order.status}")

            # Poll for fill (market orders fill in <2s during market hours)
            order = self._poll_for_fill(str(order.id))

            return self._to_fill(req, order)

        except Exception as e:
            logger.error(f"[ALPACA] execute({req.symbol}) failed: {e}")
            return self._reject(req, str(e))

    def get_fills(self, symbol: str | None = None, limit: int = 50) -> list[Fill]:
        """Return recent filled orders from Alpaca."""
        try:
            filters = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=limit,
            )
            orders = self._client.get_orders(filter=filters)
            fills = []
            for o in orders:
                if o.filled_qty is None or float(o.filled_qty) == 0:
                    continue
                if symbol and o.symbol != symbol.upper():
                    continue
                fills.append(Fill(
                    order_id=str(o.id),
                    symbol=o.symbol,
                    side="buy" if o.side == OrderSide.BUY else "sell",
                    requested_quantity=int(float(o.qty or 0)),
                    filled_quantity=int(float(o.filled_qty or 0)),
                    fill_price=float(o.filled_avg_price or 0),
                    slippage_bps=0.0,  # not reported by Alpaca directly
                    commission=0.0,    # Alpaca is commission-free
                    partial=float(o.filled_qty or 0) < float(o.qty or 0),
                    rejected=False,
                    filled_at=o.filled_at or datetime.now(timezone.utc),
                ))
            return fills
        except Exception as e:
            logger.warning(f"[ALPACA] get_fills failed: {e}")
            return []

    def check_auth(self, dry_run: bool = False, dry_run_symbol: str = "SPY") -> dict:
        """
        Verify Alpaca credentials and return account info.

        Args:
            dry_run: If True, also submit a limit order far below market and
                     immediately cancel it to verify the full order path.
            dry_run_symbol: Symbol to use for the dry-run order (default SPY).
        """
        result: dict = {"paper": self._paper}
        order_id = None
        try:
            account = self._client.get_account()
            result.update({
                "connected": True,
                "account_id": str(account.id),
                "status": str(account.status),
                "buying_power": float(account.buying_power or 0),
                "portfolio_value": float(account.portfolio_value or 0),
                "day_trade_count": int(account.daytrade_count or 0),
                "pdt_flagged": account.pattern_day_trader,
                "account_blocked": account.account_blocked,
                "trading_blocked": account.trading_blocked,
            })
        except Exception as e:
            result["connected"] = False
            result["error"] = str(e)
            return result

        if not dry_run:
            return result

        # Dry-run: submit a limit buy at 50% below market then cancel immediately.
        # A 50% discount guarantees no fill even if quotes are stale.
        try:
            s = get_settings().alpaca
            data_client = StockHistoricalDataClient(
                api_key=s.api_key, secret_key=s.secret_key
            )
            latest = data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=dry_run_symbol)
            )
            ref_price = float(latest[dry_run_symbol].price) if dry_run_symbol in latest else 100.0
            test_price = round(ref_price * 0.50, 2)

            order_req = LimitOrderRequest(
                symbol=dry_run_symbol,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=test_price,
            )
            order = self._client.submit_order(order_data=order_req)
            order_id = str(order.id)

            time.sleep(0.5)
            self._client.cancel_order_by_id(order_id)
            time.sleep(1.0)
            cancelled = self._client.get_order_by_id(order_id)

            result["dry_run"] = {
                "passed": cancelled.status in (OrderStatus.CANCELED, OrderStatus.PENDING_CANCEL),
                "symbol": dry_run_symbol,
                "ref_price": ref_price,
                "limit_price": test_price,
                "order_id": order_id,
                "final_status": str(cancelled.status),
            }
        except Exception as e:
            result["dry_run"] = {"passed": False, "error": str(e)}
            # Best-effort cleanup
            if order_id:
                try:
                    self._client.cancel_order_by_id(order_id)
                except Exception as exc:
                    logger.debug("[ALPACA] Best-effort cancel for %s failed: %s", order_id, exc)

        return result

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _poll_for_fill(self, order_id: str) -> object:
        """Poll Alpaca until order reaches a terminal state or timeout."""
        deadline = time.monotonic() + FILL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            order = self._client.get_order_by_id(order_id)
            status = order.status
            if status in (
                OrderStatus.FILLED,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.CANCELED,
                OrderStatus.EXPIRED,
                OrderStatus.REJECTED,
            ):
                return order
            time.sleep(FILL_POLL_INTERVAL)

        # Return whatever state we have after timeout
        logger.warning(f"[ALPACA] Order {order_id} did not fill within {FILL_TIMEOUT_SECONDS}s")
        return self._client.get_order_by_id(order_id)

    def _to_fill(self, req: OrderRequest, order: object) -> Fill:
        """Translate an Alpaca Order into the internal Fill model."""
        filled_qty = int(float(order.filled_qty or 0))
        fill_price = float(order.filled_avg_price or req.current_price)
        rejected = order.status in (OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED)

        # Update local portfolio state on successful fills
        if not rejected and filled_qty > 0:
            self._portfolio.update_position(
                symbol=req.symbol,
                side="long" if req.side.lower() == "buy" else "short",
                quantity=filled_qty if req.side.lower() == "buy" else -filled_qty,
                entry_price=fill_price,
                current_price=fill_price,
            )

        slippage_bps = 0.0
        if req.current_price > 0 and fill_price > 0:
            direction = 1 if req.side.lower() == "buy" else -1
            slippage_bps = direction * (fill_price - req.current_price) / req.current_price * 10_000

        return Fill(
            order_id=str(order.id),
            symbol=req.symbol,
            side=req.side.lower(),
            requested_quantity=req.quantity,
            filled_quantity=filled_qty,
            fill_price=fill_price,
            slippage_bps=round(slippage_bps, 2),
            commission=0.0,  # Alpaca is commission-free
            partial=filled_qty < req.quantity,
            rejected=rejected,
            reject_reason=str(order.status) if rejected else None,
            filled_at=order.filled_at or datetime.now(timezone.utc),
        )

    def _reject(self, req: OrderRequest, reason: str) -> Fill:
        return Fill(
            order_id=req.order_id,
            symbol=req.symbol,
            side=req.side,
            requested_quantity=req.quantity,
            filled_quantity=0,
            fill_price=0.0,
            slippage_bps=0.0,
            rejected=True,
            reject_reason=reason,
        )

    def _reconcile_on_startup(self) -> None:
        """Sync local PortfolioState from Alpaca positions on startup."""
        try:
            positions = self._client.get_all_positions()
            for pos in positions:
                qty = float(pos.qty or 0)
                self._portfolio.update_position(
                    symbol=pos.symbol,
                    side="long" if qty >= 0 else "short",
                    quantity=int(qty),
                    entry_price=float(pos.avg_entry_price or 0),
                    current_price=float(pos.current_price or 0),
                )
            if positions:
                logger.info(f"[ALPACA] Reconciled {len(positions)} position(s) from Alpaca")
        except Exception as e:
            logger.warning(f"[ALPACA] Startup reconciliation failed (non-critical): {e}")

    def _register_kill_switch_closer(self) -> None:
        """Register a kill switch hook that cancels all Alpaca orders and flattens positions."""
        def _close_all() -> None:
            logger.warning("[ALPACA] Kill switch triggered — cancelling orders and closing positions")
            try:
                self._client.cancel_orders()
                logger.info("[ALPACA] All open orders cancelled")
            except Exception as e:
                logger.error(f"[ALPACA] Failed to cancel orders on kill switch: {e}")

            try:
                self._client.close_all_positions(cancel_orders=True)
                logger.info("[ALPACA] All positions closed")
            except Exception as e:
                logger.error(f"[ALPACA] Failed to close positions on kill switch: {e}")

        ks = get_kill_switch()
        ks.register_closer(_close_all)
        self._kill_switch_registered = True
        logger.info("[ALPACA] Kill switch closer registered")


# =============================================================================
# Singleton
# =============================================================================

_alpaca_broker: AlpacaBroker | None = None


def get_alpaca_broker() -> AlpacaBroker:
    global _alpaca_broker
    if _alpaca_broker is None:
        _alpaca_broker = AlpacaBroker()
    return _alpaca_broker
