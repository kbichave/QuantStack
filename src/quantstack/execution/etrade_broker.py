# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
eTrade broker adapter — same interface as PaperBroker, real fills.

Agents never call this directly. The flow calls get_broker() which returns
either this or PaperBroker depending on USE_REAL_TRADING.

Execution flow for every order:
  1. Auth check — fail fast if not authenticated
  2. Live quote via eTrade API — use actual market price as reference
  3. Preview order — get estimated cost/commission (required for production)
  4. Place order — market or limit via eTrade
  5. Poll for fill — wait up to FILL_TIMEOUT_SECONDS for execution
  6. Translate eTrade Order → internal Fill model
  7. Update PortfolioState — same as PaperBroker, broker-agnostic

Kill switch integration:
  On trigger: cancel all open eTrade orders, then market-sell all positions.
  Registered automatically on first use via _register_kill_switch_closer().

Startup reconciliation:
  On __init__, syncs PortfolioState from eTrade positions so the local DB
  matches what eTrade actually holds. Mismatches are logged as warnings.

Environment variables:
  USE_REAL_TRADING       = false (default) — set true to route here
  ETRADE_CONSUMER_KEY    = required
  ETRADE_CONSUMER_SECRET = required
  ETRADE_SANDBOX         = true (default) — set false for live account
  ETRADE_ACCOUNT_ID_KEY  = optional — auto-selects first account if not set
  ETRADE_FILL_TIMEOUT    = 30 (seconds to wait for a fill before giving up)
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from loguru import logger

from quantstack.execution.adapters.etrade.auth import ETradeAuthManager
from quantstack.execution.adapters.etrade.client import ETradeClient
from quantstack.execution.adapters.etrade.models import (
    OrderAction,
    OrderDuration,
    OrderLeg,
    OrderRequest as ETradeOrderRequest,
    OrderStatus,
    OrderType,
    SecurityType,
)
from quantstack.execution.kill_switch import get_kill_switch
from quantstack.execution.paper_broker import Fill, OrderRequest
from quantstack.execution.portfolio_state import Position, get_portfolio_state

# =============================================================================
# ETRADE BROKER
# =============================================================================


class EtradeBroker:
    """
    Live broker adapter wrapping the eTrade MCP client.

    Implements the same execute(OrderRequest) -> Fill interface as PaperBroker.
    The trading flow and risk gate call this without knowing it's real.

    Authentication must happen out-of-band before the first order:
      1. Call get_auth_url() to get the OAuth URL
      2. User visits URL and gets verifier code
      3. Call complete_auth(verifier_code) to save tokens

    Tokens persist to ~/.etrade_tokens.json and survive restarts.
    They expire at midnight Eastern — the broker auto-refreshes if within 2h.
    """

    FILL_TIMEOUT_SECONDS = int(os.getenv("ETRADE_FILL_TIMEOUT", "30"))
    FILL_POLL_INTERVAL = 2  # seconds between status polls

    def __init__(self):
        self._auth = ETradeAuthManager()
        self._client = ETradeClient(self._auth)
        self._portfolio = get_portfolio_state()
        self._account_id_key: str | None = os.getenv("ETRADE_ACCOUNT_ID_KEY")
        self._kill_switch_registered = False

        # Resolve account key and reconcile portfolio on startup
        if self._auth.is_authenticated():
            self._ensure_account_key()
            self._reconcile_on_startup()
        else:
            logger.warning(
                "[ETRADE] Not authenticated — call /etrade/auth before trading. "
                "Account reconciliation deferred until auth completes."
            )

    # -------------------------------------------------------------------------
    # Public interface (matches PaperBroker)
    # -------------------------------------------------------------------------

    def execute(self, req: OrderRequest) -> Fill:
        """
        Execute an order via eTrade. Returns a Fill matching PaperBroker's format.

        Risk gate must be called BEFORE this — this method does not re-check risk.
        """
        # Refresh token if expiring soon
        if self._auth.is_authenticated() and self._auth.needs_refresh():
            try:
                self._auth.refresh_token()
                logger.info("[ETRADE] Token refreshed before order execution")
            except Exception as e:
                logger.warning(f"[ETRADE] Token refresh failed: {e}")

        # Auth gate
        if not self._auth.is_authenticated():
            return self._reject(
                req, "eTrade not authenticated — call /etrade/auth first"
            )

        self._ensure_account_key()
        if not self._kill_switch_registered:
            self._register_kill_switch_closer()

        logger.info(
            f"[ETRADE] {req.side.upper()} {req.quantity} {req.symbol} "
            f"@ {req.order_type} (ref ${req.current_price:.2f})"
        )

        try:
            # Get live quote to use as reference price (override stale cached price)
            live_price = self._get_live_price(req.symbol) or req.current_price

            # Build eTrade order
            action = OrderAction.BUY if req.side.lower() == "buy" else OrderAction.SELL
            etype = OrderType.MARKET if req.order_type == "market" else OrderType.LIMIT

            etrade_req = ETradeOrderRequest(
                account_id_key=self._account_id_key,
                order_type=etype,
                price_type=etype.value,
                limit_price=req.limit_price if req.order_type == "limit" else None,
                order_term=OrderDuration.DAY,
                legs=[
                    OrderLeg(
                        symbol=req.symbol.upper(),
                        security_type=SecurityType.EQ,
                        order_action=action,
                        quantity=req.quantity,
                    )
                ],
            )

            # Preview — required for production; also gives us estimated cost for audit log
            preview = self._client.preview_order(self._account_id_key, etrade_req)
            logger.info(
                f"[ETRADE] Preview: commission=${preview.estimated_commission:.2f} "
                f"total=${preview.estimated_total_amount:.2f}"
            )

            # Log any preview warnings (margin warnings, etc.)
            for msg in preview.messages or []:
                logger.warning(f"[ETRADE] Preview message: {msg}")

            # Place order using preview_id
            order = self._client.place_order(
                self._account_id_key, etrade_req, preview.preview_id
            )

            logger.info(
                f"[ETRADE] Order placed: id={order.order_id} status={order.status}"
            )

            # Poll for fill on DAY market orders (usually fills immediately)
            if req.order_type == "market":
                order = self._poll_for_fill(order.order_id)

            return self._to_fill(req, order, live_price, preview.estimated_commission)

        except Exception as e:
            logger.error(f"[ETRADE] execute({req.symbol}) failed: {e}")
            return self._reject(req, str(e))

    def get_fills(self, limit: int = 50) -> list[Fill]:
        """
        Return recent fills from eTrade order history.

        Unlike PaperBroker, these come from the live eTrade API.
        We translate them to the internal Fill model for consistency.
        """
        if not self._auth.is_authenticated():
            return []
        try:
            orders = self._client.get_orders(self._account_id_key, status="EXECUTED")
            fills = []
            for o in orders[:limit]:
                fill = Fill(
                    order_id=str(o.order_id),
                    symbol=o.legs[0].symbol if o.legs else "UNKNOWN",
                    side=(
                        "buy"
                        if (o.legs and "BUY" in o.legs[0].order_action.value)
                        else "sell"
                    ),
                    requested_quantity=o.quantity_ordered or 0,
                    filled_quantity=o.quantity_filled or 0,
                    fill_price=o.execution_price or 0.0,
                    slippage_bps=0.0,  # eTrade doesn't report slippage directly
                    commission=o.commission or 0.0,
                    partial=(o.quantity_filled or 0) < (o.quantity_ordered or 0),
                    rejected=False,
                    filled_at=o.executed_time or datetime.now(),
                )
                fills.append(fill)
            return fills
        except Exception as e:
            logger.warning(f"[ETRADE] get_fills failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # Auth helpers (used by server.py /etrade/* endpoints)
    # -------------------------------------------------------------------------

    def get_auth_url(self) -> str:
        """Step 1 of OAuth — returns URL the user must visit."""
        return self._auth.get_authorization_url()

    def complete_auth(self, verifier_code: str) -> bool:
        """Step 2 of OAuth — exchange verifier for access token."""
        try:
            self._auth.complete_authorization(verifier_code)
            self._ensure_account_key()
            self._reconcile_on_startup()
            self._register_kill_switch_closer()
            logger.info("[ETRADE] Authentication complete")
            return True
        except Exception as e:
            logger.error(f"[ETRADE] complete_auth failed: {e}")
            return False

    def auth_status(self) -> dict:
        """Returns current auth state for the /etrade/status endpoint."""
        if not self._auth.is_authenticated():
            return {
                "authenticated": False,
                "needs_auth": True,
                "sandbox": self._auth.sandbox,
                "account_id_key": None,
                "auth_url": self.get_auth_url(),
            }
        return {
            "authenticated": True,
            "needs_auth": False,
            "sandbox": self._auth.sandbox,
            "needs_refresh": self._auth.needs_refresh(),
            "account_id_key": self._account_id_key,
        }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _ensure_account_key(self) -> None:
        """Auto-select first account if ETRADE_ACCOUNT_ID_KEY not set."""
        if self._account_id_key:
            return
        try:
            accounts = self._client.get_accounts()
            if not accounts:
                raise RuntimeError("No eTrade accounts found")
            self._account_id_key = accounts[0].account_id_key
            logger.info(
                f"[ETRADE] Auto-selected account: {accounts[0].account_name} "
                f"({self._account_id_key})"
            )
        except Exception as e:
            logger.error(f"[ETRADE] Account selection failed: {e}")

    def _get_live_price(self, symbol: str) -> float | None:
        """Fetch current market price from eTrade."""
        try:
            quotes = self._client.get_quote([symbol.upper()])
            if quotes:
                return float(quotes[0].last_price or quotes[0].close or 0.0)
        except Exception as e:
            logger.warning(f"[ETRADE] Quote fetch for {symbol} failed: {e}")
        return None

    def _poll_for_fill(self, order_id: str):
        """
        Poll eTrade order status until EXECUTED, REJECTED, or timeout.

        Market orders on equities typically fill within 1–2 seconds during
        market hours. We poll for up to FILL_TIMEOUT_SECONDS.
        """
        deadline = time.time() + self.FILL_TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                orders = self._client.get_orders(self._account_id_key)
                for o in orders:
                    if str(o.order_id) == str(order_id):
                        if o.status in (OrderStatus.EXECUTED, OrderStatus.PARTIAL):
                            return o
                        if o.status in (
                            OrderStatus.REJECTED,
                            OrderStatus.CANCELLED,
                            OrderStatus.EXPIRED,
                        ):
                            return o
            except Exception as e:
                logger.warning(f"[ETRADE] Poll error for {order_id}: {e}")
            time.sleep(self.FILL_POLL_INTERVAL)

        logger.warning(
            f"[ETRADE] Order {order_id} did not fill within {self.FILL_TIMEOUT_SECONDS}s"
        )
        # Return last known state
        try:
            orders = self._client.get_orders(self._account_id_key)
            for o in orders:
                if str(o.order_id) == str(order_id):
                    return o
        except Exception:
            pass
        return None

    def _to_fill(
        self, req: OrderRequest, order, live_price: float, commission: float
    ) -> Fill:
        """Translate eTrade Order → internal Fill model."""
        if order is None:
            return self._reject(req, "Order object not returned from eTrade")

        if order.status in (
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        ):
            return Fill(
                order_id=str(order.order_id),
                symbol=req.symbol,
                side=req.side,
                requested_quantity=req.quantity,
                filled_quantity=0,
                fill_price=0.0,
                slippage_bps=0.0,
                rejected=True,
                reject_reason=f"eTrade order {order.status.value}",
            )

        fill_price = float(order.execution_price or live_price or req.current_price)
        filled_qty = int(order.quantity_filled or req.quantity)
        ref_price = req.current_price or live_price or fill_price
        slippage_bps = (
            abs(fill_price - ref_price) / ref_price * 10_000 if ref_price > 0 else 0.0
        )

        fill = Fill(
            order_id=str(order.order_id),
            symbol=req.symbol,
            side=req.side,
            requested_quantity=req.quantity,
            filled_quantity=filled_qty,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            commission=float(order.commission or commission),
            partial=filled_qty < req.quantity,
            rejected=False,
        )

        self._update_portfolio(fill, req)
        return fill

    def _update_portfolio(self, fill: Fill, req: OrderRequest) -> None:
        """Mirror PaperBroker's portfolio update logic."""
        if fill.rejected or fill.filled_quantity == 0:
            return

        direction = 1 if fill.side.lower() == "buy" else -1
        cash_delta = (
            -direction * fill.filled_quantity * fill.fill_price - fill.commission
        )
        self._portfolio.adjust_cash(cash_delta)

        if direction > 0:
            self._portfolio.upsert_position(
                Position(
                    symbol=fill.symbol,
                    quantity=fill.filled_quantity,
                    avg_cost=fill.fill_price,
                    side="long",
                    current_price=fill.fill_price,
                )
            )
        else:
            self._portfolio.close_position(
                fill.symbol,
                exit_price=fill.fill_price,
                quantity=fill.filled_quantity,
            )

    def _reject(self, req: OrderRequest, reason: str) -> Fill:
        logger.warning(f"[ETRADE] REJECTED {req.symbol}: {reason}")
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

    # -------------------------------------------------------------------------
    # Startup reconciliation
    # -------------------------------------------------------------------------

    def _reconcile_on_startup(self) -> None:
        """
        Sync PortfolioState from eTrade's actual positions.

        Overwrites local DB with eTrade's ground truth. Any discrepancies
        are logged. This runs once per process startup after auth.
        """
        if not self._account_id_key:
            logger.warning("[ETRADE] Skipping reconciliation — no account key")
            return

        try:
            etrade_positions = self._client.get_positions(self._account_id_key)
            balance = self._client.get_account_balance(self._account_id_key)

            # Build broker position list in reconcile() format
            broker_pos = []
            for ep in etrade_positions:
                # Ignore options for now (symbol contains option notation)
                if ep.symbol and len(ep.symbol) <= 5:
                    broker_pos.append(
                        {
                            "symbol": ep.symbol,
                            "quantity": int(abs(ep.quantity)),
                            "side": "long" if ep.quantity > 0 else "short",
                            "avg_cost": (
                                float(ep.cost_basis / abs(ep.quantity))
                                if ep.quantity and ep.cost_basis
                                else 0.0
                            ),
                        }
                    )

            mismatches = self._portfolio.reconcile(broker_pos)
            if mismatches:
                logger.warning(
                    f"[ETRADE] Reconciliation: {len(mismatches)} mismatches — "
                    "local DB updated to match eTrade"
                )
                # Sync: upsert eTrade positions into local DB
                for bp in broker_pos:
                    ep_price = next(
                        (
                            e.current_price
                            for e in etrade_positions
                            if e.symbol == bp["symbol"]
                        ),
                        bp["avg_cost"],
                    )
                    self._portfolio.upsert_position(
                        Position(
                            symbol=bp["symbol"],
                            quantity=bp["quantity"],
                            avg_cost=bp["avg_cost"],
                            side=bp["side"],
                            current_price=float(ep_price or bp["avg_cost"]),
                        )
                    )
            else:
                logger.info("[ETRADE] Reconciliation clean — DB matches eTrade")

            # Sync cash from eTrade balance
            etrade_cash = float(
                balance.cash_available_for_investment or balance.net_cash or 0
            )
            local_cash = self._portfolio.get_cash()
            if abs(etrade_cash - local_cash) > 1.0:  # $1 tolerance
                logger.warning(
                    f"[ETRADE] Cash mismatch: local=${local_cash:,.2f} "
                    f"etrade=${etrade_cash:,.2f} — syncing to eTrade"
                )
                self._portfolio.adjust_cash(etrade_cash - local_cash)

        except Exception as e:
            logger.error(f"[ETRADE] Startup reconciliation failed: {e}")

    # -------------------------------------------------------------------------
    # Kill switch integration
    # -------------------------------------------------------------------------

    def _register_kill_switch_closer(self) -> None:
        """
        Register a position closer with the kill switch.

        When triggered: cancel all open eTrade orders, then market-sell
        all long positions. This runs in < 5 seconds for typical portfolios.
        """
        def close_all_etrade_positions() -> None:
            """Emergency close: cancel orders → market sell all longs."""
            logger.critical("[ETRADE] Kill switch triggered — closing all positions")
            if not self._account_id_key or not self._auth.is_authenticated():
                logger.error("[ETRADE] Cannot close positions — not authenticated")
                return

            # Step 1: Cancel all open orders
            try:
                open_orders = self._client.get_orders(
                    self._account_id_key, status="OPEN"
                )
                for order in open_orders:
                    try:
                        self._client.cancel_order(
                            self._account_id_key, str(order.order_id)
                        )
                        logger.info(f"[ETRADE] Cancelled order {order.order_id}")
                    except Exception as e:
                        logger.error(
                            f"[ETRADE] Cancel order {order.order_id} failed: {e}"
                        )
            except Exception as e:
                logger.error(f"[ETRADE] Get open orders failed during kill: {e}")

            # Step 2: Market sell all long equity positions
            try:
                etrade_positions = self._client.get_positions(self._account_id_key)
                for ep in etrade_positions:
                    if (
                        ep.quantity
                        and ep.quantity > 0
                        and ep.symbol
                        and len(ep.symbol) <= 5
                    ):
                        try:
                            req = OrderRequest(
                                symbol=ep.symbol,
                                side="sell",
                                quantity=int(ep.quantity),
                                order_type="market",
                                current_price=float(ep.current_price or 0.0),
                            )
                            self.execute(req)
                            logger.info(f"[ETRADE] Closed position: {ep.symbol}")
                        except Exception as e:
                            logger.error(
                                f"[ETRADE] Close position {ep.symbol} failed: {e}"
                            )
            except Exception as e:
                logger.error(f"[ETRADE] Get positions failed during kill: {e}")

        get_kill_switch().register_position_closer(close_all_etrade_positions)
        self._kill_switch_registered = True
        logger.info("[ETRADE] Kill switch position closer registered")


# =============================================================================
# SINGLETON
# =============================================================================

_etrade_broker: EtradeBroker | None = None


def get_etrade_broker() -> EtradeBroker:
    """Get the singleton EtradeBroker instance."""
    global _etrade_broker
    if _etrade_broker is None:
        _etrade_broker = EtradeBroker()
    return _etrade_broker
