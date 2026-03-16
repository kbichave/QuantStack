"""
AlpacaBrokerClient — BrokerInterface implementation wrapping alpaca-py.

Uses:
    alpaca.trading.client.TradingClient  — account, orders, positions
    alpaca.data.historical.StockHistoricalDataClient  — quotes and bars

Auth: static API keys (ALPACA_API_KEY + ALPACA_SECRET_KEY).
      No OAuth flow — keys are permanent.

Paper vs live
-------------
Set ``ALPACA_PAPER=true`` (default) to use paper trading endpoint.
Set ``ALPACA_PAPER=false`` for live trading.  The only difference is the
base URL used by ``TradingClient``; the data feed is the same.
"""

from __future__ import annotations

from loguru import logger
from quantcore.execution.broker import (
    BrokerAuthError,
    BrokerConnectionError,
    BrokerError,
    BrokerInterface,
    BrokerOrderError,
)
from quantcore.execution.unified_models import (
    UnifiedAccount,
    UnifiedBalance,
    UnifiedOrder,
    UnifiedOrderPreview,
    UnifiedOrderResult,
    UnifiedPosition,
    UnifiedQuote,
)

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
        StopLimitOrderRequest,
        StopOrderRequest,
    )

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


def _require_alpaca() -> None:
    if not _ALPACA_AVAILABLE:
        raise ImportError(
            "alpaca-py is required for AlpacaBrokerClient. Run: uv pip install -e '.[alpaca]'"
        )


class AlpacaBrokerClient(BrokerInterface):
    """BrokerInterface implementation for Alpaca Markets.

    Args:
        api_key:    Alpaca API key (falls back to ALPACA_API_KEY env var).
        secret_key: Alpaca secret key (falls back to ALPACA_SECRET_KEY env var).
        paper:      True = paper trading endpoint, False = live.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ) -> None:
        _require_alpaca()
        import os

        self._api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        if not self._api_key or not self._secret_key:
            raise BrokerAuthError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY are required. Add them to your .env file."
            )
        self._paper = paper
        self._trading = TradingClient(self._api_key, self._secret_key, paper=paper)
        self._data_cli = StockHistoricalDataClient(self._api_key, self._secret_key)

    # ── Account & balance ─────────────────────────────────────────────────────

    def get_accounts(self) -> list[UnifiedAccount]:
        try:
            acct = self._trading.get_account()
        except Exception as exc:
            raise BrokerConnectionError(f"Alpaca get_account failed: {exc}") from exc
        return [
            UnifiedAccount(
                account_id=str(acct.id),
                account_type=str(acct.account_type) if hasattr(acct, "account_type") else "margin",
                currency="USD",
                status=str(acct.status),
            )
        ]

    def get_balance(self, account_id: str) -> UnifiedBalance:
        try:
            acct = self._trading.get_account()
        except Exception as exc:
            raise BrokerConnectionError(f"Alpaca get_account failed: {exc}") from exc
        return UnifiedBalance(
            account_id=str(acct.id),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            portfolio_value=float(acct.portfolio_value),
            day_trade_buying_power=float(acct.daytrading_buying_power)
            if hasattr(acct, "daytrading_buying_power")
            else None,
            maintenance_margin=float(acct.maintenance_margin)
            if hasattr(acct, "maintenance_margin")
            else None,
        )

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self, account_id: str) -> list[UnifiedPosition]:
        try:
            positions = self._trading.get_all_positions()
        except Exception as exc:
            raise BrokerConnectionError(f"Alpaca get_positions failed: {exc}") from exc
        return [
            UnifiedPosition(
                account_id=account_id,
                symbol=p.symbol,
                quantity=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealised_pnl=float(p.unrealized_pl),
                unrealised_pnl_pct=float(p.unrealized_plpc),
                side=str(p.side),
            )
            for p in positions
        ]

    # ── Market data ───────────────────────────────────────────────────────────

    def get_quote(self, symbols: list[str]) -> list[UnifiedQuote]:
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self._data_cli.get_stock_latest_quote(req)
        except Exception as exc:
            raise BrokerConnectionError(f"Alpaca get_quote failed: {exc}") from exc
        results = []
        for sym in symbols:
            q = quotes.get(sym)
            if q:
                results.append(
                    UnifiedQuote(
                        symbol=sym,
                        bid=float(q.bid_price),
                        ask=float(q.ask_price),
                        last=float(q.ask_price),  # last not in quote; use ask
                        bid_size=float(q.bid_size),
                        ask_size=float(q.ask_size),
                        timestamp=q.timestamp,
                    )
                )
            else:
                results.append(UnifiedQuote(symbol=sym, bid=0, ask=0, last=0))
        return results

    # ── Orders ────────────────────────────────────────────────────────────────

    def preview_order(self, account_id: str, order: UnifiedOrder) -> UnifiedOrderPreview:
        # Alpaca does not have a preview/dry-run endpoint.
        # Return a basic estimate: zero commission (Alpaca is commission-free),
        # estimated fill at current mid-price.
        try:
            quotes = self.get_quote([order.symbol])
            mid = quotes[0].mid if quotes else 0.0
        except BrokerError:
            mid = order.limit_price or 0.0

        fill_price = order.limit_price or mid
        return UnifiedOrderPreview(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            estimated_fill_price=fill_price,
            estimated_commission=0.0,
            estimated_total_cost=fill_price * order.quantity,
            warnings=["Alpaca does not provide a preview endpoint; cost is estimated"],
        )

    def place_order(self, account_id: str, order: UnifiedOrder) -> UnifiedOrderResult:
        side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        tif = tif_map.get(order.time_in_force.lower(), TimeInForce.DAY)

        try:
            if order.order_type == "market":
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    extended_hours=order.extended_hours,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == "limit":
                if order.limit_price is None:
                    raise BrokerOrderError("limit_price required for limit orders", order)
                req = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                    extended_hours=order.extended_hours,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == "stop":
                req = StopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                )
            elif order.order_type == "stop_limit":
                req = StopLimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                )
            else:
                raise BrokerOrderError(f"Unsupported order type: {order.order_type}", order)

            result = self._trading.submit_order(req)
        except BrokerOrderError:
            raise
        except Exception as exc:
            raise BrokerOrderError(f"Alpaca order failed: {exc}", order) from exc

        return UnifiedOrderResult(
            order_id=str(result.id),
            client_order_id=str(result.client_order_id) if result.client_order_id else None,
            symbol=result.symbol,
            side=str(result.side),
            quantity=float(result.qty),
            order_type=str(result.order_type),
            limit_price=float(result.limit_price) if result.limit_price else None,
            stop_price=float(result.stop_price) if result.stop_price else None,
            status=str(result.status),
            filled_qty=float(result.filled_qty) if result.filled_qty else 0.0,
            avg_fill_price=float(result.filled_avg_price) if result.filled_avg_price else None,
            created_at=result.created_at,
        )

    def cancel_order(self, account_id: str, order_id: str) -> bool:
        try:
            self._trading.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            logger.warning(f"[Alpaca] Cancel order {order_id} failed: {exc}")
            return False

    def get_orders(
        self,
        account_id: str,
        status: str | None = None,
    ) -> list[UnifiedOrderResult]:
        try:
            req_kwargs = {}
            if status == "open":
                req_kwargs["status"] = OrderStatus.OPEN
            elif status == "filled":
                req_kwargs["status"] = OrderStatus.FILLED
            elif status == "cancelled":
                req_kwargs["status"] = OrderStatus.CANCELED

            req = GetOrdersRequest(**req_kwargs)
            orders = self._trading.get_orders(req)
        except Exception as exc:
            raise BrokerConnectionError(f"Alpaca get_orders failed: {exc}") from exc

        return [
            UnifiedOrderResult(
                order_id=str(o.id),
                client_order_id=str(o.client_order_id) if o.client_order_id else None,
                symbol=o.symbol,
                side=str(o.side),
                quantity=float(o.qty),
                order_type=str(o.order_type),
                limit_price=float(o.limit_price) if o.limit_price else None,
                stop_price=float(o.stop_price) if o.stop_price else None,
                status=str(o.status),
                filled_qty=float(o.filled_qty) if o.filled_qty else 0.0,
                avg_fill_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                created_at=o.created_at,
                updated_at=o.updated_at,
            )
            for o in orders
        ]
