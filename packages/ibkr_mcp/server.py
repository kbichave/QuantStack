"""
ibkr_mcp server — FastMCP server exposing Interactive Brokers tools.

Start with: ``ibkr-mcp`` or ``python -m ibkr_mcp.server``

Auth (environment variables):
    IBKR_HOST       IB Gateway host (default 127.0.0.1)
    IBKR_PORT       Gateway port: 4001=IB Gateway, 7497=TWS (default 4001)
    IBKR_CLIENT_ID  Client ID (default 1)

If IB Gateway is not running at startup, the server starts in a degraded
state.  All tools return {\"success\": false, \"error\": \"...\"} until the
gateway comes online and ``connect_gateway`` is called.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any, List, Optional

from fastmcp import FastMCP
from loguru import logger


# ---------------------------------------------------------------------------
# Server context
# ---------------------------------------------------------------------------

class _Ctx:
    mgr: Any       = None   # IBKRConnectionManager
    account_id: str = ""


_ctx = _Ctx()


@asynccontextmanager
async def lifespan(server: FastMCP):
    logger.info("[ibkr_mcp] Starting…")
    try:
        from ibkr_mcp.connection import IBKRConnectionManager

        _ctx.mgr = IBKRConnectionManager.get_instance(
            host      = os.getenv("IBKR_HOST",      "127.0.0.1"),
            port      = int(os.getenv("IBKR_PORT",  "4001")),
            client_id = int(os.getenv("IBKR_CLIENT_ID", "1")),
        )
        _ctx.mgr.connect()
        accounts = _ctx.mgr.ib.managedAccounts()
        _ctx.account_id = accounts[0] if accounts else ""
        logger.info(f"[ibkr_mcp] Connected  account={_ctx.account_id}")
    except Exception as exc:
        logger.warning(
            f"[ibkr_mcp] IB Gateway not available at startup: {exc}. "
            "Tools will return errors until gateway is connected."
        )

    yield
    if _ctx.mgr:
        _ctx.mgr.disconnect()
    logger.info("[ibkr_mcp] Shutdown")


mcp = FastMCP("ibkr_mcp", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_ib():
    if _ctx.mgr is None or not _ctx.mgr.is_connected():
        raise RuntimeError(
            f"IB Gateway not connected at "
            f"{os.getenv('IBKR_HOST','127.0.0.1')}:"
            f"{os.getenv('IBKR_PORT','4001')}. "
            "Start IB Gateway and call connect_gateway."
        )
    return _ctx.mgr.ib


def _dc_to_dict(obj) -> dict:
    d = asdict(obj)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def connect_gateway(
    host: str = "127.0.0.1",
    port: int = 4001,
) -> dict:
    """Explicitly connect to IB Gateway.  Use after gateway starts post-server-launch.

    Args:
        host: IB Gateway host.
        port: 4001 for IB Gateway, 7497 for TWS.
    """
    try:
        from ibkr_mcp.connection import IBKRConnectionManager
        mgr = IBKRConnectionManager.get_instance(host=host, port=port)
        mgr.connect()
        accounts = mgr.ib.managedAccounts()
        _ctx.mgr = mgr
        _ctx.account_id = accounts[0] if accounts else ""
        return {
            "success":    True,
            "host":       host,
            "port":       port,
            "account_id": _ctx.account_id,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_connection_status() -> dict:
    """Check whether IB Gateway is connected."""
    connected = _ctx.mgr is not None and _ctx.mgr.is_connected()
    return {
        "success":    True,
        "connected":  connected,
        "host":       os.getenv("IBKR_HOST", "127.0.0.1"),
        "port":       int(os.getenv("IBKR_PORT", "4001")),
        "account_id": _ctx.account_id,
    }


@mcp.tool()
def get_accounts() -> dict:
    """List all accounts managed by IB Gateway."""
    try:
        ib = _require_ib()
        accounts = ib.managedAccounts()
        return {"success": True, "accounts": accounts}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_balance(account_id: Optional[str] = None) -> dict:
    """Return cash, net liquidation value, and buying power.

    Args:
        account_id: IB account ID (uses default account if omitted).
    """
    try:
        ib     = _require_ib()
        acct   = account_id or _ctx.account_id
        values = ib.accountValues(acct)
        summary = {
            v.tag: float(v.value) if _is_float(v.value) else v.value
            for v in values
            if v.tag in (
                "NetLiquidation", "TotalCashValue", "BuyingPower",
                "MaintMarginReq", "AvailableFunds", "UnrealizedPnL",
            )
            and v.currency == "USD"
        }
        return {"success": True, "account_id": acct, "balance": summary}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_positions(account_id: Optional[str] = None) -> dict:
    """Return all open positions.

    Args:
        account_id: IB account ID (uses default account if omitted).
    """
    try:
        ib   = _require_ib()
        acct = account_id or _ctx.account_id
        port = ib.portfolio(acct)
        positions = []
        for item in port:
            positions.append({
                "symbol":        item.contract.symbol,
                "sec_type":      item.contract.secType,
                "quantity":      item.position,
                "avg_cost":      item.averageCost,
                "market_price":  item.marketPrice,
                "market_value":  item.marketValue,
                "unrealised_pnl": item.unrealizedPNL,
                "realised_pnl":  item.realizedPNL,
            })
        return {"success": True, "positions": positions, "count": len(positions)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_quote(symbols: List[str]) -> dict:
    """Return real-time snapshot bid/ask/last for up to 20 symbols.

    Args:
        symbols: List of ticker symbols.
    """
    try:
        import ib_insync as iblib
        ib = _require_ib()
        contracts = [iblib.Stock(s, "SMART", "USD") for s in symbols]
        tickers   = ib.reqTickers(*contracts)
        quotes = []
        for sym, t in zip(symbols, tickers):
            quotes.append({
                "symbol": sym,
                "bid":    t.bid   if t.bid   and t.bid   > 0 else None,
                "ask":    t.ask   if t.ask   and t.ask   > 0 else None,
                "last":   t.last  if t.last  and t.last  > 0 else None,
                "close":  t.close if t.close and t.close > 0 else None,
            })
        return {"success": True, "quotes": quotes}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_historical_bars(
    symbol:    str,
    timeframe: str = "1h",
    duration:  str = "30 D",
) -> dict:
    """Return OHLCV history from IB Gateway reqHistoricalData.

    Args:
        symbol:    Ticker symbol.
        timeframe: \"5 secs\" | \"1 min\" | \"5 mins\" | \"15 mins\" | \"1 hour\" | \"1 day\".
        duration:  IB duration string, e.g. \"30 D\", \"1 W\", \"3 M\".
    """
    try:
        import ib_insync as iblib
        ib       = _require_ib()
        contract = iblib.Stock(symbol, "SMART", "USD")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime       = "",
            durationStr       = duration,
            barSizeSetting    = timeframe,
            whatToShow        = "TRADES",
            useRTH            = True,
            formatDate        = 1,
        )
        records = [
            {
                "timestamp": b.date.isoformat() if hasattr(b.date, "isoformat") else str(b.date),
                "open":   float(b.open),
                "high":   float(b.high),
                "low":    float(b.low),
                "close":  float(b.close),
                "volume": float(b.volume),
            }
            for b in bars
        ]
        return {"success": True, "symbol": symbol, "bars": records}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_option_chains(
    symbol: str,
    expiry: Optional[str] = None,
) -> dict:
    """Return options chain expirations and strikes from IB Gateway.

    Args:
        symbol: Underlying ticker symbol.
        expiry: Target expiry YYYYMMDD (returns all expirations if omitted).
    """
    try:
        import ib_insync as iblib
        ib       = _require_ib()
        contract = iblib.Stock(symbol, "SMART", "USD")
        chains   = ib.reqSecDefOptParams(symbol, "", "STK", contract.conId or 0)
        results = []
        for c in chains:
            exp_filter = [e for e in c.expirations if not expiry or e == expiry]
            results.append({
                "exchange":    c.exchange,
                "expirations": exp_filter,
                "strikes":     list(c.strikes)[:20],   # first 20 strikes for brevity
            })
        return {"success": True, "symbol": symbol, "chains": results}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def place_order(
    symbol:      str,
    side:        str,
    quantity:    float,
    order_type:  str = "market",
    limit_price: Optional[float] = None,
    account_id:  Optional[str] = None,
) -> dict:
    """Submit an equity order via IB Gateway.

    Args:
        symbol:      Ticker symbol.
        side:        \"BUY\" or \"SELL\".
        quantity:    Number of shares.
        order_type:  \"market\" | \"limit\".
        limit_price: Required for limit orders.
        account_id:  IB account ID (uses default if omitted).
    """
    try:
        import ib_insync as iblib
        ib       = _require_ib()
        acct     = account_id or _ctx.account_id
        contract = iblib.Stock(symbol, "SMART", "USD")

        if order_type.lower() == "market":
            order = iblib.MarketOrder(side.upper(), quantity, account=acct)
        elif order_type.lower() == "limit":
            if limit_price is None:
                return {"success": False, "error": "limit_price required for limit orders"}
            order = iblib.LimitOrder(side.upper(), quantity, limit_price, account=acct)
        else:
            return {"success": False, "error": f"Unsupported order_type: {order_type}"}

        trade = ib.placeOrder(contract, order)
        return {
            "success":  True,
            "order_id": trade.order.orderId,
            "status":   str(trade.orderStatus.status),
            "symbol":   symbol,
            "side":     side,
            "quantity": quantity,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def cancel_order(order_id: int) -> dict:
    """Cancel an open IB order by order ID.

    Args:
        order_id: IB integer order ID.
    """
    try:
        import ib_insync as iblib
        ib = _require_ib()
        trades = [t for t in ib.trades() if t.order.orderId == order_id]
        if not trades:
            return {"success": False, "error": f"Order {order_id} not found"}
        ib.cancelOrder(trades[0].order)
        return {"success": True, "order_id": order_id}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_orders(status: Optional[str] = None) -> dict:
    """Return open and/or completed orders.

    Args:
        status: \"open\" | \"filled\" | None for all.
    """
    try:
        ib = _require_ib()
        trades = ib.trades()
        results = []
        for t in trades:
            s = str(t.orderStatus.status).lower()
            if status and s != status.lower():
                continue
            results.append({
                "order_id":   t.order.orderId,
                "symbol":     t.contract.symbol,
                "side":       t.order.action,
                "quantity":   t.order.totalQuantity,
                "order_type": t.order.orderType,
                "limit_price": t.order.lmtPrice if t.order.lmtPrice else None,
                "status":     str(t.orderStatus.status),
                "filled":     t.orderStatus.filled,
                "avg_fill":   t.orderStatus.avgFillPrice if t.orderStatus.avgFillPrice else None,
            })
        return {"success": True, "orders": results, "count": len(results)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
