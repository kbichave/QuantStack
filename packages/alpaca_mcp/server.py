"""
alpaca_mcp server — FastMCP server exposing Alpaca brokerage tools.

Start with: ``alpaca-mcp`` (pyproject.toml script entry point)
Or:         ``python -m alpaca_mcp.server``

Tools:
    get_auth_status     Validate API key connectivity
    get_account         Account summary
    get_balance         Cash / buying power / portfolio value
    get_positions       Open positions with P&L
    get_quote           Real-time quotes (up to 50 symbols)
    get_bars            Historical OHLCV bars
    preview_order       Estimate cost without submitting
    place_order         Submit market or limit equity order
    cancel_order        Cancel an open order
    get_orders          Order history with optional status filter
    get_option_chains   Options chain snapshot (requires options data sub)

Auth (environment variables):
    ALPACA_API_KEY      required
    ALPACA_SECRET_KEY   required
    ALPACA_PAPER        true (default) | false
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastmcp import FastMCP
from loguru import logger

# ---------------------------------------------------------------------------
# Server context
# ---------------------------------------------------------------------------


class ServerContext:
    broker: Any = None  # AlpacaBrokerClient
    account_id: str = ""


_ctx = ServerContext()


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialise AlpacaBrokerClient on startup; log warning if auth fails."""
    logger.info("[alpaca_mcp] Starting…")
    try:
        from alpaca_mcp.client import AlpacaBrokerClient

        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        _ctx.broker = AlpacaBrokerClient(paper=paper)

        # Resolve the account_id once at startup
        accounts = _ctx.broker.get_accounts()
        _ctx.account_id = accounts[0].account_id if accounts else ""
        logger.info(f"[alpaca_mcp] Connected  account={_ctx.account_id}  paper={paper}")
    except Exception as exc:
        logger.warning(f"[alpaca_mcp] Startup error: {exc}. Tools will return errors.")

    yield

    logger.info("[alpaca_mcp] Shutdown")


mcp = FastMCP("alpaca_mcp", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_broker() -> Any:
    if _ctx.broker is None:
        raise RuntimeError(
            "Alpaca broker is not initialised. Check ALPACA_API_KEY / "
            "ALPACA_SECRET_KEY environment variables."
        )
    return _ctx.broker


def _dc_to_dict(obj) -> dict:
    """Dataclass → JSON-safe dict (datetime fields → ISO strings)."""
    d = asdict(obj)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_auth_status() -> dict:
    """Check whether the Alpaca API credentials are valid and the broker is reachable."""
    try:
        broker = _require_broker()
        ok = broker.check_auth()
        return {
            "success": ok,
            "paper": os.getenv("ALPACA_PAPER", "true").lower() == "true",
            "account_id": _ctx.account_id,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_account() -> dict:
    """Return Alpaca account summary including status and account type."""
    try:
        broker = _require_broker()
        accounts = broker.get_accounts()
        return {
            "success": True,
            "accounts": [_dc_to_dict(a) for a in accounts],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_balance() -> dict:
    """Return cash, buying power, and portfolio value."""
    try:
        broker = _require_broker()
        balance = broker.get_balance(_ctx.account_id)
        return {"success": True, "balance": _dc_to_dict(balance)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_positions() -> dict:
    """Return all open positions with quantity, entry price, and unrealised P&L."""
    try:
        broker = _require_broker()
        positions = broker.get_positions(_ctx.account_id)
        return {
            "success": True,
            "positions": [_dc_to_dict(p) for p in positions],
            "count": len(positions),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_quote(symbols: list[str]) -> dict:
    """Return real-time best-bid/offer quotes for up to 50 symbols.

    Args:
        symbols: List of ticker symbols, e.g. [\"SPY\", \"AAPL\"].
    """
    try:
        broker = _require_broker()
        quotes = broker.get_quote(symbols)
        return {
            "success": True,
            "quotes": [_dc_to_dict(q) for q in quotes],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_bars(
    symbol: str,
    timeframe: str = "1h",
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> dict:
    """Return historical OHLCV bars for a symbol.

    Args:
        symbol:    Ticker symbol.
        timeframe: Bar size: \"1m\", \"5m\", \"15m\", \"30m\", \"1h\", \"4h\", \"1d\", \"1w\".
        start:     ISO-8601 start datetime (e.g. \"2024-01-01\").
        end:       ISO-8601 end datetime.
        limit:     Maximum number of bars to return (default 200).
    """
    try:
        from quantcore.config.timeframes import Timeframe
        from quantcore.data.adapters.alpaca import AlpacaAdapter

        _TF_MAP = {
            "1m": Timeframe.M1,
            "5m": Timeframe.M5,
            "15m": Timeframe.M15,
            "30m": Timeframe.M30,
            "1h": Timeframe.H1,
            "4h": Timeframe.H4,
            "1d": Timeframe.D1,
            "1w": Timeframe.W1,
        }
        tf = _TF_MAP.get(timeframe.lower())
        if tf is None:
            return {"success": False, "error": f"Unknown timeframe '{timeframe}'"}

        broker = _require_broker()
        adapter = AlpacaAdapter(
            api_key=broker._api_key,
            secret_key=broker._secret_key,
        )
        start_dt = datetime.fromisoformat(start) if start else None
        end_dt = datetime.fromisoformat(end) if end else None
        df = adapter.fetch_ohlcv(symbol, tf, start_date=start_dt, end_date=end_dt)
        df = df.tail(limit)

        records = []
        for ts, row in df.iterrows():
            rec = {"timestamp": ts.isoformat()}
            rec.update({k: float(v) for k, v in row.items() if v is not None})
            records.append(rec)

        return {"success": True, "symbol": symbol, "timeframe": timeframe, "bars": records}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def preview_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: float | None = None,
) -> dict:
    """Estimate cost and commission for an order without submitting it.

    Args:
        symbol:      Ticker symbol.
        side:        \"buy\" or \"sell\".
        quantity:    Number of shares.
        order_type:  \"market\" or \"limit\".
        limit_price: Required for limit orders.
    """
    try:
        from quantcore.execution.unified_models import UnifiedOrder

        broker = _require_broker()
        order = UnifiedOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
        )
        preview = broker.preview_order(_ctx.account_id, order)
        return {"success": True, "preview": _dc_to_dict(preview)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def place_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: float | None = None,
    time_in_force: str = "day",
    extended_hours: bool = False,
) -> dict:
    """Submit an equity order to Alpaca.

    Args:
        symbol:         Ticker symbol.
        side:           \"buy\" or \"sell\".
        quantity:       Number of shares.
        order_type:     \"market\" | \"limit\" | \"stop\" | \"stop_limit\".
        limit_price:    Required for limit / stop_limit orders.
        time_in_force:  \"day\" | \"gtc\" | \"ioc\" | \"fok\".
        extended_hours: True to allow pre/post-market execution (limit orders only).
    """
    try:
        from quantcore.execution.unified_models import UnifiedOrder

        broker = _require_broker()
        order = UnifiedOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            extended_hours=extended_hours,
        )
        result = broker.place_order(_ctx.account_id, order)
        return {"success": True, "order": _dc_to_dict(result)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def cancel_order(order_id: str) -> dict:
    """Cancel an open order by its Alpaca order ID.

    Args:
        order_id: Alpaca UUID for the order.
    """
    try:
        broker = _require_broker()
        ok = broker.cancel_order(_ctx.account_id, order_id)
        return {"success": ok, "order_id": order_id}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_orders(
    status: str | None = None,
    limit: int = 50,
) -> dict:
    """Return order history.

    Args:
        status: Filter by status — \"open\", \"filled\", \"cancelled\", or None for all.
        limit:  Maximum number of orders to return.
    """
    try:
        broker = _require_broker()
        orders = broker.get_orders(_ctx.account_id, status=status)
        return {
            "success": True,
            "orders": [_dc_to_dict(o) for o in orders[:limit]],
            "count": len(orders),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def get_option_chains(
    symbol: str,
    expiry: str | None = None,
) -> dict:
    """Return an options chain snapshot for a symbol.

    Note: Requires an Alpaca Options Data subscription.

    Args:
        symbol: Underlying ticker symbol.
        expiry: Target expiry date in YYYY-MM-DD format (optional).
    """
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest

        broker = _require_broker()
        cli = OptionHistoricalDataClient(broker._api_key, broker._secret_key)
        req_kwargs = {"underlying_symbol": symbol}
        if expiry:
            req_kwargs["expiration_date"] = expiry
        req = OptionChainRequest(**req_kwargs)
        chain = cli.get_option_chain(req)
        # Normalise to a list of dicts
        contracts = []
        for sym, snap in chain.items():
            contracts.append(
                {
                    "symbol": sym,
                    "bid": float(snap.latest_quote.bid_price) if snap.latest_quote else None,
                    "ask": float(snap.latest_quote.ask_price) if snap.latest_quote else None,
                }
            )
        return {"success": True, "underlying": symbol, "contracts": contracts}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
