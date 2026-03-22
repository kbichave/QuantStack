# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
eTrade MCP Server - FastMCP Implementation.

Exposes eTrade trading functionality as MCP tools.  The auth/client/model
layer lives in ``quant_pod.tools.etrade``; this module is the pure MCP
transport wrapper around it.

Tools:
    etrade_authorize        Start or complete OAuth 1.0a flow
    etrade_refresh_token    Renew access token before midnight expiry
    get_auth_status         Check authentication state
    get_accounts            List accounts
    get_account_balance     Cash / margin / buying-power summary
    get_positions           Open positions with P&L
    get_quote               Real-time quotes (up to 25 symbols)
    get_option_expiry_dates Available expirations for a symbol
    get_option_chains       Full option chain with Greeks
    preview_order           Estimate cost — always call before place_order
    place_order             Submit equity or single-leg option order
    place_spread_order      Submit multi-leg spread order
    cancel_order            Cancel an open order
    get_orders              Order history with optional status filter

Auth (environment variables):
    ETRADE_CONSUMER_KEY     required
    ETRADE_CONSUMER_SECRET  required
    ETRADE_SANDBOX          true (default) | false

Usage:
    etrade-mcp                     # via pyproject.toml script entry
    python -m etrade_mcp.server
"""

import argparse
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastmcp import FastMCP
from loguru import logger
from quantstack.tools.etrade.auth import ETradeAuthManager
from quantstack.tools.etrade.client import ETradeClient
from quantstack.tools.etrade.models import (
    MarketSession,
    OptionType,
    OrderAction,
    OrderDuration,
    OrderLeg,
    OrderRequest,
    OrderType,
    SecurityType,
    SpreadLeg,
    SpreadOrderRequest,
)

# =============================================================================
# SERVER CONTEXT
# =============================================================================


@dataclass
class ServerContext:
    """Shared state for the MCP server process."""

    auth_manager: ETradeAuthManager
    client: ETradeClient | None = None
    sandbox_mode: bool = True


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialise and clean up server resources."""
    logger.info("eTrade MCP Server starting...")

    consumer_key = os.getenv("ETRADE_CONSUMER_KEY")
    consumer_secret = os.getenv("ETRADE_CONSUMER_SECRET")

    if not consumer_key or not consumer_secret:
        logger.warning(
            "ETRADE_CONSUMER_KEY and ETRADE_CONSUMER_SECRET not set. "
            "Authentication tools will not work until configured."
        )

    sandbox = os.getenv("ETRADE_SANDBOX", "true").lower() in ("true", "1", "yes")
    auth_manager = ETradeAuthManager(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        sandbox=sandbox,
    )

    client = None
    if auth_manager.is_authenticated():
        client = ETradeClient(auth_manager)
        logger.info("eTrade client initialised with existing tokens")

    ctx = ServerContext(auth_manager=auth_manager, client=client, sandbox_mode=sandbox)
    server.context = ctx
    logger.info(f"eTrade MCP Server ready (sandbox={sandbox})")

    yield

    logger.info("eTrade MCP Server stopped")


# =============================================================================
# FASTMCP SERVER
# =============================================================================

mcp = FastMCP(
    name="eTrade Trading Platform",
    instructions=(
        "eTrade trading platform with OAuth authentication, account management, "
        "market data, and order execution. Use sandbox mode for testing. "
        "IMPORTANT: Always call preview_order before place_order."
    ),
    lifespan=lifespan,
)


# =============================================================================
# HELPERS
# =============================================================================


def _get_context() -> ServerContext:
    ctx = mcp.context
    if not ctx:
        raise ValueError("Server not initialised")
    return ctx


def _ensure_client() -> ETradeClient:
    ctx = _get_context()
    if not ctx.auth_manager.is_authenticated():
        raise ValueError(
            "Not authenticated. Call etrade_authorize first to complete OAuth flow."
        )
    if not ctx.client:
        ctx.client = ETradeClient(ctx.auth_manager)
    return ctx.client


def _serialize(obj: Any) -> Any:
    """Recursively serialise Pydantic models / datetimes to JSON-safe types."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(by_alias=True)
    if hasattr(obj, "dict"):
        return obj.dict(by_alias=True)
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# =============================================================================
# AUTH TOOLS
# =============================================================================


@mcp.tool()
async def etrade_authorize(verifier_code: str | None = None) -> dict[str, Any]:
    """
    Start or complete eTrade OAuth authorisation.

    OAuth is a three-step process:
    1. Call without verifier_code to get the authorisation URL.
    2. Visit the URL in a browser and authorise the application.
    3. Call again with the verifier_code shown on the eTrade page.

    Args:
        verifier_code: Verifier from the eTrade authorisation page.
                       Omit to get the URL for step 1.
    """
    ctx = _get_context()
    auth = ctx.auth_manager

    try:
        if verifier_code:
            success = auth.complete_authorization(verifier_code)
            if success:
                ctx.client = ETradeClient(auth)
                return {
                    "success": True,
                    "message": "Authorisation successful. Trading tools are now available.",
                    "status": _serialize(auth.get_auth_status()),
                }
            return {
                "success": False,
                "error": "Authorisation failed. Please try again.",
            }

        auth_url = auth.get_authorization_url()
        return {
            "success": True,
            "message": (
                "Visit the authorisation URL below, approve the application, "
                "then call this tool again with the verifier_code."
            ),
            "auth_url": auth_url,
            "sandbox_mode": ctx.sandbox_mode,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def etrade_refresh_token() -> dict[str, Any]:
    """
    Refresh the eTrade access token.

    Access tokens expire at midnight Eastern. Call this periodically
    to extend the session without re-authorising.
    """
    ctx = _get_context()
    auth = ctx.auth_manager

    try:
        if not auth.is_authenticated():
            return {
                "success": False,
                "error": "Not authenticated. Call etrade_authorize first.",
            }

        success = auth.refresh_token()
        return {
            "success": success,
            "status": _serialize(auth.get_auth_status()),
            "message": "Token refreshed" if success else "Token refresh failed",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_auth_status() -> dict[str, Any]:
    """Get current eTrade authentication status and token expiry."""
    ctx = _get_context()
    auth = ctx.auth_manager
    return {
        "success": True,
        "status": _serialize(auth.get_auth_status()),
        "sandbox_mode": ctx.sandbox_mode,
        "consumer_key_set": bool(auth.consumer_key),
    }


# =============================================================================
# ACCOUNT TOOLS
# =============================================================================


@mcp.tool()
async def get_accounts() -> dict[str, Any]:
    """
    List eTrade accounts for the authenticated user.

    Returns account IDs, names, and types. Use accountIdKey for
    subsequent account operations.
    """
    try:
        client = _ensure_client()
        accounts = client.get_accounts()
        return {
            "success": True,
            "count": len(accounts),
            "accounts": _serialize(accounts),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_account_balance(account_id_key: str) -> dict[str, Any]:
    """
    Get account balance and buying power.

    Args:
        account_id_key: accountIdKey from get_accounts
    """
    try:
        client = _ensure_client()
        balance = client.get_account_balance(account_id_key)
        return {"success": True, "balance": _serialize(balance)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_positions(
    account_id_key: str,
    symbol: str | None = None,
) -> dict[str, Any]:
    """
    Get open positions for an account.

    Args:
        account_id_key: accountIdKey from get_accounts
        symbol: Optional symbol to filter positions
    """
    try:
        client = _ensure_client()
        positions = client.get_positions(account_id_key, symbol=symbol)
        return {
            "success": True,
            "count": len(positions),
            "positions": _serialize(positions),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# MARKET DATA TOOLS
# =============================================================================


@mcp.tool()
async def get_quote(symbols: str) -> dict[str, Any]:
    """
    Get real-time quotes for one or more symbols.

    Args:
        symbols: Comma-separated symbols (max 25), e.g. "AAPL,MSFT,SPY"
    """
    try:
        client = _ensure_client()
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        quotes = client.get_quote(symbol_list)
        return {"success": True, "count": len(quotes), "quotes": _serialize(quotes)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_option_expiry_dates(symbol: str) -> dict[str, Any]:
    """
    Get available option expiration dates for a symbol.

    Args:
        symbol: Underlying symbol, e.g. "SPY"
    """
    try:
        client = _ensure_client()
        expirations = client.get_option_expiry_dates(symbol)
        return {
            "success": True,
            "symbol": symbol.upper(),
            "count": len(expirations),
            "expirations": _serialize(expirations),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_option_chains(
    symbol: str,
    expiration_date: str | None = None,
    strike_price_near: float | None = None,
    no_of_strikes: int = 10,
    option_type: str | None = None,
) -> dict[str, Any]:
    """
    Get option chain for a symbol with calls, puts, and Greeks.

    Args:
        symbol: Underlying symbol, e.g. "SPY"
        expiration_date: YYYY-MM-DD (omit for nearest expiry)
        strike_price_near: Centre strikes around this price (omit for ATM)
        no_of_strikes: Number of strikes (default 10)
        option_type: "CALL" or "PUT" (omit for both)
    """
    try:
        client = _ensure_client()
        chain = client.get_option_chains(
            symbol=symbol.upper(),
            expiration_date=expiration_date,
            strike_price_near=strike_price_near,
            no_of_strikes=no_of_strikes,
            option_type=option_type.upper() if option_type else None,
        )
        return {"success": True, "chain": _serialize(chain)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# ORDER TOOLS
# =============================================================================


@mcp.tool()
async def preview_order(
    account_id_key: str,
    symbol: str,
    action: str,
    quantity: int,
    order_type: str = "LIMIT",
    limit_price: float | None = None,
    stop_price: float | None = None,
    duration: str = "DAY",
    security_type: str = "EQ",
    option_type: str | None = None,
    strike_price: float | None = None,
    expiration_date: str | None = None,
) -> dict[str, Any]:
    """
    Preview an order before placement — always call this first.

    Args:
        account_id_key: accountIdKey from get_accounts
        symbol: Stock or option symbol
        action: BUY | SELL | BUY_TO_OPEN | BUY_TO_CLOSE | SELL_TO_OPEN | SELL_TO_CLOSE
        quantity: Shares or contracts
        order_type: MARKET | LIMIT | STOP | STOP_LIMIT (default LIMIT)
        limit_price: Required for LIMIT / STOP_LIMIT orders
        stop_price: Required for STOP / STOP_LIMIT orders
        duration: DAY | GOOD_TILL_CANCEL | IMMEDIATE_OR_CANCEL (default DAY)
        security_type: EQ for stocks, OPTN for options (default EQ)
        option_type: CALL | PUT (required for options)
        strike_price: Strike price (required for options)
        expiration_date: YYYY-MM-DD (required for options)
    """
    try:
        client = _ensure_client()
        leg = OrderLeg(
            symbol=symbol.upper(),
            securityType=SecurityType(security_type),
            orderAction=OrderAction(action.upper()),
            quantity=quantity,
            optionType=OptionType(option_type.upper()) if option_type else None,
            strikePrice=strike_price,
            expirationDate=expiration_date,
        )
        order_req = OrderRequest(
            accountIdKey=account_id_key,
            orderType=OrderType(order_type.upper()),
            priceType=order_type.upper(),
            limitPrice=limit_price,
            stopPrice=stop_price,
            orderTerm=(
                OrderDuration(duration.upper())
                if duration != "DAY"
                else OrderDuration.DAY
            ),
            marketSession=MarketSession.REGULAR,
            legs=[leg],
        )
        preview = client.preview_order(account_id_key, order_req)
        return {
            "success": True,
            "preview": _serialize(preview),
            "order_details": {
                "symbol": symbol.upper(),
                "action": action.upper(),
                "quantity": quantity,
                "order_type": order_type.upper(),
                "limit_price": limit_price,
                "stop_price": stop_price,
            },
            "message": "Preview successful. Use preview_id to place the order.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def place_order(
    account_id_key: str,
    symbol: str,
    action: str,
    quantity: int,
    order_type: str = "LIMIT",
    limit_price: float | None = None,
    stop_price: float | None = None,
    duration: str = "DAY",
    security_type: str = "EQ",
    option_type: str | None = None,
    strike_price: float | None = None,
    expiration_date: str | None = None,
    preview_id: str | None = None,
) -> dict[str, Any]:
    """
    Place an order. Always call preview_order first.

    Args:
        account_id_key: accountIdKey from get_accounts
        symbol: Stock or option symbol
        action: BUY | SELL | BUY_TO_OPEN | BUY_TO_CLOSE | SELL_TO_OPEN | SELL_TO_CLOSE
        quantity: Shares or contracts
        order_type: MARKET | LIMIT | STOP | STOP_LIMIT (default LIMIT)
        limit_price: Required for LIMIT / STOP_LIMIT orders
        stop_price: Required for STOP / STOP_LIMIT orders
        duration: DAY | GOOD_TILL_CANCEL | IMMEDIATE_OR_CANCEL (default DAY)
        security_type: EQ for stocks, OPTN for options (default EQ)
        option_type: CALL | PUT (required for options)
        strike_price: Strike price (required for options)
        expiration_date: YYYY-MM-DD (required for options)
        preview_id: preview_id from preview_order (strongly recommended)
    """
    try:
        client = _ensure_client()
        ctx = _get_context()

        if not ctx.sandbox_mode and not preview_id:
            logger.warning("Placing PRODUCTION order without preview_id!")

        leg = OrderLeg(
            symbol=symbol.upper(),
            securityType=SecurityType(security_type),
            orderAction=OrderAction(action.upper()),
            quantity=quantity,
            optionType=OptionType(option_type.upper()) if option_type else None,
            strikePrice=strike_price,
            expirationDate=expiration_date,
        )
        order_req = OrderRequest(
            accountIdKey=account_id_key,
            orderType=OrderType(order_type.upper()),
            priceType=order_type.upper(),
            limitPrice=limit_price,
            stopPrice=stop_price,
            orderTerm=(
                OrderDuration(duration.upper())
                if duration != "DAY"
                else OrderDuration.DAY
            ),
            marketSession=MarketSession.REGULAR,
            legs=[leg],
        )
        order = client.place_order(account_id_key, order_req, preview_id=preview_id)
        return {
            "success": True,
            "order": _serialize(order),
            "message": f"Order placed. Order ID: {order.order_id}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def place_spread_order(
    account_id_key: str,
    underlying_symbol: str,
    legs: list[dict[str, Any]],
    order_type: str = "LIMIT",
    limit_price: float | None = None,
    duration: str = "DAY",
) -> dict[str, Any]:
    """
    Place a multi-leg option spread (vertical, iron condor, calendar, etc.).

    Args:
        account_id_key: accountIdKey from get_accounts
        underlying_symbol: Underlying stock symbol, e.g. "SPY"
        legs: List of dicts, each with keys:
              option_type (CALL|PUT), strike_price, expiration_date (YYYY-MM-DD),
              action (BUY_TO_OPEN|SELL_TO_OPEN|…), quantity
        order_type: LIMIT | MARKET (default LIMIT)
        limit_price: Net credit (negative) or debit (positive)
        duration: DAY | GOOD_TILL_CANCEL (default DAY)

    Example — bull put spread:
        legs=[
            {"option_type": "PUT", "strike_price": 440, "expiration_date": "2024-03-15",
             "action": "SELL_TO_OPEN", "quantity": 1},
            {"option_type": "PUT", "strike_price": 435, "expiration_date": "2024-03-15",
             "action": "BUY_TO_OPEN", "quantity": 1},
        ],
        limit_price=-0.50
    """
    try:
        client = _ensure_client()
        spread_legs = [
            SpreadLeg(
                symbol=underlying_symbol.upper(),
                optionType=OptionType(leg["option_type"].upper()),
                strikePrice=leg["strike_price"],
                expirationDate=leg["expiration_date"],
                orderAction=OrderAction(leg["action"].upper()),
                quantity=leg["quantity"],
            )
            for leg in legs
        ]
        spread_req = SpreadOrderRequest(
            accountIdKey=account_id_key,
            underlyingSymbol=underlying_symbol.upper(),
            orderType=OrderType(order_type.upper()),
            limitPrice=limit_price,
            orderTerm=(
                OrderDuration(duration.upper())
                if duration != "DAY"
                else OrderDuration.DAY
            ),
            marketSession=MarketSession.REGULAR,
            legs=spread_legs,
        )
        order = client.place_spread_order(account_id_key, spread_req)
        return {
            "success": True,
            "order": _serialize(order),
            "message": f"Spread order placed. Order ID: {order.order_id}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def cancel_order(account_id_key: str, order_id: str) -> dict[str, Any]:
    """
    Cancel an open order.

    Args:
        account_id_key: accountIdKey from get_accounts
        order_id: Order ID to cancel
    """
    try:
        client = _ensure_client()
        success = client.cancel_order(account_id_key, order_id)
        return {
            "success": success,
            "message": (
                f"Order {order_id} cancelled" if success else "Cancellation failed"
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_orders(
    account_id_key: str,
    status: str | None = None,
) -> dict[str, Any]:
    """
    Get orders for an account.

    Args:
        account_id_key: accountIdKey from get_accounts
        status: OPEN | EXECUTED | CANCELLED (omit for all)
    """
    try:
        client = _ensure_client()
        orders = client.get_orders(account_id_key, status=status)
        return {"success": True, "count": len(orders), "orders": _serialize(orders)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    """Run the eTrade MCP server."""
    parser = argparse.ArgumentParser(description="eTrade MCP Server")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production environment (CAUTION: real money!)",
    )
    args = parser.parse_args()

    if args.production:
        os.environ["ETRADE_SANDBOX"] = "false"
        logger.warning("PRODUCTION MODE — real-money transactions enabled")
    else:
        os.environ.setdefault("ETRADE_SANDBOX", "true")
        logger.info("Sandbox mode — paper trading")

    mcp.run()


if __name__ == "__main__":
    main()
