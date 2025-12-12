# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
eTrade MCP Server - FastMCP Implementation.

Exposes eTrade trading functionality as MCP tools:
- OAuth authentication (authorize, refresh)
- Account management (list, balance, positions)
- Market data (quotes, option chains, expiry dates)
- Order management (preview, place, spread, cancel)

Usage:
    python -m etrade_mcp.server
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from etrade_mcp.auth import ETradeAuthManager
from etrade_mcp.client import ETradeClient
from etrade_mcp.models import (
    OrderAction,
    OrderDuration,
    OrderLeg,
    OrderRequest,
    OrderType,
    OptionType,
    SecurityType,
    SpreadLeg,
    SpreadOrderRequest,
    MarketSession,
)


# =============================================================================
# MCP SERVER INITIALIZATION
# =============================================================================


@dataclass
class ServerContext:
    """Shared context for MCP server."""

    auth_manager: ETradeAuthManager
    client: Optional[ETradeClient] = None
    sandbox_mode: bool = True


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize and cleanup server resources."""
    logger.info("eTrade MCP Server starting...")

    # Check for required environment variables
    consumer_key = os.getenv("ETRADE_CONSUMER_KEY")
    consumer_secret = os.getenv("ETRADE_CONSUMER_SECRET")

    if not consumer_key or not consumer_secret:
        logger.warning(
            "ETRADE_CONSUMER_KEY and ETRADE_CONSUMER_SECRET not set. "
            "Authentication tools will not work until configured."
        )

    # Initialize auth manager
    sandbox = os.getenv("ETRADE_SANDBOX", "true").lower() in ("true", "1", "yes")
    auth_manager = ETradeAuthManager(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        sandbox=sandbox,
    )

    # Initialize client if authenticated
    client = None
    if auth_manager.is_authenticated():
        client = ETradeClient(auth_manager)
        logger.info("eTrade client initialized with existing tokens")

    ctx = ServerContext(
        auth_manager=auth_manager,
        client=client,
        sandbox_mode=sandbox,
    )

    server.context = ctx
    logger.info(f"eTrade MCP Server initialized (sandbox={sandbox})")

    yield

    logger.info("eTrade MCP Server stopped")


# Create the FastMCP server
mcp = FastMCP(
    name="eTrade Trading Platform",
    instructions=(
        "eTrade trading platform with OAuth authentication, account management, "
        "market data, and order execution. Use sandbox mode for testing. "
        "IMPORTANT: Always use preview_order before place_order to verify order details."
    ),
    lifespan=lifespan,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_context() -> ServerContext:
    """Get server context with validation."""
    ctx = mcp.context
    if not ctx:
        raise ValueError("Server not initialized")
    return ctx


def _ensure_client() -> ETradeClient:
    """Get client, raising error if not authenticated."""
    ctx = _get_context()

    if not ctx.auth_manager.is_authenticated():
        raise ValueError(
            "Not authenticated. Call etrade_authorize first to complete OAuth flow."
        )

    if not ctx.client:
        ctx.client = ETradeClient(ctx.auth_manager)

    return ctx.client


def _serialize_pydantic(obj: Any) -> Any:
    """Serialize Pydantic models and other types to JSON-compatible format."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(by_alias=True)
    if hasattr(obj, "dict"):
        return obj.dict(by_alias=True)
    if isinstance(obj, list):
        return [_serialize_pydantic(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_pydantic(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# =============================================================================
# AUTH TOOLS (2)
# =============================================================================


@mcp.tool()
async def etrade_authorize(verifier_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Start or complete eTrade OAuth authorization.

    OAuth is a three-step process:
    1. Call without verifier_code to get authorization URL
    2. Visit the URL in a browser and authorize the application
    3. Call again with the verifier_code from the authorization page

    Args:
        verifier_code: The verifier code from eTrade authorization page.
                      If None, returns the authorization URL to visit.

    Returns:
        Dictionary with auth_url (step 1) or authentication status (step 3)

    Example:
        # Step 1: Get auth URL
        result = etrade_authorize()
        # Visit result["auth_url"] in browser

        # Step 2: Complete with verifier from browser
        result = etrade_authorize(verifier_code="ABC123")
    """
    ctx = _get_context()
    auth = ctx.auth_manager

    try:
        if verifier_code:
            # Step 3: Complete authorization with verifier
            success = auth.complete_authorization(verifier_code)

            if success:
                # Initialize client
                ctx.client = ETradeClient(auth)

                return {
                    "success": True,
                    "message": "Authorization successful! You can now use trading tools.",
                    "status": _serialize_pydantic(auth.get_auth_status()),
                }
            else:
                return {
                    "success": False,
                    "error": "Authorization failed. Please try again.",
                }
        else:
            # Step 1: Get authorization URL
            auth_url = auth.get_authorization_url()

            return {
                "success": True,
                "message": (
                    "Visit the authorization URL below and authorize the application. "
                    "Then copy the verifier code and call this tool again with the verifier_code."
                ),
                "auth_url": auth_url,
                "sandbox_mode": ctx.sandbox_mode,
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def etrade_refresh_token() -> Dict[str, Any]:
    """
    Refresh eTrade access token to extend session.

    Access tokens expire at midnight Eastern time. Call this tool
    periodically to keep the session alive without re-authorizing.

    Returns:
        Dictionary with refresh status
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
            "status": _serialize_pydantic(auth.get_auth_status()),
            "message": (
                "Token refreshed successfully" if success else "Token refresh failed"
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# ACCOUNT TOOLS (3)
# =============================================================================


@mcp.tool()
async def get_accounts() -> Dict[str, Any]:
    """
    Get list of eTrade accounts for the authenticated user.

    Returns account IDs, names, types, and descriptions.
    Use the accountIdKey for subsequent account operations.

    Returns:
        Dictionary with list of accounts
    """
    try:
        client = _ensure_client()
        accounts = client.get_accounts()

        return {
            "success": True,
            "count": len(accounts),
            "accounts": [_serialize_pydantic(acc) for acc in accounts],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_account_balance(account_id_key: str) -> Dict[str, Any]:
    """
    Get account balance and buying power.

    Returns cash balances, margin buying power, and option buying power.

    Args:
        account_id_key: The accountIdKey from get_accounts

    Returns:
        Dictionary with balance details
    """
    try:
        client = _ensure_client()
        balance = client.get_account_balance(account_id_key)

        return {
            "success": True,
            "balance": _serialize_pydantic(balance),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_positions(
    account_id_key: str,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get positions for an account.

    Returns all positions with cost basis, market value, and P&L.
    Optionally filter by symbol.

    Args:
        account_id_key: The accountIdKey from get_accounts
        symbol: Optional symbol to filter positions

    Returns:
        Dictionary with list of positions
    """
    try:
        client = _ensure_client()
        positions = client.get_positions(account_id_key, symbol=symbol)

        return {
            "success": True,
            "count": len(positions),
            "positions": [_serialize_pydantic(pos) for pos in positions],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# MARKET DATA TOOLS (3)
# =============================================================================


@mcp.tool()
async def get_quote(symbols: str) -> Dict[str, Any]:
    """
    Get real-time quotes for one or more symbols.

    Returns bid/ask, last price, volume, and other quote data.

    Args:
        symbols: Comma-separated list of symbols (max 25), e.g., "AAPL,MSFT,GOOGL"

    Returns:
        Dictionary with quote data for each symbol
    """
    try:
        client = _ensure_client()

        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        quotes = client.get_quote(symbol_list)

        return {
            "success": True,
            "count": len(quotes),
            "quotes": [_serialize_pydantic(q) for q in quotes],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_option_expiry_dates(symbol: str) -> Dict[str, Any]:
    """
    Get available option expiration dates for a symbol.

    Returns all available expiration dates with days to expiration.

    Args:
        symbol: Underlying symbol (e.g., "SPY", "AAPL")

    Returns:
        Dictionary with list of expiration dates
    """
    try:
        client = _ensure_client()
        expirations = client.get_option_expiry_dates(symbol)

        return {
            "success": True,
            "symbol": symbol.upper(),
            "count": len(expirations),
            "expirations": [_serialize_pydantic(exp) for exp in expirations],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_option_chains(
    symbol: str,
    expiration_date: Optional[str] = None,
    strike_price_near: Optional[float] = None,
    no_of_strikes: int = 10,
    option_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get option chain for a symbol.

    Returns calls and puts with bid/ask, Greeks, and open interest.

    Args:
        symbol: Underlying symbol (e.g., "SPY", "AAPL")
        expiration_date: Specific expiration in YYYY-MM-DD format (None for nearest)
        strike_price_near: Center strikes around this price (None for ATM)
        no_of_strikes: Number of strikes to return (default 10)
        option_type: Filter by "CALL" or "PUT" (None for both)

    Returns:
        Dictionary with calls and puts option chains

    Example:
        # Get SPY options near current price
        get_option_chains("SPY", strike_price_near=450, no_of_strikes=5)

        # Get specific expiration
        get_option_chains("AAPL", expiration_date="2024-03-15", option_type="CALL")
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

        return {
            "success": True,
            "chain": _serialize_pydantic(chain),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# ORDER TOOLS (4)
# =============================================================================


@mcp.tool()
async def preview_order(
    account_id_key: str,
    symbol: str,
    action: str,
    quantity: int,
    order_type: str = "LIMIT",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    duration: str = "DAY",
    security_type: str = "EQ",
    option_type: Optional[str] = None,
    strike_price: Optional[float] = None,
    expiration_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Preview an order before placement.

    ALWAYS preview orders before placing them to see estimated costs
    and verify order details.

    Args:
        account_id_key: The accountIdKey from get_accounts
        symbol: Stock/option symbol
        action: Order action - "BUY", "SELL", "BUY_TO_OPEN", "BUY_TO_CLOSE",
                "SELL_TO_OPEN", "SELL_TO_CLOSE"
        quantity: Number of shares/contracts
        order_type: "MARKET", "LIMIT", "STOP", "STOP_LIMIT"
        limit_price: Limit price (required for LIMIT and STOP_LIMIT)
        stop_price: Stop price (required for STOP and STOP_LIMIT)
        duration: "DAY", "GOOD_TILL_CANCEL", "IMMEDIATE_OR_CANCEL"
        security_type: "EQ" for stocks, "OPTN" for options
        option_type: "CALL" or "PUT" (required for options)
        strike_price: Strike price (required for options)
        expiration_date: Expiration in YYYY-MM-DD (required for options)

    Returns:
        Dictionary with preview_id and estimated costs

    Example:
        # Stock order
        preview_order("abc123", "AAPL", "BUY", 100, "LIMIT", limit_price=175.00)

        # Option order
        preview_order(
            "abc123", "AAPL", "BUY_TO_OPEN", 1,
            "LIMIT", limit_price=5.00,
            security_type="OPTN", option_type="CALL",
            strike_price=180, expiration_date="2024-03-15"
        )
    """
    try:
        client = _ensure_client()

        # Build order leg
        leg = OrderLeg(
            symbol=symbol.upper(),
            securityType=SecurityType(security_type),
            orderAction=OrderAction(action.upper()),
            quantity=quantity,
            optionType=OptionType(option_type.upper()) if option_type else None,
            strikePrice=strike_price,
            expirationDate=expiration_date,
        )

        # Build order request
        order_request = OrderRequest(
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

        preview = client.preview_order(account_id_key, order_request)

        return {
            "success": True,
            "preview": _serialize_pydantic(preview),
            "order_details": {
                "symbol": symbol.upper(),
                "action": action.upper(),
                "quantity": quantity,
                "order_type": order_type.upper(),
                "limit_price": limit_price,
                "stop_price": stop_price,
            },
            "message": "Order preview successful. Use the preview_id to place the order.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def place_order(
    account_id_key: str,
    symbol: str,
    action: str,
    quantity: int,
    order_type: str = "LIMIT",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    duration: str = "DAY",
    security_type: str = "EQ",
    option_type: Optional[str] = None,
    strike_price: Optional[float] = None,
    expiration_date: Optional[str] = None,
    preview_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Place an order.

    IMPORTANT: Always call preview_order first to verify order details!

    Args:
        account_id_key: The accountIdKey from get_accounts
        symbol: Stock/option symbol
        action: Order action - "BUY", "SELL", "BUY_TO_OPEN", "BUY_TO_CLOSE",
                "SELL_TO_OPEN", "SELL_TO_CLOSE"
        quantity: Number of shares/contracts
        order_type: "MARKET", "LIMIT", "STOP", "STOP_LIMIT"
        limit_price: Limit price (required for LIMIT and STOP_LIMIT)
        stop_price: Stop price (required for STOP and STOP_LIMIT)
        duration: "DAY", "GOOD_TILL_CANCEL", "IMMEDIATE_OR_CANCEL"
        security_type: "EQ" for stocks, "OPTN" for options
        option_type: "CALL" or "PUT" (required for options)
        strike_price: Strike price (required for options)
        expiration_date: Expiration in YYYY-MM-DD (required for options)
        preview_id: Preview ID from preview_order (recommended)

    Returns:
        Dictionary with order_id and order details

    Example:
        # First preview
        preview = preview_order("abc123", "AAPL", "BUY", 100, "LIMIT", 175.00)

        # Then place with preview_id
        place_order(
            "abc123", "AAPL", "BUY", 100, "LIMIT", 175.00,
            preview_id=preview["preview"]["previewId"]
        )
    """
    try:
        client = _ensure_client()
        ctx = _get_context()

        # Safety warning for production
        if not ctx.sandbox_mode and not preview_id:
            logger.warning("Placing order in PRODUCTION without preview_id!")

        # Build order leg
        leg = OrderLeg(
            symbol=symbol.upper(),
            securityType=SecurityType(security_type),
            orderAction=OrderAction(action.upper()),
            quantity=quantity,
            optionType=OptionType(option_type.upper()) if option_type else None,
            strikePrice=strike_price,
            expirationDate=expiration_date,
        )

        # Build order request
        order_request = OrderRequest(
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

        order = client.place_order(account_id_key, order_request, preview_id=preview_id)

        return {
            "success": True,
            "order": _serialize_pydantic(order),
            "message": f"Order placed successfully. Order ID: {order.order_id}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def place_spread_order(
    account_id_key: str,
    underlying_symbol: str,
    legs: List[Dict[str, Any]],
    order_type: str = "LIMIT",
    limit_price: Optional[float] = None,
    duration: str = "DAY",
) -> Dict[str, Any]:
    """
    Place a multi-leg option spread order.

    Supports vertical spreads, iron condors, butterflies, calendars, etc.

    Args:
        account_id_key: The accountIdKey from get_accounts
        underlying_symbol: Underlying stock symbol (e.g., "SPY")
        legs: List of leg dictionaries, each with:
              - option_type: "CALL" or "PUT"
              - strike_price: Strike price
              - expiration_date: YYYY-MM-DD format
              - action: "BUY_TO_OPEN", "SELL_TO_OPEN", etc.
              - quantity: Number of contracts
        order_type: "LIMIT" or "MARKET"
        limit_price: Net credit (negative) or debit (positive)
        duration: "DAY" or "GOOD_TILL_CANCEL"

    Returns:
        Dictionary with order details

    Example - Bull Put Spread:
        place_spread_order(
            "abc123",
            "SPY",
            legs=[
                {"option_type": "PUT", "strike_price": 440, "expiration_date": "2024-03-15",
                 "action": "SELL_TO_OPEN", "quantity": 1},
                {"option_type": "PUT", "strike_price": 435, "expiration_date": "2024-03-15",
                 "action": "BUY_TO_OPEN", "quantity": 1},
            ],
            limit_price=-0.50  # $0.50 credit
        )
    """
    try:
        client = _ensure_client()

        # Build spread legs
        spread_legs = []
        for leg_data in legs:
            spread_legs.append(
                SpreadLeg(
                    symbol=underlying_symbol.upper(),
                    optionType=OptionType(leg_data["option_type"].upper()),
                    strikePrice=leg_data["strike_price"],
                    expirationDate=leg_data["expiration_date"],
                    orderAction=OrderAction(leg_data["action"].upper()),
                    quantity=leg_data["quantity"],
                )
            )

        # Build spread request
        spread_request = SpreadOrderRequest(
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

        order = client.place_spread_order(account_id_key, spread_request)

        return {
            "success": True,
            "order": _serialize_pydantic(order),
            "message": f"Spread order placed. Order ID: {order.order_id}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def cancel_order(
    account_id_key: str,
    order_id: str,
) -> Dict[str, Any]:
    """
    Cancel an open order.

    Args:
        account_id_key: The accountIdKey from get_accounts
        order_id: Order ID to cancel

    Returns:
        Dictionary with cancellation status
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
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# ADDITIONAL UTILITY TOOLS
# =============================================================================


@mcp.tool()
async def get_orders(
    account_id_key: str,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get orders for an account.

    Args:
        account_id_key: The accountIdKey from get_accounts
        status: Filter by status - "OPEN", "EXECUTED", "CANCELLED"

    Returns:
        Dictionary with list of orders
    """
    try:
        client = _ensure_client()
        orders = client.get_orders(account_id_key, status=status)

        return {
            "success": True,
            "count": len(orders),
            "orders": [_serialize_pydantic(order) for order in orders],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_auth_status() -> Dict[str, Any]:
    """
    Get current eTrade authentication status.

    Returns whether authenticated, token expiration, and environment info.

    Returns:
        Dictionary with authentication status
    """
    ctx = _get_context()
    auth = ctx.auth_manager

    return {
        "success": True,
        "status": _serialize_pydantic(auth.get_auth_status()),
        "sandbox_mode": ctx.sandbox_mode,
        "consumer_key_set": bool(auth.consumer_key),
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Run the eTrade MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="eTrade MCP Server")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        default=True,
        help="Use sandbox environment (default: True)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production environment (CAUTION: Real money!)",
    )

    args = parser.parse_args()

    # Set environment
    if args.production:
        os.environ["ETRADE_SANDBOX"] = "false"
        logger.warning("⚠️  PRODUCTION MODE - Real money transactions!")
    else:
        os.environ["ETRADE_SANDBOX"] = "true"
        logger.info("Sandbox mode - Paper trading")

    # Run server
    mcp.run()


if __name__ == "__main__":
    main()
