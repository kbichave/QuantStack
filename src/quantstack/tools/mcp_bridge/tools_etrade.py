# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""eTrade tool classes wrapping MCP server calls."""

import json

from pydantic import BaseModel

from quantstack.crewai_compat import BaseTool

from ._bridge import _run_async, get_bridge
from ._schemas import (
    BalanceInput,
    OptionChainInput,
    PlaceOrderInput,
    PositionsInput,
    PreviewOrderInput,
    QuoteInput,
)


class GetQuoteTool(BaseTool):
    """Tool to get real-time quotes."""

    name: str = "get_quote"
    description: str = (
        "Get real-time quotes for one or more symbols. Returns bid/ask, last price, volume."
    )
    args_schema: type[BaseModel] = QuoteInput

    def _run(self, symbols: str) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade("get_quote", symbols=symbols)

        return json.dumps(_run_async(_exec()), indent=2)


class GetOptionChainsTool(BaseTool):
    """Tool to get option chains from eTrade."""

    name: str = "get_option_chains"
    description: str = (
        "Get option chain for a symbol with calls and puts, Greeks, and open interest."
    )
    args_schema: type[BaseModel] = OptionChainInput

    def _run(
        self,
        symbol: str,
        expiration_date: str | None = None,
        strike_price_near: float | None = None,
        no_of_strikes: int = 10,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "get_option_chains",
                symbol=symbol,
                expiration_date=expiration_date,
                strike_price_near=strike_price_near,
                no_of_strikes=no_of_strikes,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class PreviewOrderTool(BaseTool):
    """Tool to preview an order before placement."""

    name: str = "preview_order"
    description: str = (
        "Preview an order to see estimated costs. ALWAYS use before placing orders."
    )
    args_schema: type[BaseModel] = PreviewOrderInput

    def _run(
        self,
        account_id_key: str,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "LIMIT",
        limit_price: float | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "preview_order",
                account_id_key=account_id_key,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class PlaceOrderTool(BaseTool):
    """Tool to place an order."""

    name: str = "place_order"
    description: str = (
        "Place an order. ALWAYS preview first. This commits real money in production."
    )
    args_schema: type[BaseModel] = PlaceOrderInput

    def _run(
        self,
        account_id_key: str,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "LIMIT",
        limit_price: float | None = None,
        preview_id: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "place_order",
                account_id_key=account_id_key,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                preview_id=preview_id,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetPositionsTool(BaseTool):
    """Tool to get account positions."""

    name: str = "get_positions"
    description: str = "Get current positions for an account with P&L information."
    args_schema: type[BaseModel] = PositionsInput

    def _run(self, account_id_key: str, symbol: str | None = None) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "get_positions", account_id_key=account_id_key, symbol=symbol
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetAccountBalanceTool(BaseTool):
    """Tool to get account balance."""

    name: str = "get_account_balance"
    description: str = (
        "Get account balance including cash, buying power, and margin info."
    )
    args_schema: type[BaseModel] = BalanceInput

    def _run(self, account_id_key: str) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "get_account_balance", account_id_key=account_id_key
            )

        return json.dumps(_run_async(_exec()), indent=2)
