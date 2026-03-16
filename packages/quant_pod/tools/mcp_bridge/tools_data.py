# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Market data tool classes wrapping QuantCore MCP server calls."""

import json

from pydantic import BaseModel

from quant_pod.crewai_compat import BaseTool

from ._bridge import _run_async, get_bridge
from ._schemas import (
    EmptyInput,
    FetchMarketDataInput,
    LoadMarketDataInput,
    SymbolSnapshotInput,
)


class FetchMarketDataTool(BaseTool):
    """Tool to fetch OHLCV market data from Alpha Vantage."""

    name: str = "fetch_market_data"
    description: str = (
        "Fetch OHLCV market data for a symbol. Use for getting fresh data from Alpha Vantage API."
    )
    args_schema: type[BaseModel] = FetchMarketDataInput

    def _run(self, symbol: str, timeframe: str = "daily", outputsize: str = "compact") -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "fetch_market_data",
                symbol=symbol,
                timeframe=timeframe,
                outputsize=outputsize,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class LoadMarketDataTool(BaseTool):
    """Tool to load OHLCV data from local storage."""

    name: str = "load_market_data"
    description: str = "Load OHLCV data from local DuckDB storage. Faster than fetching from API."
    args_schema: type[BaseModel] = LoadMarketDataInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "load_market_data",
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ListStoredSymbolsTool(BaseTool):
    """Tool to list all symbols in local database."""

    name: str = "list_stored_symbols"
    description: str = (
        "List all symbols stored in the local database with their available timeframes."
    )
    args_schema: type[BaseModel] = EmptyInput

    def _run(self) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore("list_stored_symbols")

        return json.dumps(_run_async(_exec()), indent=2)


class GetSymbolSnapshotTool(BaseTool):
    """Tool to get comprehensive symbol snapshot."""

    name: str = "get_symbol_snapshot"
    description: str = (
        "Get a comprehensive snapshot of a symbol including price, indicators, and regime."
    )
    args_schema: type[BaseModel] = SymbolSnapshotInput

    def _run(self, symbol: str, end_date: str | None = None) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_symbol_snapshot", symbol=symbol, end_date=end_date
            )

        return json.dumps(_run_async(_exec()), indent=2)
