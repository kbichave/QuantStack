"""Institutional accumulation tools for LangGraph agents."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
async def get_institutional_accumulation(
    symbol: Annotated[str, Field(description="Ticker symbol to assess for institutional accumulation, e.g. 'RDDT', 'NVDA', 'AAPL'")],
) -> str:
    """Retrieves institutional and smart-money accumulation signals for a stock ticker symbol. Use when evaluating whether large investors, hedge funds, or corporate insiders are building positions. Computes a composite accumulation score (0-1) from insider buying clusters, gamma exposure (GEX) support levels, implied volatility (IV) skew extremes, and 13F institutional ownership trends. Provides contrarian buy-the-dip confirmation beyond price action alone. Returns JSON with accumulation_score, component breakdowns, insider transaction details, GEX signal direction, IV skew context, and a buy/hold/avoid recommendation."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
