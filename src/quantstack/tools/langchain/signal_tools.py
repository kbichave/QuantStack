"""Signal analysis tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def signal_brief(symbol: str) -> str:
    """Get a technical signal brief for a symbol including trend, momentum, and key levels.

    Returns JSON with technical, fundamental, momentum, and regime signals.
    Call this first when evaluating any symbol.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore("get_signal_brief", symbol=symbol)
    return json.dumps(result, default=str)


@tool
async def multi_signal_brief(symbols: list[str]) -> str:
    """Get signal briefs for multiple symbols in parallel.

    Returns JSON with per-symbol signal data. Use for watchlist scanning.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore("run_multi_signal_brief", symbols=symbols)
    return json.dumps(result, default=str)
