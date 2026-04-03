"""Data functions called directly by graph nodes."""

from typing import Any

from quantstack.tools.mcp_bridge._bridge import get_bridge


async def get_regime(symbol: str = "SPY") -> dict[str, Any]:
    """Get current market regime for a symbol.

    Called by load_context nodes to populate state.regime.
    """
    bridge = get_bridge()
    return await bridge.call_quantcore("get_regime", symbol=symbol)


async def get_portfolio_state() -> dict[str, Any]:
    """Get current portfolio state.

    Called by load_context nodes to populate state.portfolio_context.
    """
    bridge = get_bridge()
    return await bridge.call_quantcore("get_portfolio_state")
