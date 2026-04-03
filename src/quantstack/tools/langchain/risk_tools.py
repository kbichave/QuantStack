"""Risk analysis tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def compute_risk_metrics() -> str:
    """Compute portfolio risk metrics including VaR, max drawdown, and exposure.

    Returns JSON with portfolio-level risk assessment, position-level risks,
    and concentration warnings.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore("get_risk_metrics")
    return json.dumps(result, default=str)


@tool
async def compute_position_size(
    symbol: str,
    entry_price: float,
    stop_loss: float,
    method: str = "atr",
) -> str:
    """Compute recommended position size using ATR or Kelly criterion.

    Args:
        symbol: Ticker symbol.
        entry_price: Planned entry price.
        stop_loss: Stop loss level.
        method: Sizing method ("atr" or "kelly").
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "compute_position_size",
        symbol=symbol,
        entry_price=entry_price,
        stop_loss=stop_loss,
        method=method,
    )
    return json.dumps(result, default=str)
