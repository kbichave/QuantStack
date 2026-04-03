"""Risk gate functions called directly by graph nodes."""

from typing import Any

from quantstack.tools.mcp_bridge._bridge import get_bridge


async def validate_risk_gate(
    symbol: str,
    side: str,
    quantity: float,
    entry_price: float,
) -> dict[str, Any]:
    """Run the programmatic risk gate check.

    Called by the risk_gate conditional edge. Returns pass/fail with reasoning.
    This is the LAW — never bypass.
    """
    bridge = get_bridge()
    return await bridge.call_quantcore(
        "check_risk_limits",
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
    )
