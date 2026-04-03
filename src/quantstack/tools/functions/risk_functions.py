"""Risk gate functions called directly by graph nodes."""

from typing import Any

from loguru import logger

from quantstack.tools._state import require_ctx, _serialize


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
    try:
        ctx = require_ctx()
        daily_volume = 1_000_000  # Default for paper mode

        verdict = ctx.risk_gate.check(
            symbol=symbol,
            side=side,
            quantity=int(quantity),
            current_price=entry_price,
            daily_volume=daily_volume,
        )

        if verdict.approved:
            return {
                "approved": True,
                "approved_quantity": verdict.approved_quantity or int(quantity),
                "violations": [],
            }
        else:
            violations = [v.description for v in verdict.violations]
            return {
                "approved": False,
                "approved_quantity": 0,
                "violations": violations,
                "error": f"Risk gate rejected: {'; '.join(violations)}",
            }
    except Exception as e:
        logger.error(f"validate_risk_gate({symbol}) failed: {e}")
        return {"approved": False, "error": str(e), "violations": [str(e)]}
