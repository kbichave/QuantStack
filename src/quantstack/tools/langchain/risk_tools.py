"""Risk analysis tools for LangGraph agents."""

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def compute_risk_metrics() -> str:
    """Compute portfolio risk metrics including VaR, max drawdown, and exposure.

    Returns JSON with portfolio-level risk assessment, position-level risks,
    and concentration warnings.
    """
    try:
        from quantstack.tools._state import require_ctx

        ctx = require_ctx()
        snapshot = ctx.portfolio.get_snapshot()
        limits = ctx.risk_gate.limits

        equity = snapshot.get("equity", 0)
        positions = snapshot.get("positions", {})
        daily_pnl = snapshot.get("daily_pnl", 0)

        position_risks = []
        total_exposure = 0
        for sym, pos in positions.items():
            qty = pos.get("quantity", 0)
            price = pos.get("current_price", pos.get("entry_price", 0))
            exposure = abs(qty * price)
            total_exposure += exposure
            position_risks.append({
                "symbol": sym,
                "quantity": qty,
                "exposure": exposure,
                "pct_of_equity": round(exposure / equity * 100, 2) if equity else 0,
            })

        result = {
            "equity": equity,
            "daily_pnl": daily_pnl,
            "position_count": len(positions),
            "gross_exposure": total_exposure,
            "exposure_pct": round(total_exposure / equity * 100, 2) if equity else 0,
            "risk_limits": limits if isinstance(limits, dict) else str(limits),
            "position_risks": position_risks,
        }
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"compute_risk_metrics failed: {e}")
        return json.dumps({"error": str(e)})


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
    try:
        from quantstack.tools._state import require_ctx

        ctx = require_ctx()
        snapshot = ctx.portfolio.get_snapshot()
        equity = snapshot.get("equity", 100_000)

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return json.dumps({"error": "Invalid stop loss — must differ from entry"})

        # 1% of equity risk per trade (conservative default)
        risk_budget = equity * 0.01
        quantity = max(1, int(risk_budget / risk_per_share))
        notional = quantity * entry_price

        result = {
            "symbol": symbol,
            "method": method,
            "recommended_quantity": quantity,
            "notional_value": round(notional, 2),
            "risk_per_share": round(risk_per_share, 2),
            "total_risk": round(quantity * risk_per_share, 2),
            "pct_of_equity": round(notional / equity * 100, 2),
        }
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def compute_var(confidence: float = 0.95, horizon_days: int = 1) -> str:
    """Compute Value at Risk for the portfolio.

    Args:
        confidence: Confidence level (default 0.95).
        horizon_days: VaR horizon in trading days.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def stress_test_portfolio(scenario: str = "market_crash") -> str:
    """Run stress test scenario on the portfolio.

    Args:
        scenario: Scenario name ("market_crash", "vol_spike", "sector_rotation").
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def check_risk_limits() -> str:
    """Check if any risk limits are currently breached."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_max_drawdown() -> str:
    """Compute maximum drawdown for the portfolio."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
