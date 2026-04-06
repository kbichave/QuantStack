"""Risk analysis tools for LangGraph agents."""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

logger = logging.getLogger(__name__)


@tool
async def compute_risk_metrics() -> str:
    """Computes comprehensive portfolio risk metrics including Value at Risk (VaR), gross exposure, daily P&L, and per-position concentration analysis. Use when you need a full risk snapshot before entering new trades or during position monitoring. Returns JSON with portfolio-level equity, exposure percentage, risk limits, and per-position breakdown with percent-of-equity concentration warnings.
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
    symbol: Annotated[str, Field(description="Ticker symbol for the position to size, e.g. 'AAPL' or 'SPY'")],
    entry_price: Annotated[float, Field(description="Planned entry price for the trade")],
    stop_loss: Annotated[float, Field(description="Stop loss price level; must differ from entry_price to define risk per share")],
    method: Annotated[str, Field(description="Position sizing method: 'atr' for ATR-based sizing or 'kelly' for Kelly criterion optimal sizing")] = "atr",
) -> str:
    """Calculates the recommended position size (number of shares) for a trade based on portfolio equity, risk budget, and the distance between entry price and stop loss. Use when planning a new entry to ensure proper risk management and avoid over-concentration. Provides notional value, risk per share, total dollar risk, and percent-of-equity allocation. Supports ATR-based and Kelly criterion sizing methods.
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
async def compute_var(
    confidence: Annotated[float, Field(description="Confidence level for VaR calculation, e.g. 0.95 for 95% or 0.99 for 99%")] = 0.95,
    horizon_days: Annotated[int, Field(description="VaR time horizon in trading days, e.g. 1 for daily VaR or 10 for bi-weekly")] = 1,
) -> str:
    """Computes Value at Risk (VaR) for the current portfolio at a specified confidence level and time horizon. Use when you need to quantify potential downside loss, assess tail risk exposure, or report risk limits utilization. Returns the estimated maximum portfolio loss that will not be exceeded with the given probability over the specified holding period.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def stress_test_portfolio(
    scenario: Annotated[str, Field(description="Stress scenario name: 'market_crash' for broad selloff, 'vol_spike' for volatility surge, or 'sector_rotation' for sector-level rebalancing shock")] = "market_crash",
) -> str:
    """Runs a stress test simulation on the current portfolio under a predefined adverse scenario. Use when you need to evaluate portfolio resilience, estimate drawdown under extreme conditions, or validate hedging effectiveness. Returns projected P&L impact, position-level losses, and exposure changes under the selected stress scenario (market crash, volatility spike, sector rotation).
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def check_risk_limits() -> str:
    """Checks whether any portfolio risk limits are currently breached or near breach thresholds. Use when you need to verify compliance before placing new orders, during position monitoring, or after market moves. Returns the status of each risk limit (max exposure, concentration, drawdown, daily loss) with current utilization and breach flags."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_max_drawdown() -> str:
    """Computes the maximum drawdown (peak-to-trough decline) for the portfolio over the available equity history. Use when you need to assess worst-case historical loss, evaluate strategy risk profile, or compare drawdown against risk tolerance thresholds. Returns the max drawdown percentage, peak equity value, trough equity value, and the date range of the drawdown period."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
