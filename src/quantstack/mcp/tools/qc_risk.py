# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP risk tools — position sizing, drawdown, VaR, stress testing, and limit checks.

Extracted from ``quantcore.mcp.server`` to keep tool modules focused.
All helpers come from ``quantcore.mcp._helpers``; the ``mcp`` singleton
is imported from ``quantcore.mcp.server``.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantstack.core.options.models import OptionsPosition, OptionType
from quantstack.core.options.pricing import black_scholes_price
from quantstack.core.risk.position_sizing import ATRPositionSizer
from quantstack.core.risk.stress_testing import STRESS_SCENARIOS
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain


# =============================================================================
# RISK TOOLS
# =============================================================================


@domain(Domain.RISK)
@mcp.tool()
async def compute_position_size(
    equity: float,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = 20.0,
    alignment_score: float = 1.0,
) -> dict[str, Any]:
    """
    Calculate position size using ATR-based risk management.

    Args:
        equity: Total account equity
        entry_price: Planned entry price
        stop_loss_price: Stop loss price level
        risk_per_trade_pct: Percentage of equity to risk per trade
        max_position_pct: Maximum position as % of equity
        alignment_score: Cross-timeframe alignment score (0-1)

    Returns:
        Dictionary with position size and risk details
    """
    try:
        sizer = ATRPositionSizer(
            risk_per_trade_pct=risk_per_trade_pct,
            max_position_pct=max_position_pct,
        )

        result = sizer.calculate(
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            alignment_score=alignment_score,
        )

        return {
            "position": {
                "shares": round(result.shares, 2),
                "notional_value": round(result.notional_value, 2),
                "position_pct_of_equity": round(
                    result.notional_value / equity * 100, 2
                ),
            },
            "risk": {
                "risk_amount": round(result.risk_amount, 2),
                "risk_pct": round(result.risk_pct, 2),
                "risk_per_share": round(abs(entry_price - stop_loss_price), 2),
            },
            "adjustments": {
                "alignment_multiplier": round(result.alignment_multiplier, 2),
                "was_capped": result.notional_value >= equity * max_position_pct / 100,
            },
            "trade_details": {
                "entry_price": entry_price,
                "stop_loss": stop_loss_price,
                "stop_distance_pct": round(
                    abs(entry_price - stop_loss_price) / entry_price * 100, 2
                ),
            },
        }
    except Exception as e:
        return {"error": str(e)}


@domain(Domain.RISK)
@mcp.tool()
async def compute_max_drawdown(
    equity_curve: list[float],
) -> dict[str, Any]:
    """
    Compute maximum drawdown and drawdown statistics.

    Args:
        equity_curve: List of equity values over time

    Returns:
        Dictionary with drawdown metrics
    """
    try:
        equity = np.array(equity_curve)

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)

        # Calculate drawdown at each point
        drawdown = (equity - running_max) / running_max * 100

        # Find max drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.argmin()

        # Find peak before max drawdown
        peak_idx = running_max[: max_dd_idx + 1].argmax()

        # Find recovery point (if any)
        recovery_idx = None
        for i in range(max_dd_idx, len(equity)):
            if equity[i] >= running_max[max_dd_idx]:
                recovery_idx = i
                break

        # Calculate current drawdown
        current_dd = drawdown[-1]

        return {
            "max_drawdown_pct": round(max_dd, 2),
            "max_drawdown_idx": int(max_dd_idx),
            "peak_idx": int(peak_idx),
            "peak_value": round(equity[peak_idx], 2),
            "trough_value": round(equity[max_dd_idx], 2),
            "recovery_idx": int(recovery_idx) if recovery_idx else None,
            "drawdown_duration": int(max_dd_idx - peak_idx),
            "recovery_duration": (
                int(recovery_idx - max_dd_idx) if recovery_idx else None
            ),
            "current_drawdown_pct": round(current_dd, 2),
            "is_in_drawdown": current_dd < 0,
        }
    except Exception as e:
        return {"error": str(e)}


@domain(Domain.RISK)
@mcp.tool()
async def compute_var(
    returns: list[float],
    confidence_levels: list[float] = None,
    method: str = "historical",
    horizon_days: int = 1,
) -> dict[str, Any]:
    """
    Compute Value at Risk (VaR) and Expected Shortfall (CVaR).

    Supports multiple calculation methods:
    - Historical: Uses empirical distribution of returns
    - Parametric: Assumes normal distribution
    - Monte Carlo: Simulates future returns

    Args:
        returns: Historical returns series
        confidence_levels: VaR confidence levels (e.g., [0.95, 0.99])
        method: Calculation method ("historical", "parametric", "monte_carlo")
        horizon_days: VaR horizon in days

    Returns:
        Dictionary with VaR, CVaR, and distribution statistics
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]
    try:
        returns_arr = np.array(returns)

        if len(returns_arr) < 30:
            return {"error": "Need at least 30 returns for VaR calculation"}

        # Scale returns to horizon
        if horizon_days > 1:
            returns_arr = returns_arr * np.sqrt(horizon_days)

        result = {
            "method": method,
            "horizon_days": horizon_days,
            "sample_size": len(returns_arr),
            "var": {},
            "cvar": {},
        }

        mean = np.mean(returns_arr)
        std = np.std(returns_arr)

        for conf in confidence_levels:
            alpha = 1 - conf
            conf_str = f"{int(conf * 100)}"

            if method == "historical":
                var = -np.percentile(returns_arr, alpha * 100)
                # CVaR = average of returns below VaR
                cvar = -np.mean(returns_arr[returns_arr <= -var])

            elif method == "parametric":
                z_score = norm.ppf(alpha)
                var = -(mean + z_score * std)
                # Parametric CVaR
                cvar = -(mean - std * norm.pdf(z_score) / alpha)

            elif method == "monte_carlo":
                # Simulate 10000 returns
                simulated = np.random.normal(mean, std, 10000)
                var = -np.percentile(simulated, alpha * 100)
                cvar = -np.mean(simulated[simulated <= -var])

            else:
                return {"error": f"Unknown method: {method}"}

            result["var"][conf_str] = round(
                float(var) * 100, 4
            )  # Convert to percentage
            result["cvar"][conf_str] = round(float(cvar) * 100, 4)

        result["statistics"] = {
            "mean_return": round(float(mean) * 100, 4),
            "volatility": round(float(std) * 100, 4),
            "skewness": round(float(pd.Series(returns_arr).skew()), 4),
            "kurtosis": round(float(pd.Series(returns_arr).kurtosis()), 4),
            "max_loss": round(float(-np.min(returns_arr)) * 100, 4),
        }

        return result

    except Exception as e:
        return {"error": str(e)}


@domain(Domain.RISK)
@mcp.tool()
async def stress_test_portfolio(
    positions: list[dict[str, Any]],
    scenarios: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run stress tests on an options portfolio.

    Tests portfolio against predefined historical scenarios:
    - 2008 Lehman: -40% price, +80% vol
    - 2020 COVID: -35% price, +100% vol
    - 2018 Volmageddon: -5% price, +150% vol
    - And more...

    Args:
        positions: List of positions, each with:
            - symbol: Underlying symbol
            - option_type: "call" or "put"
            - strike: Strike price
            - expiry_days: Days to expiration
            - quantity: Number of contracts
            - current_price: Current option price
        scenarios: Specific scenarios to test (None = all)

    Returns:
        Dictionary with P&L under each scenario
    """
    try:
        if not positions:
            return {"error": "No positions provided"}

        # Convert to OptionsPosition objects
        opt_positions = []
        for pos in positions:
            opt_type = (
                OptionType.CALL
                if pos.get("option_type", "call").lower() == "call"
                else OptionType.PUT
            )
            opt_positions.append(
                OptionsPosition(
                    symbol=pos.get("symbol", "SPY"),
                    option_type=opt_type,
                    strike=float(pos.get("strike", 100)),
                    expiry_days=int(pos.get("expiry_days", 30)),
                    quantity=int(pos.get("quantity", 1)),
                    entry_price=float(pos.get("current_price", 5)),
                    current_price=float(pos.get("current_price", 5)),
                    underlying_price=float(pos.get("underlying_price", 100)),
                    iv=float(pos.get("iv", 0.25)),
                )
            )

        # Select scenarios
        if scenarios:
            test_scenarios = {
                k: v for k, v in STRESS_SCENARIOS.items() if k in scenarios
            }
        else:
            test_scenarios = STRESS_SCENARIOS

        # Run stress tests
        results = []
        for scenario_name, (price_shock, vol_shock) in test_scenarios.items():
            total_pnl = 0
            position_pnls = []

            for pos in opt_positions:
                # Apply shocks
                new_underlying = pos.underlying_price * (1 + price_shock)
                new_vol = pos.iv * (1 + vol_shock)

                # Recalculate option price (simplified)
                new_price = black_scholes_price(
                    S=new_underlying,
                    K=pos.strike,
                    T=max(pos.expiry_days / 365, 0.001),
                    sigma=new_vol,
                    r=0.05,
                    q=0,
                    option_type="call" if pos.option_type == OptionType.CALL else "put",
                )

                pnl = (new_price - pos.current_price) * pos.quantity * 100
                total_pnl += pnl
                position_pnls.append(
                    {
                        "symbol": pos.symbol,
                        "type": pos.option_type.value,
                        "strike": pos.strike,
                        "pnl": round(pnl, 2),
                    }
                )

            results.append(
                {
                    "scenario": scenario_name,
                    "price_shock_pct": round(price_shock * 100, 1),
                    "vol_shock_pct": round(vol_shock * 100, 1),
                    "total_pnl": round(total_pnl, 2),
                    "position_pnls": position_pnls,
                }
            )

        # Sort by worst case
        results.sort(key=lambda x: x["total_pnl"])

        return {
            "portfolio_size": len(positions),
            "scenarios_tested": len(results),
            "worst_case": results[0] if results else None,
            "best_case": results[-1] if results else None,
            "all_scenarios": results,
        }

    except Exception as e:
        return {"error": str(e)}


@domain(Domain.RISK)
@mcp.tool()
async def check_risk_limits(
    equity: float,
    daily_pnl: float,
    open_trades: int,
    open_exposure_pct: float,
    drawdown_pct: float,
    max_daily_loss_pct: float = 2.0,
    max_drawdown_pct: float = 10.0,
    max_concurrent_trades: int = 5,
    max_exposure_pct: float = 80.0,
) -> dict[str, Any]:
    """
    Check current risk state against limits.

    Evaluates:
    - Daily P&L limits
    - Drawdown limits
    - Position count limits
    - Exposure limits

    Args:
        equity: Current equity
        daily_pnl: Today's P&L
        open_trades: Number of open positions
        open_exposure_pct: Current exposure as % of equity
        drawdown_pct: Current drawdown from peak
        max_daily_loss_pct: Maximum daily loss allowed (%)
        max_drawdown_pct: Maximum drawdown allowed (%)
        max_concurrent_trades: Maximum open positions
        max_exposure_pct: Maximum exposure allowed (%)

    Returns:
        RiskState with status, breaches, and recommendations
    """

    try:
        messages = []
        breaches = []
        status = "NORMAL"

        daily_loss_pct = (daily_pnl / equity) * 100 if equity > 0 else 0

        # Check daily loss
        if abs(daily_loss_pct) >= max_daily_loss_pct:
            breaches.append("daily_loss")
            messages.append(
                f"Daily loss limit breached: {daily_loss_pct:.2f}% >= {max_daily_loss_pct}%"
            )
            status = "HALTED"
        elif abs(daily_loss_pct) >= max_daily_loss_pct * 0.8:
            messages.append(f"Approaching daily loss limit: {daily_loss_pct:.2f}%")
            if status != "HALTED":
                status = "CAUTION"

        # Check drawdown
        if abs(drawdown_pct) >= max_drawdown_pct:
            breaches.append("drawdown")
            messages.append(
                f"Drawdown limit breached: {drawdown_pct:.2f}% >= {max_drawdown_pct}%"
            )
            status = "HALTED"
        elif abs(drawdown_pct) >= max_drawdown_pct * 0.8:
            messages.append(f"Approaching drawdown limit: {drawdown_pct:.2f}%")
            if status not in ["HALTED"]:
                status = "CAUTION"

        # Check position count
        if open_trades >= max_concurrent_trades:
            breaches.append("position_count")
            messages.append(
                f"Position limit reached: {open_trades} >= {max_concurrent_trades}"
            )
            if status not in ["HALTED"]:
                status = "RESTRICTED"

        # Check exposure
        if open_exposure_pct >= max_exposure_pct:
            breaches.append("exposure")
            messages.append(
                f"Exposure limit breached: {open_exposure_pct:.2f}% >= {max_exposure_pct}%"
            )
            if status not in ["HALTED"]:
                status = "RESTRICTED"

        can_trade = status in ["NORMAL", "CAUTION"]
        size_multiplier = (
            1.0 if status == "NORMAL" else 0.5 if status == "CAUTION" else 0.0
        )

        return {
            "status": status,
            "can_trade": can_trade,
            "size_multiplier": size_multiplier,
            "breaches": breaches,
            "messages": messages,
            "current_state": {
                "equity": equity,
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": round(daily_loss_pct, 2),
                "drawdown_pct": drawdown_pct,
                "open_trades": open_trades,
                "open_exposure_pct": open_exposure_pct,
            },
            "limits": {
                "max_daily_loss_pct": max_daily_loss_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "max_concurrent_trades": max_concurrent_trades,
                "max_exposure_pct": max_exposure_pct,
            },
        }

    except Exception as e:
        return {"error": str(e)}
