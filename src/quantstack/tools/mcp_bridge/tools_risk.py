# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Risk management tool classes wrapping QuantCore MCP server calls."""

import json

from pydantic import BaseModel

from quantstack.crewai_compat import BaseTool

from ._bridge import _run_async, get_bridge
from ._schemas import (
    LiquidityInput,
    MaxDrawdownInput,
    PortfolioStatsInput,
    PositionSizeInput,
    RiskLimitsInput,
    StressTestInput,
    VaRInput,
)


class ComputePositionSizeTool(BaseTool):
    """Tool to calculate position size."""

    name: str = "compute_position_size"
    description: str = (
        "Calculate optimal position size using ATR-based risk management and Kelly criterion."
    )
    args_schema: type[BaseModel] = PositionSizeInput

    def _run(
        self,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade_pct: float = 1,
        max_position_pct: float = 20,
        alignment_score: float = 1,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_position_size",
                equity=equity,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                risk_per_trade_pct=risk_per_trade_pct,
                max_position_pct=max_position_pct,
                alignment_score=alignment_score,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeMaxDrawdownTool(BaseTool):
    """Tool to compute maximum drawdown."""

    name: str = "compute_max_drawdown"
    description: str = (
        "Compute maximum drawdown and drawdown statistics from equity curve."
    )
    args_schema: type[BaseModel] = MaxDrawdownInput

    def _run(self, equity_curve: list[float]) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_max_drawdown", equity_curve=equity_curve
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputePortfolioStatsTool(BaseTool):
    """Tool to compute portfolio statistics."""

    name: str = "compute_portfolio_stats"
    description: str = (
        "Compute portfolio statistics including Sharpe ratio, volatility, and risk metrics."
    )
    args_schema: type[BaseModel] = PortfolioStatsInput

    def _run(self, returns: list[float], risk_free_rate: float = 0.02) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_portfolio_stats",
                returns=returns,
                risk_free_rate=risk_free_rate,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeVaRTool(BaseTool):
    """Tool to compute Value at Risk."""

    name: str = "compute_var"
    description: str = "Compute Value at Risk (VaR) using historical simulation method."
    args_schema: type[BaseModel] = VaRInput

    def _run(
        self,
        returns: list[float],
        confidence_level: float = 0.95,
        portfolio_value: float = 100000,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_var",
                returns=returns,
                confidence_level=confidence_level,
                portfolio_value=portfolio_value,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class StressTestPortfolioTool(BaseTool):
    """Tool to stress test portfolio."""

    name: str = "stress_test_portfolio"
    description: str = "Run stress tests on portfolio with various market scenarios."
    args_schema: type[BaseModel] = StressTestInput

    def _run(self, positions: str, scenarios: str) -> str:
        async def _exec():
            bridge = get_bridge()
            pos_list = json.loads(positions)
            scen_list = json.loads(scenarios)
            return await bridge.call_quantcore(
                "stress_test_portfolio", positions=pos_list, scenarios=scen_list
            )

        return json.dumps(_run_async(_exec()), indent=2)


class CheckRiskLimitsTool(BaseTool):
    """Tool to check risk limits."""

    name: str = "check_risk_limits"
    description: str = "Check if portfolio Greeks are within risk limits."
    args_schema: type[BaseModel] = RiskLimitsInput

    def _run(
        self,
        portfolio_delta: float,
        portfolio_gamma: float,
        portfolio_vega: float,
        max_delta: float = 100,
        max_gamma: float = 50,
        max_vega: float = 5000,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "check_risk_limits",
                portfolio_delta=portfolio_delta,
                portfolio_gamma=portfolio_gamma,
                portfolio_vega=portfolio_vega,
                max_delta=max_delta,
                max_gamma=max_gamma,
                max_vega=max_vega,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class AnalyzeLiquidityTool(BaseTool):
    """Tool to analyze market liquidity."""

    name: str = "analyze_liquidity"
    description: str = (
        "Analyze market liquidity including bid-ask spread, volume, and market impact."
    )
    args_schema: type[BaseModel] = LiquidityInput

    def _run(
        self, symbol: str, timeframe: str = "daily", end_date: str | None = None
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "analyze_liquidity",
                symbol=symbol,
                timeframe=timeframe,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)
