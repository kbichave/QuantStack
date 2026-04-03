# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Market/regime and trade tool classes wrapping QuantCore MCP server calls."""

import json

from pydantic import BaseModel

from quantstack.tools.tool_base import BaseTool

from ._bridge import _run_async, get_bridge
from ._schemas import (
    EventCalendarInput,
    MarketRegimeSnapshotInput,
    ScoreTradeInput,
    ScreenerInput,
    SimulateTradeInput,
    TradeTemplateInput,
    TradingCalendarInput,
    ValidateTradeInput,
    VolumeProfileInput,
)


# =============================================================================
# QUANTCORE MARKET/REGIME TOOL CLASSES
# =============================================================================


class GetMarketRegimeSnapshotTool(BaseTool):
    """Tool to get market regime snapshot."""

    name: str = "get_market_regime_snapshot"
    description: str = (
        "Get current market regime classification (trending, ranging, volatile) with confidence."
    )
    args_schema: type[BaseModel] = MarketRegimeSnapshotInput

    def _run(self, end_date: str | None = None) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_market_regime_snapshot", end_date=end_date
            )

        return json.dumps(_run_async(_exec()), indent=2)


class AnalyzeVolumeProfileTool(BaseTool):
    """Tool to analyze volume profile."""

    name: str = "analyze_volume_profile"
    description: str = (
        "Analyze volume profile to identify support/resistance levels and value areas."
    )
    args_schema: type[BaseModel] = VolumeProfileInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        num_bins: int = 20,
        end_date: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "analyze_volume_profile",
                symbol=symbol,
                timeframe=timeframe,
                num_bins=num_bins,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetTradingCalendarTool(BaseTool):
    """Tool to get trading calendar."""

    name: str = "get_trading_calendar"
    description: str = "Get trading calendar with market holidays and trading days."
    args_schema: type[BaseModel] = TradingCalendarInput

    def _run(self, start_date: str, end_date: str) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_trading_calendar", start_date=start_date, end_date=end_date
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetEventCalendarTool(BaseTool):
    """Tool to get event calendar."""

    name: str = "get_event_calendar"
    description: str = (
        "Get upcoming market events (earnings, Fed meetings, economic releases)."
    )
    args_schema: type[BaseModel] = EventCalendarInput

    def _run(self, symbol: str | None = None, days_ahead: int = 7) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_event_calendar", symbol=symbol, days_ahead=days_ahead
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE TRADE TOOL CLASSES
# =============================================================================


class GenerateTradeTemplateTool(BaseTool):
    """Tool to generate trade template."""

    name: str = "generate_trade_template"
    description: str = (
        "Generate a trade template with entry, stop, target based on symbol analysis."
    )
    args_schema: type[BaseModel] = TradeTemplateInput

    def _run(
        self,
        symbol: str,
        direction: str,
        structure_type: str = "VERTICAL_SPREAD",
        max_risk: float = 500,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "generate_trade_template",
                symbol=symbol,
                direction=direction,
                structure_type=structure_type,
                max_risk=max_risk,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ValidateTradeTool(BaseTool):
    """Tool to validate a trade setup."""

    name: str = "validate_trade"
    description: str = (
        "Validate a trade setup against risk rules and market conditions."
    )
    args_schema: type[BaseModel] = ValidateTradeInput

    def _run(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        position_size: float,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "validate_trade",
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                position_size=position_size,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ScoreTradeStructureTool(BaseTool):
    """Tool to score a trade structure."""

    name: str = "score_trade_structure"
    description: str = (
        "Score an options trade structure based on risk/reward, probability, and Greeks."
    )
    args_schema: type[BaseModel] = ScoreTradeInput

    def _run(
        self,
        structure_type: str,
        max_profit: float,
        max_loss: float,
        probability_of_profit: float,
        days_to_expiry: int,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "score_trade_structure",
                structure_type=structure_type,
                max_profit=max_profit,
                max_loss=max_loss,
                probability_of_profit=probability_of_profit,
                days_to_expiry=days_to_expiry,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class SimulateTradeOutcomeTool(BaseTool):
    """Tool to simulate trade outcomes."""

    name: str = "simulate_trade_outcome"
    description: str = (
        "Simulate potential trade outcomes using Monte Carlo based on historical volatility."
    )
    args_schema: type[BaseModel] = SimulateTradeInput

    def _run(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: int,
        days_to_hold: int = 20,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "simulate_trade_outcome",
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                days_to_hold=days_to_hold,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class RunScreenerTool(BaseTool):
    """Tool to run market screener."""

    name: str = "run_screener"
    description: str = "Screen stocks based on technical and price criteria."
    args_schema: type[BaseModel] = ScreenerInput

    def _run(
        self,
        min_price: float = 10,
        max_price: float = 500,
        min_volume: int = 1000000,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_screener",
                min_price=min_price,
                max_price=max_price,
                min_volume=min_volume,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
            )

        return json.dumps(_run_async(_exec()), indent=2)
