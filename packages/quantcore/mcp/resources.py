# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""QuantCore MCP resources — read-only data exposed to MCP clients."""

import json

from quantcore.config.timeframes import TIMEFRAME_PARAMS, Timeframe
from quantcore.mcp.server import mcp


@mcp.resource("quantcore://symbols")
async def get_symbols_resource() -> str:
    """Get list of tracked symbols."""
    from quantcore.config.settings import get_settings

    settings = get_settings()
    return json.dumps(
        {
            "symbols": settings.symbols,
            "benchmark": settings.benchmark_symbol,
        }
    )


@mcp.resource("quantcore://config")
async def get_config_resource() -> str:
    """Get current platform configuration."""
    from quantcore.config.settings import get_settings

    settings = get_settings()
    return json.dumps(
        {
            "database_path": settings.database_path,
            "data_start_date": settings.data_start_date,
            "data_end_date": settings.data_end_date,
            "train_end_date": settings.train_end_date,
            "risk_per_trade_bps": settings.max_risk_per_trade_bps,
            "max_concurrent_trades": settings.max_concurrent_trades,
            "transaction_cost_bps": settings.total_transaction_cost_bps,
        }
    )


@mcp.resource("quantcore://indicators")
async def get_indicators_resource() -> str:
    """Get available indicators catalog."""
    from quantcore.mcp.tools.indicators import list_available_indicators

    result = await list_available_indicators()
    return json.dumps(result)


@mcp.resource("quantcore://timeframes")
async def get_timeframes_resource() -> str:
    """Get available timeframes and their parameters."""
    timeframes = {}
    for tf in Timeframe:
        params = TIMEFRAME_PARAMS[tf]
        timeframes[tf.value] = {
            "ema_fast": params.ema_fast,
            "ema_slow": params.ema_slow,
            "rsi_period": params.rsi_period,
            "atr_period": params.atr_period,
            "resample_rule": params.resample_rule,
        }
    return json.dumps({"timeframes": timeframes})
