"""Strategy lifecycle tools for LangGraph agents."""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

logger = logging.getLogger(__name__)


@tool
async def fetch_strategy_registry(
    status: Annotated[str | None, Field(description="Filter strategies by lifecycle status: 'active', 'forward_testing', 'draft', 'retired', or None to list all strategies")] = None,
) -> str:
    """Retrieves the full strategy registry with details for each registered strategy including ID, name, symbol, instrument type, time horizon, backtest summary, and status. Use when surveying the current strategy roster, checking which strategies are active or in forward testing, or auditing the strategy pipeline. Returns JSON array of strategy records sorted by creation date descending."""
    try:
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT strategy_id, name, status, symbol, instrument_type, "
                    "time_horizon, backtest_summary, created_at "
                    "FROM strategies WHERE status = ? ORDER BY created_at DESC",
                    [status],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT strategy_id, name, status, symbol, instrument_type, "
                    "time_horizon, backtest_summary, created_at "
                    "FROM strategies ORDER BY created_at DESC",
                ).fetchall()

        strategies = []
        cols = ["strategy_id", "name", "status", "symbol", "instrument_type",
                "time_horizon", "backtest_summary", "created_at"]
        for row in rows:
            entry = {}
            for i, col in enumerate(cols):
                val = row[i]
                if col == "backtest_summary" and isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (ValueError, TypeError):
                        pass
                if col == "created_at" and val is not None:
                    val = str(val)
                entry[col] = val
            strategies.append(entry)

        return json.dumps({"success": True, "strategies": strategies, "count": len(strategies)}, default=str)
    except Exception as e:
        logger.error(f"fetch_strategy_registry failed: {e}")
        return json.dumps({"error": str(e)})


@tool
async def register_strategy(
    name: Annotated[str, Field(description="Unique human-readable strategy name, e.g. 'AAPL_mean_reversion_swing'")],
    symbol: Annotated[str, Field(description="Ticker symbol the strategy trades, e.g. 'AAPL', 'SPY', 'QQQ'")],
    strategy_type: Annotated[str, Field(description="Strategy archetype: 'mean_reversion', 'momentum', 'trend_following', 'stat_arb', 'options_spread', etc.")],
    entry_rules: Annotated[list | None, Field(description="List of entry rule dicts defining conditions to open a position, e.g. [{'indicator': 'RSI', 'condition': 'below', 'value': 30}]")] = None,
    exit_rules: Annotated[list | None, Field(description="List of exit rule dicts defining conditions to close a position, e.g. [{'type': 'stop_loss', 'value': 0.05}]")] = None,
    parameters: Annotated[dict | None, Field(description="Strategy-specific parameters dict, e.g. {'lookback': 20, 'threshold': 1.5}")] = None,
    description: Annotated[str, Field(description="Human-readable description of the strategy thesis, edge, and expected market regime")] = "",
    instrument_type: Annotated[str, Field(description="Instrument class: 'equity' for stocks/ETFs or 'options' for options strategies")] = "equity",
    time_horizon: Annotated[str, Field(description="Trading time horizon: 'day' for intraday, 'swing' for multi-day, 'investment' for long-term holds")] = "swing",
) -> str:
    """Registers a new trading strategy in the strategy registry with draft status. Use after completing research and backtesting to persist a strategy for promotion to forward testing and eventually live trading. Provides entry rules, exit rules, parameters, and metadata. Returns JSON with the assigned strategy_id and creation confirmation."""
    try:
        from quantstack.tools._shared import register_strategy_impl

        result = await register_strategy_impl(
            name=name,
            symbol=symbol,
            parameters=parameters or {"type": strategy_type},
            entry_rules=entry_rules or [],
            exit_rules=exit_rules or [],
            description=description,
            source="research_graph",
            instrument_type=instrument_type,
            time_horizon=time_horizon,
        )
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"register_strategy failed: {e}")
        return json.dumps({"error": str(e)})


@tool
async def get_strategy(
    strategy_id: Annotated[str, Field(description="Unique strategy identifier (UUID) to retrieve full details for")],
) -> str:
    """Retrieves complete details for a single strategy by ID including name, rules, parameters, backtest results, and current status. Use when inspecting a specific strategy before trade execution, reviewing its configuration, or preparing a promotion/retirement decision. Returns JSON with all strategy fields."""
    try:
        from quantstack.tools._shared import get_strategy_impl

        result = await get_strategy_impl(strategy_id=strategy_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def update_strategy(
    strategy_id: Annotated[str, Field(description="Unique strategy identifier (UUID) of the strategy to modify")],
    updates: Annotated[dict, Field(description="Dictionary of fields to update, e.g. {'description': 'new thesis', 'parameters': {'lookback': 30}}")],
) -> str:
    """Updates a strategy's metadata, parameters, rules, or status fields in the registry. Use when tuning strategy parameters after performance review, correcting configuration errors, or changing descriptive metadata. Returns JSON confirmation with the updated fields."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
