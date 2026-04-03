"""Strategy lifecycle tools for LangGraph agents."""

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def fetch_strategy_registry(status: str | None = None) -> str:
    """Fetch strategies from the registry, optionally filtered by status.

    Args:
        status: Filter by status ("active", "forward_testing", "retired", or None for all).

    Returns JSON with strategy details: ID, type, symbol, performance, status.
    """
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
    name: str,
    symbol: str,
    strategy_type: str,
    entry_rules: list | None = None,
    exit_rules: list | None = None,
    parameters: dict | None = None,
    description: str = "",
    instrument_type: str = "equity",
    time_horizon: str = "swing",
) -> str:
    """Register a new strategy in the registry as draft.

    Args:
        name: Unique strategy name.
        symbol: Ticker symbol.
        strategy_type: Strategy type (e.g., "mean_reversion", "momentum").
        entry_rules: List of entry rule dicts.
        exit_rules: List of exit rule dicts.
        parameters: Strategy parameters dict.
        description: Human-readable description.
        instrument_type: "equity" or "options".
        time_horizon: "swing", "investment", or "day".
    """
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
async def get_strategy(strategy_id: str) -> str:
    """Get details for a specific strategy.

    Args:
        strategy_id: Strategy ID to look up.
    """
    try:
        from quantstack.tools._shared import get_strategy_impl

        result = await get_strategy_impl(strategy_id=strategy_id)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def update_strategy(strategy_id: str, updates: dict) -> str:
    """Update a strategy's metadata or status.

    Args:
        strategy_id: Strategy ID to update.
        updates: Dict of fields to update.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
