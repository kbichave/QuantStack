"""Phase 4 — Decoder tools for the QuantPod MCP server.

Reverse-engineer trading strategies from historical trade signals or the
system's own trade history.

Tools:
  - decode_strategy     — decode strategy from external trade signals
  - decode_from_trades  — decode from the system's own closed trades or fills
"""

import asyncio
from typing import Any

from loguru import logger

from quantstack.crews.decoder_crew import decode_signals
from quantstack.mcp._state import (
    _serialize,
    live_db_or_error,
    require_ctx,
    require_live_db,
)
from quantstack.mcp.server import mcp
from quantstack.mcp.tools.strategy import register_strategy


@mcp.tool()
async def decode_strategy(
    signals: list[dict[str, Any]],
    source_name: str = "unknown",
    strategy_name: str | None = None,
) -> dict[str, Any]:
    """
    Reverse-engineer a trading strategy from historical trade signals.

    Analyzes entry patterns (timing, direction bias), exit patterns (holding
    period, target vs time-based), sizing patterns, and regime affinity.

    Args:
        signals: List of trade signal dicts. Each must have:
            symbol, direction, entry_time, entry_price, exit_time, exit_price.
            Optional: size, notes.
        source_name: Name of the signal source (e.g., "discord_trader_x").
        strategy_name: If provided, auto-registers the decoded strategy.

    Returns:
        DecodedStrategy with entry_trigger, exit_trigger, timing_pattern,
        win_rate, regime_affinity, edge_hypothesis, and per-IC analysis.
    """
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, decode_signals, signals, source_name
        )

        if not result.get("success"):
            return result

        # Auto-register if strategy_name provided
        if strategy_name and result.get("decoded_strategy"):
            decoded = result["decoded_strategy"]
            reg_result = await register_strategy(
                name=strategy_name,
                description=decoded.get("edge_hypothesis", ""),
                parameters={},
                entry_rules=[{"decoded_trigger": decoded.get("entry_trigger", "")}],
                exit_rules=[{"decoded_trigger": decoded.get("exit_trigger", "")}],
                regime_affinity=decoded.get("regime_affinity", {}),
                source="decoded",
            )
            result["registered"] = reg_result

        return result
    except Exception as e:
        logger.error(f"[quantpod_mcp] decode_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def decode_from_trades(
    source: str = "closed_trades",
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    source_name: str = "self",
) -> dict[str, Any]:
    """
    Decode strategy patterns from the system's own trade history.

    Pulls trades from the closed_trades or fills table and feeds them
    to the decoder.

    Args:
        source: "closed_trades" or "fills".
        symbol: Filter by symbol. None = all.
        start_date: Start date filter (YYYY-MM-DD).
        end_date: End date filter (YYYY-MM-DD).
        source_name: Label for the decoded source.

    Returns:
        DecodedStrategy from historical trades.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        if source == "closed_trades":
            query = "SELECT symbol, side, entry_price, exit_price, opened_at, closed_at FROM closed_trades"
        elif source == "fills":
            query = "SELECT symbol, side, fill_price, fill_price, filled_at, filled_at FROM fills WHERE rejected = FALSE"
        else:
            return {
                "success": False,
                "error": f"Unknown source: {source}. Use 'closed_trades' or 'fills'.",
            }

        conditions = []
        params = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_date:
            if source == "closed_trades":
                conditions.append("closed_at >= ?")
            else:
                conditions.append("filled_at >= ?")
            params.append(start_date)
        if end_date:
            if source == "closed_trades":
                conditions.append("closed_at <= ?")
            else:
                conditions.append("filled_at <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY 5"  # Order by entry/fill time

        rows = ctx.db.execute(query, params).fetchall()

        if not rows:
            return {"success": False, "error": "No trades found matching filters"}

        # Convert to signal format
        signals = []
        for r in rows:
            signals.append(
                {
                    "symbol": r[0],
                    "direction": "long" if r[1] == "long" else "short",
                    "entry_price": r[2],
                    "exit_price": r[3],
                    "entry_time": str(r[4]),
                    "exit_time": str(r[5]),
                }
            )

        return await decode_strategy(
            signals=signals,
            source_name=source_name,
        )
    except Exception as e:
        logger.error(f"[quantpod_mcp] decode_from_trades failed: {e}")
        return {"success": False, "error": str(e)}
