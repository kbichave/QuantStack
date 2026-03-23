# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Shared tool implementations — importable by other tool modules without
creating circular dependencies through mcp.server.

This module does NOT import mcp.server or @mcp.tool() — it only contains
the raw business logic that multiple MCP tools need to share.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from quantstack.mcp._state import live_db_or_error


async def get_strategy_impl(
    strategy_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Core logic for get_strategy — callable from other tool modules."""
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        if strategy_id:
            row = ctx.db.execute(
                "SELECT * FROM strategies WHERE strategy_id = ?", [strategy_id]
            ).fetchone()
        elif name:
            row = ctx.db.execute(
                "SELECT * FROM strategies WHERE name = ?", [name]
            ).fetchone()
        else:
            return {"success": False, "error": "Provide strategy_id or name"}

        if row is None:
            return {"success": False, "error": "Strategy not found"}

        cols = [
            "strategy_id",
            "name",
            "description",
            "asset_class",
            "regime_affinity",
            "parameters",
            "entry_rules",
            "exit_rules",
            "risk_params",
            "backtest_summary",
            "walkforward_summary",
            "status",
            "source",
            "created_at",
            "updated_at",
            "created_by",
            "instrument_type",
            "time_horizon",
            "holding_period_days",
        ]
        record = {}
        for i, col in enumerate(cols):
            val = row[i]
            if isinstance(val, str) and col in (
                "regime_affinity",
                "parameters",
                "entry_rules",
                "exit_rules",
                "risk_params",
                "backtest_summary",
                "walkforward_summary",
            ):
                try:
                    val = json.loads(val)
                except (ValueError, TypeError):
                    pass
            if col in ("created_at", "updated_at") and val is not None:
                val = str(val)
            record[col] = val

        return {"success": True, "strategy": record}
    except Exception as e:
        logger.error(f"[_impl] get_strategy failed: {e}")
        return {"success": False, "error": str(e)}
