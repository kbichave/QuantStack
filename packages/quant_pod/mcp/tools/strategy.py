"""Phase 2 — Strategy Registry CRUD tools.

Extracted from server.py to reduce module size. These four tools manage the
persistent strategy catalog: register, list, get, and update.
"""

from typing import Any

from loguru import logger

from quant_pod.mcp.server import mcp
from quant_pod.mcp._state import require_ctx, live_db_or_error, _serialize


@mcp.tool()
async def register_strategy(
    name: str,
    parameters: dict[str, Any],
    entry_rules: list[dict[str, Any]],
    exit_rules: list[dict[str, Any]],
    description: str = "",
    asset_class: str = "equities",
    regime_affinity: dict[str, float] | None = None,
    risk_params: dict[str, Any] | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    """
    Register a new strategy in the persistent catalog.

    Args:
        name: Unique human-readable name.
        parameters: Indicator settings (e.g., {"rsi_period": 14}).
        entry_rules: List of rule dicts (e.g., [{"indicator": "rsi_14", "condition": "crosses_below", "value": 30}]).
        exit_rules: List of rule dicts for exit conditions.
        description: Free-text strategy description.
        asset_class: "equities", "options", "futures", "fx_crypto".
        regime_affinity: Map of regime → suitability score (0-1).
        risk_params: Sizing/stop config (e.g., {"stop_loss_atr": 2.0}).
        source: "manual", "decoded", "workshop", "generated".

    Returns:
        Dict with strategy_id and status.
    """
    import json
    import uuid

    ctx, err = live_db_or_error()
    if err:
        return err
    strategy_id = f"strat_{uuid.uuid4().hex[:12]}"

    try:
        ctx.db.execute(
            """
            INSERT INTO strategies
                (strategy_id, name, description, asset_class, regime_affinity,
                 parameters, entry_rules, exit_rules, risk_params, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?)
            """,
            [
                strategy_id,
                name,
                description,
                asset_class,
                json.dumps(regime_affinity or {}),
                json.dumps(parameters),
                json.dumps(entry_rules),
                json.dumps(exit_rules),
                json.dumps(risk_params or {}),
                source,
            ],
        )
        logger.info(f"[quantpod_mcp] Registered strategy {strategy_id}: {name}")
        return {"success": True, "strategy_id": strategy_id, "status": "draft"}
    except Exception as e:
        logger.error(f"[quantpod_mcp] register_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_strategies(
    status: str | None = None,
    asset_class: str | None = None,
) -> dict[str, Any]:
    """
    List strategies from the registry, optionally filtered.

    Args:
        status: Filter by status (draft, backtested, forward_testing, live, failed, retired).
        asset_class: Filter by asset class.

    Returns:
        Dict with list of strategy summaries.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        conditions = []
        params = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if asset_class:
            conditions.append("asset_class = ?")
            params.append(asset_class)

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = ctx.db.execute(
            f"SELECT strategy_id, name, description, asset_class, status, source, "
            f"created_at, updated_at FROM strategies{where} ORDER BY updated_at DESC",
            params,
        ).fetchall()

        strategies = [
            {
                "strategy_id": r[0],
                "name": r[1],
                "description": r[2],
                "asset_class": r[3],
                "status": r[4],
                "source": r[5],
                "created_at": str(r[6]) if r[6] else None,
                "updated_at": str(r[7]) if r[7] else None,
            }
            for r in rows
        ]
        return {"success": True, "strategies": strategies, "total": len(strategies)}
    except Exception as e:
        logger.error(f"[quantpod_mcp] list_strategies failed: {e}")
        return {"success": False, "error": str(e), "strategies": [], "total": 0}


@mcp.tool()
async def get_strategy(
    strategy_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """
    Get full strategy details by ID or name.

    Args:
        strategy_id: Strategy UUID.
        name: Strategy name (used if strategy_id is None).

    Returns:
        Full strategy record.
    """
    import json as _json

    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        if strategy_id:
            row = ctx.db.execute(
                "SELECT * FROM strategies WHERE strategy_id = ?", [strategy_id]
            ).fetchone()
        elif name:
            row = ctx.db.execute("SELECT * FROM strategies WHERE name = ?", [name]).fetchone()
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
                    val = _json.loads(val)
                except (ValueError, TypeError):
                    pass
            if col in ("created_at", "updated_at") and val is not None:
                val = str(val)
            record[col] = val

        return {"success": True, "strategy": record}
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def update_strategy(
    strategy_id: str,
    status: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    entry_rules: list[dict[str, Any]] | None = None,
    exit_rules: list[dict[str, Any]] | None = None,
    risk_params: dict[str, Any] | None = None,
    regime_affinity: dict[str, float] | None = None,
    backtest_summary: dict[str, Any] | None = None,
    walkforward_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Update fields of an existing strategy.

    Only non-None fields are updated. Pass the fields you want to change.

    Args:
        strategy_id: Strategy to update.
        status: New status (draft, backtested, forward_testing, live, failed, retired).
        description: Updated description.
        parameters: Updated indicator parameters.
        entry_rules: Updated entry rules.
        exit_rules: Updated exit rules.
        risk_params: Updated risk parameters.
        regime_affinity: Updated regime affinity map.
        backtest_summary: Backtest results to store.
        walkforward_summary: Walk-forward results to store.

    Returns:
        Updated strategy record.
    """
    import json as _json

    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        sets = []
        params = []
        field_map = {
            "status": status,
            "description": description,
        }
        json_fields = {
            "parameters": parameters,
            "entry_rules": entry_rules,
            "exit_rules": exit_rules,
            "risk_params": risk_params,
            "regime_affinity": regime_affinity,
            "backtest_summary": backtest_summary,
            "walkforward_summary": walkforward_summary,
        }

        for col, val in field_map.items():
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(val)

        for col, val in json_fields.items():
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(_json.dumps(val))

        if not sets:
            return {"success": False, "error": "No fields to update"}

        sets.append("updated_at = CURRENT_TIMESTAMP")
        params.append(strategy_id)

        ctx.db.execute(
            f"UPDATE strategies SET {', '.join(sets)} WHERE strategy_id = ?",
            params,
        )
        logger.info(f"[quantpod_mcp] Updated strategy {strategy_id}")

        # Return the updated record
        return await get_strategy(strategy_id=strategy_id)
    except Exception as e:
        logger.error(f"[quantpod_mcp] update_strategy failed: {e}")
        return {"success": False, "error": str(e)}
