"""Phase 2 — Strategy Registry CRUD tools.

Extracted from server.py to reduce module size. These four tools manage the
persistent strategy catalog: register, list, get, and update.
"""

import json
import uuid
from typing import Any

import numpy as np
from loguru import logger

from quantstack.core.features.technical_indicators import TechnicalIndicators
from quantstack.data import DataStore
from quantstack.db import pg_conn
from quantstack.learning.drift_detector import TRACKED_FEATURES, DriftDetector
from quantstack.mcp._state import _serialize, live_db_or_error, require_ctx
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain
from quantstack.mcp.tools._impl import get_strategy_impl as _get_strategy_impl
from quantstack.mcp.tools._tool_def import tool_def



@domain(Domain.RESEARCH)
@tool_def()
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
    instrument_type: str = "equity",
    time_horizon: str = "swing",
    holding_period_days: int = 5,
    symbol: str | None = None,
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
        instrument_type: "equity", "options", "multi_leg" — what this strategy trades.
        time_horizon: "intraday", "swing", "position", "investment" — holding cadence.
        holding_period_days: Expected holding period in days (default 5 for swing).
        symbol: Ticker symbol this strategy targets (e.g., "SPY", "META").

    Returns:
        Dict with strategy_id and status.
    """
    _, err = live_db_or_error()
    if err:
        return err
    strategy_id = f"strat_{uuid.uuid4().hex[:12]}"

    try:
        with pg_conn() as conn:
            row = conn.execute(
                "SELECT strategy_id FROM strategies WHERE name = ?", [name]
            ).fetchone()
            if row:
                return {
                    "success": False,
                    "error": f"A strategy named '{name}' already exists (id={row[0]})",
                }
            conn.execute(
                """
                INSERT INTO strategies
                    (strategy_id, name, description, asset_class, regime_affinity,
                     parameters, entry_rules, exit_rules, risk_params, status, source,
                     instrument_type, time_horizon, holding_period_days, symbol)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?, ?, ?, ?)
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
                    instrument_type,
                    time_horizon,
                    holding_period_days,
                    symbol,
                ],
            )
        logger.info(f"[quantpod_mcp] Registered strategy {strategy_id}: {name}")
        return {"success": True, "strategy_id": strategy_id, "status": "draft"}
    except Exception as e:
        logger.error(f"[quantpod_mcp] register_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def list_strategies(
    status: str | None = None,
    asset_class: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """
    List strategies from the registry, optionally filtered.

    Args:
        status: Filter by status (draft, backtested, forward_testing, live, failed, retired).
        asset_class: Filter by asset class.
        symbol: Filter by ticker symbol (e.g., "SPY", "META").

    Returns:
        Dict with list of strategy summaries.
    """
    _, err = live_db_or_error()
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
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol.upper())

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with pg_conn() as conn:
            rows = conn.execute(
                f"SELECT strategy_id, name, description, asset_class, status, source, "
                f"created_at, updated_at, symbol FROM strategies{where} ORDER BY updated_at DESC",
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
                "symbol": r[8],
            }
            for r in rows
        ]
        return {"success": True, "strategies": strategies, "total": len(strategies)}
    except Exception as e:
        logger.error(f"[quantpod_mcp] list_strategies failed: {e}")
        return {"success": False, "error": str(e), "strategies": [], "total": 0}


@domain(Domain.RESEARCH)
@tool_def()
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
    return await _get_strategy_impl(strategy_id=strategy_id, name=name)


@domain(Domain.RESEARCH)
@tool_def()
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
    instrument_type: str | None = None,
    time_horizon: str | None = None,
    holding_period_days: int | None = None,
    symbol: str | None = None,
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
        instrument_type: "equity", "options", "multi_leg".
        time_horizon: "intraday", "swing", "position", "investment".
        holding_period_days: Expected holding period in days.
        symbol: Ticker symbol this strategy targets.

    Returns:
        Updated strategy record.
    """
    _, err = live_db_or_error()
    if err:
        return err
    try:
        sets = []
        params = []
        field_map = {
            "status": status,
            "description": description,
            "instrument_type": instrument_type,
            "time_horizon": time_horizon,
            "holding_period_days": holding_period_days,
            "symbol": symbol,
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
                params.append(json.dumps(val))

        if not sets:
            return {"success": False, "error": "No fields to update"}

        sets.append("updated_at = CURRENT_TIMESTAMP")
        params.append(strategy_id)

        with pg_conn() as conn:
            conn.execute(
                f"UPDATE strategies SET {', '.join(sets)} WHERE strategy_id = ?",
                params,
            )
        logger.info(f"[quantpod_mcp] Updated strategy {strategy_id}")

        # Auto-create drift baseline when promoted to forward_testing
        if status == "forward_testing":
            _create_drift_baseline(strategy_id)

        updated_fields = [k for k, v in {**field_map, **json_fields}.items() if v is not None]
        return {"success": True, "strategy_id": strategy_id, "updated_fields": updated_fields}
    except Exception as e:
        logger.error(f"[quantpod_mcp] update_strategy failed: {e}")
        return {"success": False, "error": str(e)}


def _create_drift_baseline(strategy_id: str) -> None:
    """
    Create drift detection baseline from recent market data.

    Called automatically when a strategy is promoted to forward_testing.
    Best-effort — failure does not block the promotion.
    """
    try:
        store = DataStore(read_only=True)
        detector = DriftDetector()

        # Try to load recent data — use SPY as a reasonable broad-market proxy
        # for baseline feature distributions
        df = store.load_ohlcv("SPY", "1D")
        if df is None or len(df) < 60:
            logger.info(
                f"[drift_baseline] Insufficient data for {strategy_id}, skipping baseline"
            )
            return

        # Compute technical indicators
        ti = TechnicalIndicators(df)
        features: dict[str, np.ndarray] = {}

        feature_methods = {
            "rsi_14": lambda: ti.rsi(14),
            "atr_pct": lambda: ti.atr(14) / df["close"] * 100,
            "adx_14": lambda: ti.adx(14),
            "bb_pct": lambda: ti.bollinger_pct_b(20, 2.0),
        }

        for name, method in feature_methods.items():
            try:
                values = method()
                if values is not None:
                    arr = np.asarray(values, dtype=np.float64).ravel()
                    arr = arr[np.isfinite(arr)]
                    if len(arr) > 20:
                        features[name] = arr
            except Exception as exc:
                logger.debug(f"[strategy] drift baseline feature '{name}' computation failed: {exc}")
                continue

        if features:
            detector.set_baseline(strategy_id, features)
            logger.info(
                f"[drift_baseline] Created baseline for {strategy_id}: "
                f"{list(features.keys())} ({len(next(iter(features.values())))} samples)"
            )
        else:
            logger.info(f"[drift_baseline] No features computed for {strategy_id}")

    except Exception as exc:
        logger.debug(f"[drift_baseline] Failed for {strategy_id} (non-critical): {exc}")


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
