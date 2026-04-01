# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Shared tool implementations — importable by other tool modules without
creating circular dependencies through mcp.server.

This module does NOT import mcp.server, _app.py, or @mcp.tool() — it only
contains the raw business logic that multiple MCP tools need to share.

Leaf in the dependency graph: imports only from quantstack.db,
quantstack.mcp._state, quantstack.mcp._helpers, quantstack.core.*.
"""

from __future__ import annotations

import asyncio
import json
import json as _json
import uuid
from typing import Any

import numpy as np
from loguru import logger

from quantstack.core.backtesting.engine import BacktestConfig, BacktestEngine
from quantstack.db import pg_conn
from quantstack.mcp._helpers import _get_reader
from quantstack.mcp._state import _serialize, live_db_or_error
from quantstack.strategies.signal_generator import (
    fetch_price_data as _fetch_price_data,
    generate_signals_from_rules as _generate_signals_from_rules,
)


async def get_strategy_impl(
    strategy_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Core logic for get_strategy — callable from other tool modules."""
    _, err = live_db_or_error()
    if err:
        return err
    try:
        with pg_conn() as conn:
            if strategy_id:
                row = conn.execute(
                    "SELECT * FROM strategies WHERE strategy_id = ?", [strategy_id]
                ).fetchone()
            elif name:
                row = conn.execute(
                    "SELECT * FROM strategies WHERE name = ?", [name]
                ).fetchone()
            else:
                return {"success": False, "error": "Provide strategy_id or name"}

        if row is None:
            searched = f"strategy_id={strategy_id!r}" if strategy_id else f"name={name!r}"
            return {
                "success": False,
                "error": f"Strategy not found ({searched}). Use list_strategies to see available strategies.",
            }

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
            "symbol",
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


async def register_strategy_impl(
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
    """Core logic for register_strategy — callable from other tool modules."""
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
        logger.info(f"[_impl] Registered strategy {strategy_id}: {name}")
        return {"success": True, "strategy_id": strategy_id, "status": "draft"}
    except Exception as e:
        logger.error(f"[_impl] register_strategy failed: {e}")
        return {"success": False, "error": str(e)}


async def run_backtest_impl(
    strategy_id: str,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.10,
    commission: float = 1.0,
    slippage_pct: float = 0.001,
) -> dict[str, Any]:
    """Core logic for run_backtest — callable from other tool modules."""
    _, err = live_db_or_error()
    if err:
        return err

    try:
        # 1. Load strategy
        strat_result = await get_strategy_impl(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {
                "success": False,
                "error": strat_result.get("error", "Strategy not found"),
            }
        strat = strat_result["strategy"]

        # 2. Fetch price data
        price_data = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_price_data, symbol, start_date, end_date
        )
        if price_data is None or price_data.empty:
            return {"success": False, "error": f"No price data available for {symbol}"}

        # 3. Generate signals from rules
        entry_rules = strat.get("entry_rules", [])
        exit_rules = strat.get("exit_rules", [])
        parameters = strat.get("parameters", {})

        # Inject symbol into parameters for indicator computation and feature enrichment
        parameters["symbol"] = symbol

        if not entry_rules:
            return {"success": False, "error": "Strategy has no entry_rules"}

        signals = _generate_signals_from_rules(
            price_data, entry_rules, exit_rules, parameters
        )

        # 4. Run backtest
        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            commission_per_trade=commission,
            slippage_pct=slippage_pct,
        )
        engine = BacktestEngine(config=config)
        result = engine.run(signals, price_data)

        # 5. Compute additional metrics
        calmar = 0.0
        if result.max_drawdown > 0:
            calmar = (result.total_return / 100.0) / (result.max_drawdown / 100.0)

        avg_pnl = 0.0
        if result.trades:
            avg_pnl = np.mean([t["pnl"] for t in result.trades])

        summary = {
            "symbol": symbol,
            "total_trades": result.total_trades,
            "win_rate": round(result.win_rate, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "max_drawdown": round(result.max_drawdown, 2),
            "total_return_pct": round(result.total_return, 2),
            "profit_factor": round(result.profit_factor, 4),
            "calmar_ratio": round(calmar, 4),
            "avg_trade_pnl": round(avg_pnl, 2),
            "start_date": (
                str(price_data.index[0].date())
                if hasattr(price_data.index[0], "date")
                else str(price_data.index[0])
            ),
            "end_date": (
                str(price_data.index[-1].date())
                if hasattr(price_data.index[-1], "date")
                else str(price_data.index[-1])
            ),
            "bars_tested": len(price_data),
            "trades": result.trades,
        }

        # 6. Persist summary on strategy record
        with pg_conn() as conn:
            conn.execute(
                "UPDATE strategies SET backtest_summary = ?, status = CASE WHEN status = 'draft' THEN 'backtested' ELSE status END, updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
                [_json.dumps(summary), strategy_id],
            )

        # Return metrics only — omit trades list to keep response within token limits.
        summary_return = {k: v for k, v in summary.items() if k != "trades"}
        return {"success": True, **summary_return, "strategy_id": strategy_id}

    except Exception as e:
        logger.error(f"[_impl] run_backtest failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "strategy_id": strategy_id,
            "symbol": symbol,
        }
