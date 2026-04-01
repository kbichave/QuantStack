# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Walk-forward validation service — callable without MCP imports.

Uses WalkForwardValidator (L3) and DataStore (L2) directly, with strategy
data loaded from PostgreSQL via quantstack.db (L1).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd

from loguru import logger

from quantstack.core.backtesting.engine import BacktestConfig, BacktestEngine
from quantstack.core.validation.purged_cv import WalkForwardValidator
from quantstack.data.storage import DataStore
from quantstack.db import open_db_readonly


def _load_strategy(strategy_id: str) -> dict[str, Any] | None:
    """Load strategy record from DB (read-only)."""
    try:
        conn = open_db_readonly()
        row = conn.execute(
            "SELECT strategy_id, name, parameters, entry_rules, exit_rules "
            "FROM strategies WHERE strategy_id = ?",
            [strategy_id],
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "strategy_id": row[0],
            "name": row[1],
            "parameters": json.loads(row[2]) if isinstance(row[2], str) else (row[2] or {}),
            "entry_rules": json.loads(row[3]) if isinstance(row[3], str) else (row[3] or []),
            "exit_rules": json.loads(row[4]) if isinstance(row[4], str) else (row[4] or []),
        }
    except Exception as exc:
        logger.debug(f"[walkforward_service] _load_strategy failed: {exc}")
        return None


async def run_walkforward(
    strategy_id: str,
    symbol: str,
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 504,
    use_purged_cv: bool = True,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """
    Run walk-forward validation for a strategy on a symbol.

    Loads strategy and price data directly from DB/DataStore,
    then runs WalkForwardValidator. No MCP dependency.

    Returns:
        WalkForwardResult dict, or None on failure.
    """
    try:
        strat = await asyncio.to_thread(_load_strategy, strategy_id)
        if strat is None:
            return {"success": False, "error": f"Strategy {strategy_id} not found"}

        entry_rules = strat.get("entry_rules", [])
        if not entry_rules:
            return {"success": False, "error": "Strategy has no entry_rules"}

        store = DataStore(read_only=True)
        df = await asyncio.to_thread(store.load_ohlcv, symbol, "1D")
        if df is not None and not df.empty:
            df = df[df.index >= pd.Timestamp("2010-01-01")]
            logger.info(
                f"Walk-forward: floored data at 2010-01-01, {len(df)} bars remaining"
            )
        if df is None or len(df) < min_train_size + test_size:
            return {
                "success": False,
                "error": f"Insufficient data for {symbol} ({len(df) if df is not None else 0} bars)",
            }

        validator = WalkForwardValidator(
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=min_train_size,
        )

        result = await asyncio.to_thread(
            validator.validate,
            data=df,
            entry_rules=entry_rules,
            exit_rules=strat.get("exit_rules", []),
            parameters=strat.get("parameters", {}),
        )

        return {
            "success": True,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "n_splits": n_splits,
            "result": result,
        }

    except Exception as exc:
        logger.warning(f"[walkforward_service] run_walkforward failed: {exc}")
        return {"success": False, "error": str(exc)}
