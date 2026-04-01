# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
IntradaySignalEvaluator — bridges IncrementalFeatures to UnifiedOrder.

This is the ``SignalEvaluator`` callback consumed by ``AsyncExecutionLoop``.
On every bar it checks active intraday strategies against the current
feature vector and returns a UnifiedOrder if an entry or exit signal fires.

Rule format (same as strategy registry entry_rules / exit_rules):
    {"indicator": "rsi", "condition": "below", "value": 30}
    {"indicator": "ema_cross", "condition": "above", "value": 0}

Supported conditions: above, below, crosses_above, crosses_below, between.
Indicators map to IncrementalFeatures field names.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, time
from typing import Any

import pytz
from loguru import logger

from quantstack.data.streaming.incremental_features import IncrementalFeatures
from quantstack.core.execution.unified_models import UnifiedOrder

from quantstack.intraday.position_manager import IntradayPositionManager

ET = pytz.timezone("US/Eastern")


class IntradaySignalEvaluator:
    """Evaluate intraday entry/exit signals from incremental features.

    Args:
        strategies: Active intraday strategies (list of dicts from DB).
            Each must have: name, entry_rules, exit_rules, parameters,
            and optionally risk_params.position_size_pct.
        position_manager: IntradayPositionManager for trade count + exit tracking.
        entry_cutoff_et: No new entries after this time (ET), e.g. "15:30".
        max_trades_per_day: Hard cap on total trades in one session.
        default_quantity: Default share quantity when strategy doesn't specify.
    """

    def __init__(
        self,
        strategies: list[dict[str, Any]],
        position_manager: IntradayPositionManager,
        entry_cutoff_et: str = "15:30",
        max_trades_per_day: int = 50,
        default_quantity: int = 100,
    ) -> None:
        self._strategies = strategies
        self._pm = position_manager
        h, m = entry_cutoff_et.split(":")
        self._entry_cutoff = time(int(h), int(m))
        self._max_trades = max_trades_per_day
        self._default_qty = default_quantity

        # Track previous feature values per symbol for crossover detection
        self._prev_features: dict[str, dict[str, float]] = {}

    async def __call__(self, features: IncrementalFeatures) -> UnifiedOrder | None:
        """Called by AsyncExecutionLoop on every bar.

        Returns a UnifiedOrder to submit, or None to hold.
        """
        if not features.is_warm:
            return None

        symbol = features.symbol
        feat_dict = _features_to_dict(features)

        # Check for exit signals first (exits always allowed, even after cutoff)
        pos = self._pm._tracker.get_position(symbol)
        if pos and pos.quantity != 0:
            exit_order = self._check_exit_signals(symbol, feat_dict, pos)
            if exit_order:
                return exit_order

        # Entry gate checks
        now_et = datetime.now(ET).time()
        if now_et >= self._entry_cutoff:
            return None
        if self._pm.trades_today >= self._max_trades:
            return None
        if self._pm.is_flattened:
            return None
        # Don't enter if already in a position for this symbol
        if pos and pos.quantity != 0:
            return None

        # Check entry signals
        entry_order = self._check_entry_signals(symbol, feat_dict, features)

        # Store current features for next-bar crossover detection
        self._prev_features[symbol] = feat_dict

        return entry_order

    # ── Entry signal evaluation ─────────────────────────────────────────────

    def _check_entry_signals(
        self,
        symbol: str,
        feat_dict: dict[str, float],
        features: IncrementalFeatures,
    ) -> UnifiedOrder | None:
        """Check all strategies for entry signals on this symbol."""
        prev = self._prev_features.get(symbol, {})

        for strat in self._strategies:
            entry_rules = strat.get("entry_rules", [])
            if not entry_rules:
                continue

            # All rules must pass (AND logic)
            if all(_evaluate_scalar_rule(r, feat_dict, prev) for r in entry_rules):
                side = strat.get("parameters", {}).get("direction", "buy")
                qty = self._resolve_quantity(strat)

                logger.info(
                    f"[SignalEval] ENTRY {symbol} {side} {qty} "
                    f"strategy={strat.get('name', '?')} "
                    f"rsi={feat_dict.get('rsi', 0):.1f} ema_cross={feat_dict.get('ema_cross', 0):.2f}"
                )

                return UnifiedOrder(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type="market",
                    time_in_force="day",
                    client_order_id=f"intraday_{strat.get('name', 'unknown')}_{symbol}",
                )

        return None

    # ── Exit signal evaluation ──────────────────────────────────────────────

    def _check_exit_signals(
        self,
        symbol: str,
        feat_dict: dict[str, float],
        pos: Any,
    ) -> UnifiedOrder | None:
        """Check all strategies for exit signals on an open position."""
        prev = self._prev_features.get(symbol, {})

        for strat in self._strategies:
            exit_rules = strat.get("exit_rules", [])
            if not exit_rules:
                continue

            if all(_evaluate_scalar_rule(r, feat_dict, prev) for r in exit_rules):
                side = "sell" if pos.quantity > 0 else "buy"
                qty = abs(pos.quantity)

                logger.info(
                    f"[SignalEval] EXIT {symbol} {side} {qty} "
                    f"strategy={strat.get('name', '?')}"
                )

                return UnifiedOrder(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type="market",
                    time_in_force="day",
                    client_order_id=f"intraday_exit_{strat.get('name', 'unknown')}_{symbol}",
                )

        return None

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _resolve_quantity(self, strategy: dict) -> float:
        """Get position size from strategy risk_params or use default."""
        risk_params = strategy.get("risk_params", {})
        if isinstance(risk_params, str):
            return self._default_qty
        return risk_params.get("quantity", self._default_qty)


# ---------------------------------------------------------------------------
# Scalar rule evaluation (operates on single feature dict, not DataFrame)
# ---------------------------------------------------------------------------


def _features_to_dict(features: IncrementalFeatures) -> dict[str, float]:
    """Convert IncrementalFeatures to a flat dict for rule evaluation."""
    d = asdict(features)
    # Remove non-numeric fields
    for key in ("symbol", "timestamp", "timeframe", "is_warm"):
        d.pop(key, None)
    return {k: v for k, v in d.items() if isinstance(v, (int, float)) and v is not None}


def _evaluate_scalar_rule(
    rule: dict[str, Any],
    current: dict[str, float],
    prev: dict[str, float],
) -> bool:
    """Evaluate a single rule against the current feature values.

    Rule format: {"indicator": "rsi", "condition": "below", "value": 30}

    Supported conditions:
      above, below, crosses_above, crosses_below, between
    """
    indicator = rule.get("indicator", "")
    condition = rule.get("condition", "")
    value = rule.get("value")

    cur_val = current.get(indicator)
    if cur_val is None:
        return False

    if condition == "above":
        return cur_val > value
    elif condition == "below":
        return cur_val < value
    elif condition == "crosses_above":
        prev_val = prev.get(indicator)
        if prev_val is None:
            return False
        return prev_val <= value and cur_val > value
    elif condition == "crosses_below":
        prev_val = prev.get(indicator)
        if prev_val is None:
            return False
        return prev_val >= value and cur_val < value
    elif condition == "between":
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return value[0] <= cur_val <= value[1]
        return False

    return False
