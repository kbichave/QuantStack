# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Capital allocation engine for the /meta orchestrator.

Inputs: current regime + list of eligible strategies from registry.
Output: AllocationPlan with per-strategy capital weights.

Allocation logic:
  1. Filter strategies whose regime_affinity includes current regime
  2. Rank by: live Sharpe (if available) > backtest Sharpe, weighted by recency
  3. Allocate capital: top strategies get larger share
  4. forward_testing strategies capped at 10% allocation
  5. Total allocation <= max_gross_exposure from risk_gate
  6. Signal conflicts resolved conservatively

This module is pure logic — no DB access.  The tools layer
calls this with pre-fetched data.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


def compute_allocation(
    regime: str,
    regime_confidence: float,
    strategies: list[dict[str, Any]],
    max_gross_exposure_pct: float = 1.5,
    forward_testing_cap: float = 0.10,
) -> dict[str, Any]:
    """
    Compute capital allocations for eligible strategies.

    Args:
        regime: Current regime label (e.g., "trending_up").
        regime_confidence: Confidence in the regime classification (0-1).
        strategies: List of strategy dicts from the registry.
            Each must have: strategy_id, name, status, regime_affinity,
            backtest_summary, risk_params.
        max_gross_exposure_pct: Maximum total allocation (from risk_gate).
        forward_testing_cap: Max allocation for forward_testing strategies.

    Returns:
        Dict with allocations, total_allocated_pct, warnings.
    """
    eligible = []
    warnings = []

    for strat in strategies:
        status = strat.get("status", "draft")
        if status not in ("live", "forward_testing"):
            continue

        affinity = strat.get("regime_affinity") or {}
        if isinstance(affinity, str):
            try:
                affinity = json.loads(affinity)
            except (ValueError, TypeError):
                affinity = {}

        # Find the best-matching regime key
        regime_score = _match_regime(regime, affinity)
        if regime_score <= 0:
            continue

        # Get ranking Sharpe (prefer live stats, fall back to backtest)
        bt = strat.get("backtest_summary") or {}
        if isinstance(bt, str):
            try:
                bt = json.loads(bt)
            except (ValueError, TypeError):
                bt = {}

        sharpe = bt.get("sharpe_ratio", 0.0)

        eligible.append(
            {
                "strategy_id": strat["strategy_id"],
                "strategy_name": strat.get("name", ""),
                "status": status,
                "regime_score": regime_score,
                "ranking_sharpe": sharpe,
                "risk_params": strat.get("risk_params") or {},
                "symbols": _extract_symbols(bt),
            }
        )

    if not eligible:
        return {
            "allocations": [],
            "total_allocated_pct": 0.0,
            "unallocated_pct": 1.0,
            "warnings": [f"No eligible strategies for regime '{regime}'"],
            "conflicts_resolved": 0,
        }

    # Sort by regime_score * ranking_sharpe (composite rank)
    eligible.sort(
        key=lambda s: s["regime_score"] * max(0.01, s["ranking_sharpe"]),
        reverse=True,
    )

    # Allocate capital
    allocations = []
    total = 0.0
    max_total = max_gross_exposure_pct  # cap as fraction

    for strat in eligible:
        if total >= max_total:
            break

        # Base allocation from regime score
        base = strat["regime_score"] * 0.15  # max 15% per strategy at score=1.0

        # Cap forward_testing strategies
        if strat["status"] == "forward_testing":
            base = min(base, forward_testing_cap)

        # Cap by strategy's own risk_params
        risk_params = strat.get("risk_params", {})
        if isinstance(risk_params, str):
            try:
                risk_params = json.loads(risk_params)
            except (ValueError, TypeError):
                risk_params = {}

        max_pos = risk_params.get(
            "max_position_pct", risk_params.get("position_pct", 0.10)
        )
        base = min(base, max_pos)

        # Don't exceed total cap
        base = min(base, max_total - total)

        if base <= 0.005:  # Skip < 0.5% allocations
            continue

        mode = "paper" if strat["status"] == "forward_testing" else "live"

        allocations.append(
            {
                "strategy_id": strat["strategy_id"],
                "strategy_name": strat["strategy_name"],
                "capital_pct": round(base, 4),
                "symbols": strat["symbols"],
                "mode": mode,
                "regime_score": round(strat["regime_score"], 2),
                "ranking_sharpe": round(strat["ranking_sharpe"], 4),
                "reasoning": (
                    f"Regime score {strat['regime_score']:.2f} x "
                    f"Sharpe {strat['ranking_sharpe']:.2f} = "
                    f"rank {strat['regime_score'] * strat['ranking_sharpe']:.3f}. "
                    f"{'Capped at forward_testing limit.' if strat['status'] == 'forward_testing' else ''}"
                ),
            }
        )
        total += base

    # Scale down if regime confidence is low
    if regime_confidence < 0.6:
        scale = regime_confidence / 0.6
        for a in allocations:
            a["capital_pct"] = round(a["capital_pct"] * scale, 4)
        total *= scale
        warnings.append(
            f"Regime confidence {regime_confidence:.0%} < 60%: allocations scaled to {scale:.0%}"
        )

    return {
        "allocations": allocations,
        "total_allocated_pct": round(total, 4),
        "unallocated_pct": round(1.0 - total, 4),
        "warnings": warnings,
        "conflicts_resolved": 0,
    }


def resolve_conflicts(
    proposed_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Resolve signal conflicts across strategies for the same symbols.

    Rules:
      - Same symbol, different directions:
        One confidence > 0.85 and other < 0.65 -> keep high confidence
        Both > 0.75 -> SKIP (genuine disagreement)
        Otherwise -> keep higher confidence
      - Same symbol, same direction:
        Use more conservative entry (worse price for us)
        Position size = min of individual allocations

    Args:
        proposed_trades: List of trade dicts with:
            symbol, action, confidence, strategy_id, capital_pct,
            and optionally limit_price, position_size

    Returns:
        Dict with resolved_trades, resolutions, conflicts_count.
    """
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for trade in proposed_trades:
        by_symbol[trade.get("symbol", "")].append(trade)

    resolved_trades = []
    resolutions = []
    conflicts = 0

    for symbol, trades in by_symbol.items():
        if len(trades) == 1:
            resolved_trades.append(trades[0])
            resolutions.append(
                {
                    "symbol": symbol,
                    "action": "keep",
                    "reasoning": "No conflict — single strategy",
                    "kept_strategy": trades[0].get("strategy_id"),
                    "skipped_strategies": [],
                }
            )
            continue

        # Group by direction
        buys = [t for t in trades if t.get("action") == "buy"]
        sells = [t for t in trades if t.get("action") == "sell"]

        if buys and sells:
            # Conflicting directions
            conflicts += 1
            best_buy = max(buys, key=lambda t: t.get("confidence", 0))
            best_sell = max(sells, key=lambda t: t.get("confidence", 0))

            buy_conf = best_buy.get("confidence", 0)
            sell_conf = best_sell.get("confidence", 0)

            if buy_conf > 0.85 and sell_conf < 0.65:
                resolved_trades.append(best_buy)
                resolutions.append(
                    {
                        "symbol": symbol,
                        "action": "keep",
                        "reasoning": f"Buy confidence {buy_conf:.0%} >> sell {sell_conf:.0%}",
                        "kept_strategy": best_buy.get("strategy_id"),
                        "skipped_strategies": [t.get("strategy_id") for t in sells],
                    }
                )
            elif sell_conf > 0.85 and buy_conf < 0.65:
                resolved_trades.append(best_sell)
                resolutions.append(
                    {
                        "symbol": symbol,
                        "action": "keep",
                        "reasoning": f"Sell confidence {sell_conf:.0%} >> buy {buy_conf:.0%}",
                        "kept_strategy": best_sell.get("strategy_id"),
                        "skipped_strategies": [t.get("strategy_id") for t in buys],
                    }
                )
            elif buy_conf > 0.75 and sell_conf > 0.75:
                # Genuine disagreement — skip
                resolutions.append(
                    {
                        "symbol": symbol,
                        "action": "skip",
                        "reasoning": (
                            f"Genuine disagreement: buy {buy_conf:.0%} vs sell {sell_conf:.0%}. "
                            "Both > 75% — skipping to avoid false signal."
                        ),
                        "kept_strategy": None,
                        "skipped_strategies": [t.get("strategy_id") for t in trades],
                    }
                )
            else:
                # Keep higher confidence
                winner = best_buy if buy_conf >= sell_conf else best_sell
                loser_side = sells if buy_conf >= sell_conf else buys
                resolved_trades.append(winner)
                resolutions.append(
                    {
                        "symbol": symbol,
                        "action": "keep",
                        "reasoning": f"Higher confidence wins: {max(buy_conf, sell_conf):.0%} vs {min(buy_conf, sell_conf):.0%}",
                        "kept_strategy": winner.get("strategy_id"),
                        "skipped_strategies": [
                            t.get("strategy_id") for t in loser_side
                        ],
                    }
                )
        else:
            # Same direction — merge conservatively
            all_trades = buys or sells
            best = max(all_trades, key=lambda t: t.get("confidence", 0))
            # Conservative: use smallest position size
            min_pct = min(t.get("capital_pct", 0.025) for t in all_trades)
            merged = {**best, "capital_pct": min_pct}
            resolved_trades.append(merged)
            resolutions.append(
                {
                    "symbol": symbol,
                    "action": "adjust",
                    "reasoning": (
                        f"Same direction ({best.get('action')}): "
                        f"merged {len(all_trades)} signals, conservative sizing at {min_pct:.1%}"
                    ),
                    "kept_strategy": best.get("strategy_id"),
                    "skipped_strategies": [],
                }
            )

    return {
        "resolved_trades": resolved_trades,
        "resolutions": resolutions,
        "conflicts_count": conflicts,
    }


def _match_regime(current_regime: str, affinity: dict[str, float]) -> float:
    """
    Match current regime against a strategy's regime_affinity dict.

    Handles partial matches: "trending_up" matches keys like
    "trending", "trending_up", "up".
    """
    if not affinity:
        return 0.0

    # Direct match
    if current_regime in affinity:
        return affinity[current_regime]

    # Partial match: split on underscore
    parts = current_regime.lower().split("_")
    best = 0.0
    for key, score in affinity.items():
        key_parts = key.lower().split("_")
        if any(p in key_parts for p in parts):
            best = max(best, score * 0.8)  # 80% weight for partial match

    return best


def _extract_symbols(backtest_summary: dict) -> list[str]:
    """Extract symbol list from a backtest summary."""
    if not backtest_summary:
        return []
    symbol = backtest_summary.get("symbol")
    if symbol:
        return [symbol]
    return backtest_summary.get("symbols", [])
