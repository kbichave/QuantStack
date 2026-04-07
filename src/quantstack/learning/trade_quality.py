# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Trade quality aggregation — rolling averages from trade_quality_scores.

The TradeEvaluator writes 6-dimension LLM scores to trade_quality_scores
for each closed trade. This module provides read-side aggregation for
downstream consumers (daily_plan, supervisor reports).
"""

from __future__ import annotations

from loguru import logger

from quantstack.db import db_conn

_QUALITY_DIMENSIONS = [
    "execution_quality",
    "thesis_accuracy",
    "risk_management",
    "timing_quality",
    "sizing_quality",
    "overall_score",
]

_MIN_TRADES_FOR_SUMMARY = 5


def get_trade_quality_summary(
    strategy_id: str | None = None,
    window: int = 30,
) -> dict | None:
    """
    Compute rolling averages per quality dimension over the last ``window`` scored trades.

    Args:
        strategy_id: If provided, filter to a specific strategy (via closed_trades join).
        window: Number of most recent scored trades to aggregate.

    Returns:
        {
            "dimensions": {"execution_quality": 6.2, "thesis_accuracy": 7.1, ...},
            "weakest": "execution_quality",
            "weakest_score": 6.2,
            "trade_count": 28,
        }
        Returns None if fewer than 5 scored trades exist.
    """
    try:
        with db_conn() as conn:
            if strategy_id:
                conn.execute(
                    """
                    SELECT tqs.execution_quality, tqs.thesis_accuracy,
                           tqs.risk_management, tqs.timing_quality,
                           tqs.sizing_quality, tqs.overall_score
                    FROM trade_quality_scores tqs
                    JOIN closed_trades ct ON tqs.trade_id = ct.id
                    WHERE ct.strategy_id = %s
                    ORDER BY tqs.scored_at DESC
                    LIMIT %s
                    """,
                    [strategy_id, window],
                )
            else:
                conn.execute(
                    """
                    SELECT execution_quality, thesis_accuracy,
                           risk_management, timing_quality,
                           sizing_quality, overall_score
                    FROM trade_quality_scores
                    ORDER BY scored_at DESC
                    LIMIT %s
                    """,
                    [window],
                )
            rows = conn.fetchall()

        if len(rows) < _MIN_TRADES_FOR_SUMMARY:
            return None

        # Compute averages per dimension
        dimensions: dict[str, float] = {}
        for dim in _QUALITY_DIMENSIONS:
            values = [row[dim] for row in rows if row[dim] is not None]
            dimensions[dim] = round(sum(values) / len(values), 2) if values else 0.0

        # Find weakest dimension (excluding overall_score)
        scoreable = {k: v for k, v in dimensions.items() if k != "overall_score"}
        weakest = min(scoreable, key=scoreable.get)  # type: ignore[arg-type]

        return {
            "dimensions": dimensions,
            "weakest": weakest,
            "weakest_score": scoreable[weakest],
            "trade_count": len(rows),
        }

    except Exception as exc:
        logger.warning(f"[TradeQuality] get_trade_quality_summary failed: {exc}")
        return None
