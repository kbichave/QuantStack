"""
A/B testing framework for ML model champion/challenger evaluation.

Orchestrates shadow model evaluation and automatic swap/rollback decisions
based on rolling information coefficient (IC) comparison.

Rules:
  - Challenger IC > champion IC for 2 consecutive weeks → swap
  - Champion rolling IC < 50% of its backtest IC → rollback to previous
  - Shadow predictions must not affect the live trading path

Delegates to model_registry.py for persistence and state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
from loguru import logger

from quantstack.db import db_conn
from quantstack.ml.model_registry import (
    ModelVersion,
    promote_challenger,
    query_champion,
    retire_model,
)


@dataclass
class ABTestResult:
    """Result of a weekly A/B evaluation for a strategy."""

    strategy_id: str
    champion_id: str | None
    challenger_id: str | None
    champion_weekly_ics: list[float] = field(default_factory=list)
    challenger_weekly_ics: list[float] = field(default_factory=list)
    consecutive_wins: int = 0
    should_swap: bool = False
    should_rollback: bool = False
    reason: str = ""


class ABTestManager:
    """Manages shadow model evaluation and swap/rollback logic."""

    CONSECUTIVE_WINS_TO_SWAP = 2
    ROLLBACK_IC_RATIO = 0.5  # rollback if IC drops below 50% of backtest IC

    def record_shadow_prediction(
        self,
        model_id: str,
        symbol: str,
        predicted: float,
        realized: float | None,
        prediction_date: date | None = None,
    ) -> None:
        """Record a shadow prediction for later A/B comparison.

        Args:
            model_id: The model that produced the prediction.
            symbol: Ticker symbol.
            predicted: Predicted probability (0-1).
            realized: Actual forward return (None if not yet known).
            prediction_date: Date of prediction (defaults to today).
        """
        prediction_date = prediction_date or date.today()
        now = datetime.now(timezone.utc)

        with db_conn() as conn:
            conn.execute(
                "INSERT INTO model_shadow_predictions "
                "(model_id, symbol, prediction_date, prediction, realized_return, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                (model_id, symbol, prediction_date, predicted, realized, now),
            )

    def evaluate_weekly(self, strategy_id: str) -> ABTestResult:
        """Compare champion vs best challenger over recent weekly windows.

        Computes weekly IC (Spearman rank correlation between predictions
        and realized returns) for the last 4 weeks. If the challenger
        outperforms the champion for CONSECUTIVE_WINS_TO_SWAP consecutive
        weeks, recommends a swap.

        Returns:
            ABTestResult with swap/rollback recommendation.
        """
        champion = query_champion(strategy_id)
        if champion is None:
            return ABTestResult(
                strategy_id=strategy_id,
                champion_id=None,
                challenger_id=None,
                reason="no_champion",
            )

        # Find the best challenger
        with db_conn() as conn:
            challengers = conn.fetchall(
                "SELECT DISTINCT model_id FROM model_shadow_predictions "
                "WHERE model_id IN ("
                "  SELECT model_id FROM model_registry "
                "  WHERE strategy_id = %s AND status = 'challenger'"
                ") ORDER BY model_id DESC LIMIT 1",
                (strategy_id,),
            )

        if not challengers:
            return ABTestResult(
                strategy_id=strategy_id,
                champion_id=champion.model_id,
                challenger_id=None,
                reason="no_challenger_predictions",
            )

        challenger_id = challengers[0]["model_id"]

        # Compute weekly ICs for both models (last 4 weeks)
        champion_ics = self._compute_weekly_ics(champion.model_id, n_weeks=4)
        challenger_ics = self._compute_weekly_ics(challenger_id, n_weeks=4)

        # Count consecutive wins (most recent first)
        consecutive_wins = 0
        min_weeks = min(len(champion_ics), len(challenger_ics))
        for i in range(min_weeks):
            if challenger_ics[i] > champion_ics[i]:
                consecutive_wins += 1
            else:
                break

        should_swap = consecutive_wins >= self.CONSECUTIVE_WINS_TO_SWAP

        reason = "no_action"
        if should_swap:
            reason = (
                f"challenger {challenger_id} outperformed champion for "
                f"{consecutive_wins} consecutive weeks"
            )

        return ABTestResult(
            strategy_id=strategy_id,
            champion_id=champion.model_id,
            challenger_id=challenger_id,
            champion_weekly_ics=champion_ics,
            challenger_weekly_ics=challenger_ics,
            consecutive_wins=consecutive_wins,
            should_swap=should_swap,
            reason=reason,
        )

    def check_rollback(self, strategy_id: str) -> tuple[bool, str]:
        """Check if the champion should be rolled back due to IC degradation.

        If the champion's rolling IC (last 20 trading days) drops below
        50% of its backtest IC, recommend rollback.

        Returns:
            (should_rollback, reason)
        """
        champion = query_champion(strategy_id)
        if champion is None:
            return False, "no_champion"

        if champion.backtest_ic <= 0:
            return False, "no_baseline_ic"

        # Compute rolling IC for champion
        with db_conn() as conn:
            rows = conn.fetchall(
                "SELECT prediction, realized_return FROM model_shadow_predictions "
                "WHERE model_id = %s AND realized_return IS NOT NULL "
                "ORDER BY prediction_date DESC LIMIT 20",
                (champion.model_id,),
            )

        if len(rows) < 10:
            return False, "insufficient_data"

        predictions = np.array([r["prediction"] for r in rows])
        realized = np.array([r["realized_return"] for r in rows])

        # Spearman IC
        from scipy.stats import spearmanr
        ic = float(spearmanr(predictions, realized).statistic)
        if np.isnan(ic):
            ic = 0.0

        threshold = champion.backtest_ic * self.ROLLBACK_IC_RATIO
        if ic < threshold:
            return True, (
                f"rolling IC {ic:.4f} < {self.ROLLBACK_IC_RATIO:.0%} of "
                f"backtest IC {champion.backtest_ic:.4f}"
            )

        return False, "ic_healthy"

    def execute_swap(self, strategy_id: str, challenger_id: str) -> None:
        """Promote challenger to champion."""
        promote_challenger(challenger_id)
        logger.info(
            f"[ab_testing] Swapped champion for {strategy_id}: "
            f"promoted {challenger_id}"
        )

    def execute_rollback(self, strategy_id: str) -> bool:
        """Roll back to the previous champion (most recently retired).

        Returns True if rollback succeeded, False if no previous model found.
        """
        with db_conn() as conn:
            row = conn.fetchone(
                "SELECT model_id FROM model_registry "
                "WHERE strategy_id = %s AND status = 'retired' "
                "ORDER BY retired_at DESC LIMIT 1",
                (strategy_id,),
            )

        if row is None:
            logger.warning(f"[ab_testing] No retired model to rollback to for {strategy_id}")
            return False

        previous_id = row["model_id"]

        # Retire current champion, promote previous
        champion = query_champion(strategy_id)
        if champion:
            retire_model(champion.model_id)

        with db_conn() as conn:
            conn.execute(
                "UPDATE model_registry SET status = 'champion', "
                "promoted_at = %s WHERE model_id = %s",
                (datetime.now(timezone.utc), previous_id),
            )

        logger.info(
            f"[ab_testing] Rolled back {strategy_id}: restored {previous_id}"
        )
        return True

    def _compute_weekly_ics(
        self, model_id: str, n_weeks: int = 4
    ) -> list[float]:
        """Compute per-week IC for a model (most recent first).

        Returns list of weekly IC values, ordered from most recent to oldest.
        """
        with db_conn() as conn:
            rows = conn.fetchall(
                "SELECT prediction_date, prediction, realized_return "
                "FROM model_shadow_predictions "
                "WHERE model_id = %s AND realized_return IS NOT NULL "
                "ORDER BY prediction_date DESC "
                f"LIMIT {n_weeks * 5}",  # ~5 trading days per week
                (model_id,),
            )

        if len(rows) < 5:
            return []

        # Group by week number
        from collections import defaultdict
        weekly: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            d = row["prediction_date"]
            week_key = d.isocalendar()[1]
            weekly[week_key].append((row["prediction"], row["realized_return"]))

        # Compute IC per week (most recent first)
        week_keys = sorted(weekly.keys(), reverse=True)[:n_weeks]
        ics: list[float] = []
        for wk in week_keys:
            pairs = weekly[wk]
            if len(pairs) < 3:
                ics.append(0.0)
                continue
            preds = np.array([p[0] for p in pairs])
            rets = np.array([p[1] for p in pairs])
            if preds.std() == 0 or rets.std() == 0:
                ics.append(0.0)
                continue
            from scipy.stats import spearmanr
            ic = float(spearmanr(preds, rets).statistic)
            ics.append(ic if not np.isnan(ic) else 0.0)

        return ics
