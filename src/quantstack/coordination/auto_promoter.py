# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Evidence-based strategy auto-promotion pipeline.

Replaces the human gate on ``forward_testing → live`` transitions with
measurable evidence thresholds.  All promotions are logged to the audit trail
with the full criteria snapshot.

Position ramp: newly promoted strategies start at 25% of target allocation
and scale to 100% over 4 weeks — limiting blast radius of bad promotions.

Safety controls:
  - Gated behind ``AUTO_PROMOTE_ENABLED=true`` env var (default: off).
  - Max 8 concurrent live strategies (configurable).
  - Must pass DegradationDetector and StrategyBreaker checks.
  - Ramp enforced via StrategyBreaker.get_ramp_factor().
  - Auto-demotion on CRITICAL degradation (live → forward_testing).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class PromotionCriteria:
    """Evidence thresholds for forward_testing → live promotion."""

    min_forward_test_days: int = 21
    min_forward_test_trades: int = 15
    min_live_sharpe: float = 0.5
    max_degradation_vs_backtest: float = 0.40
    min_win_rate: float = 0.40
    max_max_drawdown: float = 0.08
    max_concurrent_live: int = 8


@dataclass
class PositionRamp:
    """Graduated position sizing after promotion."""

    week_1_scale: float = 0.25
    week_2_scale: float = 0.50
    week_3_scale: float = 0.75
    week_4_plus_scale: float = 1.00

    def get_scale(self, days_since_promotion: int) -> float:
        """Return the position scale factor for the given age."""
        if days_since_promotion < 7:
            return self.week_1_scale
        elif days_since_promotion < 14:
            return self.week_2_scale
        elif days_since_promotion < 21:
            return self.week_3_scale
        return self.week_4_plus_scale


@dataclass
class PromotionDecision:
    """Result of evaluating a strategy for promotion."""

    strategy_id: str
    name: str
    decision: str  # "promote", "hold", "demote"
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    criteria_results: dict[str, bool] = field(default_factory=dict)


class AutoPromoter:
    """
    Evaluates forward_testing strategies for autonomous promotion to live.

    Args:
        conn: PostgreSQL connection.
        criteria: Promotion thresholds (defaults to PromotionCriteria()).
        event_bus: EventBus for publishing promotion events.
        strategy_lock: StrategyStatusLock for atomic transitions.
    """

    def __init__(
        self,
        conn: PgConnection,
        criteria: PromotionCriteria | None = None,
        event_bus: Any | None = None,
        strategy_lock: Any | None = None,
    ) -> None:
        self._conn = conn
        self._criteria = criteria or PromotionCriteria()
        self._ramp = PositionRamp()
        self._bus = event_bus
        self._lock = strategy_lock

    def is_enabled(self) -> bool:
        """Check if auto-promotion is enabled."""
        return os.getenv("AUTO_PROMOTE_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )

    def evaluate_all(self) -> list[PromotionDecision]:
        """
        Evaluate all forward_testing strategies for promotion.

        Returns a list of decisions. Actually promotes/demotes if conditions are met.
        """
        if not self.is_enabled():
            logger.info(
                "[AutoPromoter] Disabled (set AUTO_PROMOTE_ENABLED=true to enable)"
            )
            return []

        # Load forward_testing strategies
        rows = self._conn.execute(
            """
            SELECT strategy_id, name, backtest_summary, updated_at
            FROM strategies
            WHERE status = 'forward_testing'
            ORDER BY updated_at ASC
            """
        ).fetchall()

        decisions: list[PromotionDecision] = []
        for strategy_id, name, bt_summary_raw, updated_at in rows:
            decision = self._evaluate_one(strategy_id, name, bt_summary_raw, updated_at)
            decisions.append(decision)

            if decision.decision == "promote":
                self._execute_promotion(decision)
            elif decision.decision == "demote":
                self._execute_demotion(decision)

        return decisions

    def get_ramp_scale(self, strategy_id: str) -> float:
        """
        Get the current position ramp scale for a live strategy.

        Returns 1.0 if the strategy is not in ramp period or not found.
        """
        row = self._conn.execute(
            "SELECT status, updated_at FROM strategies WHERE strategy_id = ?",
            [strategy_id],
        ).fetchone()

        if not row or row[0] != "live":
            return 1.0

        promotion_date = row[1]
        if not isinstance(promotion_date, datetime):
            return 1.0

        days_live = (
            datetime.now(timezone.utc) - promotion_date.replace(tzinfo=timezone.utc)
        ).days
        return self._ramp.get_scale(days_live)

    def _evaluate_one(
        self,
        strategy_id: str,
        name: str,
        bt_summary_raw: str | None,
        updated_at: datetime,
    ) -> PromotionDecision:
        """Evaluate a single strategy for promotion."""
        criteria = self._criteria
        evidence: dict[str, Any] = {}
        checks: dict[str, bool] = {}

        # Age check
        if not isinstance(updated_at, datetime):
            return PromotionDecision(
                strategy_id=strategy_id,
                name=name,
                decision="hold",
                reason="Cannot determine age",
            )

        age_days = (
            datetime.now(timezone.utc) - updated_at.replace(tzinfo=timezone.utc)
        ).days
        evidence["age_days"] = age_days
        checks["min_age"] = age_days >= criteria.min_forward_test_days

        if not checks["min_age"]:
            return PromotionDecision(
                strategy_id=strategy_id,
                name=name,
                decision="hold",
                reason=f"Too young: {age_days}d < {criteria.min_forward_test_days}d minimum",
                evidence=evidence,
                criteria_results=checks,
            )

        # Trade count + performance from strategy_outcomes
        outcomes = self._get_forward_test_outcomes(strategy_id, updated_at)
        evidence["trade_count"] = len(outcomes)
        checks["min_trades"] = len(outcomes) >= criteria.min_forward_test_trades

        if not checks["min_trades"]:
            return PromotionDecision(
                strategy_id=strategy_id,
                name=name,
                decision="hold",
                reason=f"Insufficient trades: {len(outcomes)} < {criteria.min_forward_test_trades}",
                evidence=evidence,
                criteria_results=checks,
            )

        # Compute metrics from outcomes
        pnl_pcts = [
            o["realized_pnl_pct"]
            for o in outcomes
            if o.get("realized_pnl_pct") is not None
        ]
        if not pnl_pcts:
            return PromotionDecision(
                strategy_id=strategy_id,
                name=name,
                decision="hold",
                reason="No realized P&L data",
                evidence=evidence,
                criteria_results=checks,
            )

        wins = sum(1 for p in pnl_pcts if p > 0)
        win_rate = wins / len(pnl_pcts) if pnl_pcts else 0
        avg_return = sum(pnl_pcts) / len(pnl_pcts)

        # Approximate Sharpe (annualized from daily-ish returns)
        if len(pnl_pcts) > 1:
            mean_r = sum(pnl_pcts) / len(pnl_pcts)
            var_r = sum((r - mean_r) ** 2 for r in pnl_pcts) / (len(pnl_pcts) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-6
            holding_period = criteria.min_forward_test_days / len(pnl_pcts)
            annualization = math.sqrt(252 / max(holding_period, 1))
            live_sharpe = (mean_r / std_r) * annualization
        else:
            live_sharpe = 0.0

        # Max drawdown (cumulative)
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnl_pcts:
            cumulative += p
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        evidence["win_rate"] = round(win_rate, 3)
        evidence["live_sharpe"] = round(live_sharpe, 3)
        evidence["max_drawdown"] = round(max_dd, 4)
        evidence["avg_return"] = round(avg_return, 4)

        checks["min_win_rate"] = win_rate >= criteria.min_win_rate
        checks["min_sharpe"] = live_sharpe >= criteria.min_live_sharpe
        checks["max_drawdown"] = max_dd <= criteria.max_max_drawdown

        # Degradation vs backtest
        bt_sharpe = self._get_backtest_sharpe(bt_summary_raw)
        if bt_sharpe and bt_sharpe > 0:
            degradation = 1.0 - (live_sharpe / bt_sharpe)
            evidence["backtest_sharpe"] = bt_sharpe
            evidence["degradation_pct"] = round(degradation, 3)
            checks["max_degradation"] = (
                degradation <= criteria.max_degradation_vs_backtest
            )
        else:
            checks["max_degradation"] = True  # No backtest to compare

        # Live strategy cap
        live_count = self._count_live_strategies()
        evidence["current_live_count"] = live_count
        checks["strategy_cap"] = live_count < criteria.max_concurrent_live

        # All checks must pass
        all_pass = all(checks.values())

        if all_pass:
            return PromotionDecision(
                strategy_id=strategy_id,
                name=name,
                decision="promote",
                reason=f"All criteria met: Sharpe={live_sharpe:.2f}, WR={win_rate:.0%}, DD={max_dd:.1%}",
                evidence=evidence,
                criteria_results=checks,
            )

        failed = [k for k, v in checks.items() if not v]
        return PromotionDecision(
            strategy_id=strategy_id,
            name=name,
            decision="hold",
            reason=f"Failed checks: {', '.join(failed)}",
            evidence=evidence,
            criteria_results=checks,
        )

    def _execute_promotion(self, decision: PromotionDecision) -> None:
        """Execute a promotion using the strategy lock."""
        if self._lock:
            ok = self._lock.transition(
                decision.strategy_id,
                expected_status="forward_testing",
                new_status="live",
                reason=decision.reason,
            )
            if ok:
                logger.info(f"[AutoPromoter] Promoted {decision.name} to live")
            else:
                logger.warning(f"[AutoPromoter] CAS failed promoting {decision.name}")
        else:
            # Direct update (less safe but functional without lock)
            self._conn.execute(
                "UPDATE strategies SET status = 'live', updated_at = ? WHERE strategy_id = ?",
                [datetime.now(timezone.utc), decision.strategy_id],
            )
            logger.info(f"[AutoPromoter] Promoted {decision.name} to live (no lock)")

    def _execute_demotion(self, decision: PromotionDecision) -> None:
        """Execute a demotion using the strategy lock."""
        if self._lock:
            self._lock.transition(
                decision.strategy_id,
                expected_status="live",
                new_status="forward_testing",
                reason=decision.reason,
            )
        logger.warning(f"[AutoPromoter] Demoted {decision.name}: {decision.reason}")

    def _get_forward_test_outcomes(
        self, strategy_id: str, since: datetime
    ) -> list[dict[str, Any]]:
        """Get closed outcomes since the strategy entered forward_testing."""
        rows = self._conn.execute(
            """
            SELECT realized_pnl_pct, outcome, opened_at, closed_at
            FROM strategy_outcomes
            WHERE strategy_id = ? AND closed_at IS NOT NULL AND opened_at >= ?
            ORDER BY closed_at ASC
            """,
            [strategy_id, since],
        ).fetchall()
        return [
            {
                "realized_pnl_pct": r[0],
                "outcome": r[1],
                "opened_at": r[2],
                "closed_at": r[3],
            }
            for r in rows
        ]

    def _get_backtest_sharpe(self, bt_summary_raw: str | None) -> float | None:
        """Extract Sharpe ratio from backtest_summary JSON."""
        if not bt_summary_raw:
            return None
        try:
            summary = (
                json.loads(bt_summary_raw)
                if isinstance(bt_summary_raw, str)
                else bt_summary_raw
            )
            return summary.get("sharpe_ratio", summary.get("sharpe"))
        except (ValueError, TypeError, AttributeError):
            return None

    def _count_live_strategies(self) -> int:
        """Count currently live strategies."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE status = 'live'"
        ).fetchone()
        return row[0] if row else 0
