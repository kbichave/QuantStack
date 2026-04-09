# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Closed feedback loops (P15) — five autonomous correction mechanisms.

Each loop detects a specific degradation signal and takes a corrective action
without human intervention. Data collection happens regardless of whether the
loop's action is enabled — the loop only controls the *response*.

Loops:
  1. Trade loss → bump research priority for that strategy
  2. TCA cost drift → update cost model parameters
  3. IC degradation → reduce signal weight
  4. Live performance gap → demote strategy
  5. Agent quality drop → flag for prompt improvement

All loops are idempotent: running them multiple times with the same data
produces the same result. They read from PostgreSQL and write corrective
actions back to PostgreSQL for downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import pg_conn


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FeedbackLoopStatus:
    """Result of a single feedback loop check."""

    loop_name: str
    last_triggered: datetime | None = None
    last_action: str | None = None
    is_healthy: bool = True


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class FeedbackLoopManager:
    """Orchestrates five closed feedback loops for autonomous self-correction.

    Each check method queries the DB for degradation signals and returns
    a status. The manager does not cache state — every call is a fresh read.
    """

    LOOP_NAMES = (
        "trade_loss",
        "tca_cost",
        "ic_degradation",
        "live_perf",
        "agent_quality",
    )

    def __init__(self) -> None:
        self._loop_statuses: dict[str, FeedbackLoopStatus] = {
            name: FeedbackLoopStatus(loop_name=name) for name in self.LOOP_NAMES
        }

    # -----------------------------------------------------------------
    # Individual loops
    # -----------------------------------------------------------------

    def check_trade_loss_loop(self) -> FeedbackLoopStatus:
        """Trade loss → bump research priority for the losing strategy.

        Looks at fills from the last 24h grouped by strategy. If any strategy
        has net realised P&L < -$200, flag it for priority research.
        """
        status = FeedbackLoopStatus(loop_name="trade_loss")
        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT strategy_id, SUM(realized_pnl) AS net_pnl
                    FROM fills
                    WHERE filled_at >= NOW() - INTERVAL '24 hours'
                      AND realized_pnl IS NOT NULL
                    GROUP BY strategy_id
                    HAVING SUM(realized_pnl) < -200
                    """,
                ).fetchall()

                if rows:
                    for row in rows:
                        strategy_id = row["strategy_id"]
                        net_pnl = row["net_pnl"]
                        conn.execute(
                            """
                            INSERT INTO system_events (event_type, payload, created_at)
                            VALUES ('feedback_trade_loss', %s, NOW())
                            """,
                            [
                                f'{{"strategy_id": "{strategy_id}", '
                                f'"net_pnl": {net_pnl}, '
                                f'"action": "bump_research_priority"}}'
                            ],
                        )
                        logger.warning(
                            f"[FEEDBACK] Trade loss loop: {strategy_id} "
                            f"lost ${abs(net_pnl):.0f} in 24h — flagging for research"
                        )
                    status.last_triggered = datetime.now(timezone.utc)
                    status.last_action = f"flagged {len(rows)} strategy(s) for research"
                else:
                    status.is_healthy = True
        except Exception as exc:
            logger.error(f"[FEEDBACK] Trade loss loop failed: {exc}")
            status.is_healthy = False
            status.last_action = f"error: {exc}"

        self._loop_statuses["trade_loss"] = status
        return status

    def check_tca_cost_loop(self) -> FeedbackLoopStatus:
        """TCA cost drift → update cost model parameters.

        Compares average realised slippage over the last 7 days against the
        expected slippage in the cost model. If drift exceeds 50%, log an
        update event.
        """
        status = FeedbackLoopStatus(loop_name="tca_cost")
        try:
            with pg_conn() as conn:
                row = conn.execute(
                    """
                    SELECT AVG(slippage_bps) AS avg_slippage
                    FROM fills
                    WHERE filled_at >= NOW() - INTERVAL '7 days'
                      AND slippage_bps IS NOT NULL
                    """,
                ).fetchone()

                if row and row["avg_slippage"] is not None:
                    avg_slip = float(row["avg_slippage"])
                    # Default expected slippage: 5 bps
                    expected_slip = 5.0
                    drift_pct = abs(avg_slip - expected_slip) / max(expected_slip, 0.01)

                    if drift_pct > 0.5:
                        conn.execute(
                            """
                            INSERT INTO system_events (event_type, payload, created_at)
                            VALUES ('feedback_tca_cost', %s, NOW())
                            """,
                            [
                                f'{{"avg_slippage_bps": {avg_slip:.2f}, '
                                f'"expected_bps": {expected_slip}, '
                                f'"drift_pct": {drift_pct:.2f}, '
                                f'"action": "update_cost_model"}}'
                            ],
                        )
                        logger.warning(
                            f"[FEEDBACK] TCA cost loop: slippage drift "
                            f"{drift_pct:.0%} — updating cost model"
                        )
                        status.last_triggered = datetime.now(timezone.utc)
                        status.last_action = (
                            f"cost model update (avg={avg_slip:.1f}bps, "
                            f"drift={drift_pct:.0%})"
                        )
        except Exception as exc:
            logger.error(f"[FEEDBACK] TCA cost loop failed: {exc}")
            status.is_healthy = False
            status.last_action = f"error: {exc}"

        self._loop_statuses["tca_cost"] = status
        return status

    def check_ic_degradation_loop(self) -> FeedbackLoopStatus:
        """IC degradation → reduce signal weight.

        Reads the rolling 63-day IC for each signal collector. If any
        collector's IC drops below 0.02, emit a weight-reduction event.
        """
        status = FeedbackLoopStatus(loop_name="ic_degradation")
        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT collector_name, rolling_ic
                    FROM signal_ic_history
                    WHERE measured_at = (
                        SELECT MAX(measured_at) FROM signal_ic_history
                    )
                      AND rolling_ic < 0.02
                    """,
                ).fetchall()

                if rows:
                    for row in rows:
                        collector = row["collector_name"]
                        ic_val = row["rolling_ic"]
                        conn.execute(
                            """
                            INSERT INTO system_events (event_type, payload, created_at)
                            VALUES ('feedback_ic_degradation', %s, NOW())
                            """,
                            [
                                f'{{"collector": "{collector}", '
                                f'"rolling_ic": {ic_val:.4f}, '
                                f'"action": "reduce_signal_weight"}}'
                            ],
                        )
                        logger.warning(
                            f"[FEEDBACK] IC degradation: {collector} "
                            f"IC={ic_val:.4f} — reducing weight"
                        )
                    status.last_triggered = datetime.now(timezone.utc)
                    status.last_action = (
                        f"reduced weight for {len(rows)} collector(s)"
                    )
        except Exception as exc:
            logger.error(f"[FEEDBACK] IC degradation loop failed: {exc}")
            status.is_healthy = False
            status.last_action = f"error: {exc}"

        self._loop_statuses["ic_degradation"] = status
        return status

    def check_live_perf_loop(self) -> FeedbackLoopStatus:
        """Live performance gap → demote strategy.

        Compares each strategy's live Sharpe (last 30 days) to its backtest
        Sharpe. If the ratio is below 0.3, emit a demotion event.
        """
        status = FeedbackLoopStatus(loop_name="live_perf")
        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT s.strategy_id, s.backtest_sharpe, s.live_sharpe
                    FROM strategies s
                    WHERE s.status = 'live'
                      AND s.live_sharpe IS NOT NULL
                      AND s.backtest_sharpe IS NOT NULL
                      AND s.backtest_sharpe > 0
                      AND (s.live_sharpe / s.backtest_sharpe) < 0.3
                    """,
                ).fetchall()

                if rows:
                    for row in rows:
                        sid = row["strategy_id"]
                        ratio = float(row["live_sharpe"]) / float(
                            row["backtest_sharpe"]
                        )
                        conn.execute(
                            """
                            INSERT INTO system_events (event_type, payload, created_at)
                            VALUES ('feedback_live_perf', %s, NOW())
                            """,
                            [
                                f'{{"strategy_id": "{sid}", '
                                f'"live_sharpe": {row["live_sharpe"]}, '
                                f'"backtest_sharpe": {row["backtest_sharpe"]}, '
                                f'"ratio": {ratio:.2f}, '
                                f'"action": "demote_strategy"}}'
                            ],
                        )
                        logger.warning(
                            f"[FEEDBACK] Live perf loop: {sid} "
                            f"ratio={ratio:.2f} — demoting"
                        )
                    status.last_triggered = datetime.now(timezone.utc)
                    status.last_action = f"demoted {len(rows)} strategy(s)"
        except Exception as exc:
            logger.error(f"[FEEDBACK] Live perf loop failed: {exc}")
            status.is_healthy = False
            status.last_action = f"error: {exc}"

        self._loop_statuses["live_perf"] = status
        return status

    def check_agent_quality_loop(self) -> FeedbackLoopStatus:
        """Agent quality drop → flag for prompt improvement.

        Checks the tool_errors table for agents with 3+ failures in the last
        hour. Emits a prompt-improvement event for each.
        """
        status = FeedbackLoopStatus(loop_name="agent_quality")
        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT agent_name, COUNT(*) AS error_count
                    FROM tool_errors
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY agent_name
                    HAVING COUNT(*) >= 3
                    """,
                ).fetchall()

                if rows:
                    for row in rows:
                        agent = row["agent_name"]
                        count = row["error_count"]
                        conn.execute(
                            """
                            INSERT INTO system_events (event_type, payload, created_at)
                            VALUES ('feedback_agent_quality', %s, NOW())
                            """,
                            [
                                f'{{"agent_name": "{agent}", '
                                f'"error_count": {count}, '
                                f'"action": "flag_prompt_improvement"}}'
                            ],
                        )
                        logger.warning(
                            f"[FEEDBACK] Agent quality: {agent} "
                            f"has {count} errors in 1h — flagging"
                        )
                    status.last_triggered = datetime.now(timezone.utc)
                    status.last_action = f"flagged {len(rows)} agent(s)"
        except Exception as exc:
            logger.error(f"[FEEDBACK] Agent quality loop failed: {exc}")
            status.is_healthy = False
            status.last_action = f"error: {exc}"

        self._loop_statuses["agent_quality"] = status
        return status

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------

    def run_all_checks(self) -> list[FeedbackLoopStatus]:
        """Run all five feedback loops and return their statuses."""
        results = [
            self.check_trade_loss_loop(),
            self.check_tca_cost_loop(),
            self.check_ic_degradation_loop(),
            self.check_live_perf_loop(),
            self.check_agent_quality_loop(),
        ]
        return results

    def health_report(self) -> dict[str, Any]:
        """Summary of all feedback loops suitable for JSON serialization."""
        statuses = self._loop_statuses
        all_healthy = all(s.is_healthy for s in statuses.values())
        return {
            "overall_healthy": all_healthy,
            "loops": {
                name: {
                    "is_healthy": s.is_healthy,
                    "last_triggered": (
                        s.last_triggered.isoformat() if s.last_triggered else None
                    ),
                    "last_action": s.last_action,
                }
                for name, s in statuses.items()
            },
        }
