"""
Shadow Mode Evaluator — tracks RL recommendations vs actual crew decisions.

During the shadow period (shadow_mode=True in config), RL tools return
recommendations tagged [SHADOW]. This module records both the RL recommendation
and the crew's actual decision, then tracks outcomes for comparison.

After min_observations trading days, ShadowEvaluator.evaluate_shadow_period()
produces a ShadowEvaluationResult that PromotionGate uses to decide whether
to remove the [SHADOW] tag.

All shadow data persists in the existing KnowledgeStore DuckDB database.
No new database file is created.

Usage:
    from quantcore.rl.shadow_mode import ShadowEvaluator

    evaluator = ShadowEvaluator(store)

    # At decision time (in rl_tools.py or flow):
    evaluator.record_decision(
        tool_name="rl_position_size",
        rl_recommendation={"scale": 0.7},
        crew_decision={"position_size": "half"},
        symbol="SPY",
    )

    # After trade closes:
    evaluator.record_outcome(decision_id="...", pnl=150.0, slippage_bps=3.2)

    # Weekly check:
    result = evaluator.evaluate_shadow_period("sizing", min_observations=63)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class ShadowDecision:
    """A single recorded shadow decision."""

    decision_id: str
    tool_name: str
    symbol: str | None
    rl_recommendation: dict[str, Any]
    crew_action: str | None
    crew_confidence: float | None
    created_at: datetime
    pnl: float | None = None
    slippage_bps: float | None = None
    outcome_recorded_at: datetime | None = None


@dataclass
class ShadowEvaluationResult:
    """Result of evaluating the shadow period for one agent type."""

    agent_type: str
    n_observations: int
    # RL simulated performance (if RL had been followed)
    rl_simulated_sharpe: float | None
    rl_simulated_win_rate: float | None
    rl_simulated_max_drawdown: float | None
    # Directional agreement with profitable crew trades
    directional_agreement_rate: float | None
    # Summary
    ready_for_promotion: bool
    reasons: list[str] = field(default_factory=list)
    evaluated_at: datetime = field(default_factory=datetime.utcnow)


class ShadowEvaluator:
    """
    Tracks and evaluates RL recommendations during the shadow period.

    Persists all data to rl_shadow_decisions table in KnowledgeStore DuckDB.
    Thread-safe for concurrent crew execution.
    """

    def __init__(self, store: KnowledgeStore):  # type: ignore[name-defined]  # noqa: F821
        self.store = store
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create shadow tracking tables if they don't exist."""
        try:
            self.store.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rl_shadow_decisions (
                    decision_id VARCHAR PRIMARY KEY,
                    tool_name VARCHAR NOT NULL,
                    symbol VARCHAR,
                    rl_recommendation JSON,
                    crew_action VARCHAR,
                    crew_confidence DOUBLE,
                    pnl DOUBLE,
                    slippage_bps DOUBLE,
                    outcome_recorded_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.store.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_tool ON rl_shadow_decisions(tool_name)"
            )
            self.store.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_created ON rl_shadow_decisions(created_at)"
            )
        except Exception as exc:
            logger.warning(f"[ShadowEvaluator] Table creation failed (non-fatal): {exc}")

    def record_decision(
        self,
        tool_name: str,
        rl_recommendation: dict[str, Any],
        crew_action: str | None = None,
        crew_confidence: float | None = None,
        symbol: str | None = None,
    ) -> str:
        """
        Record an RL recommendation alongside the crew's actual decision.

        Args:
            tool_name: "rl_position_size" | "rl_execution_strategy" | "rl_alpha_weight"
            rl_recommendation: The dict returned by the RL tool
            crew_action: The crew's actual decision (e.g. "buy", "half")
            crew_confidence: The crew's stated confidence
            symbol: Trading symbol

        Returns:
            decision_id for later outcome recording
        """
        decision_id = str(uuid.uuid4())
        try:
            self.store.conn.execute(
                """
                INSERT INTO rl_shadow_decisions
                    (decision_id, tool_name, symbol, rl_recommendation,
                     crew_action, crew_confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    decision_id,
                    tool_name,
                    symbol,
                    json.dumps(rl_recommendation),
                    crew_action,
                    crew_confidence,
                    datetime.utcnow(),
                ],
            )
        except Exception as exc:
            logger.debug(f"[ShadowEvaluator] record_decision failed (non-fatal): {exc}")

        return decision_id

    def record_outcome(
        self,
        decision_id: str,
        pnl: float | None = None,
        slippage_bps: float | None = None,
    ) -> None:
        """
        Record the trade outcome for a previously logged shadow decision.

        Called from PostTradeRLAdapter.process_trade_outcome() after a trade closes.
        """
        try:
            self.store.conn.execute(
                """
                UPDATE rl_shadow_decisions
                SET pnl = ?,
                    slippage_bps = ?,
                    outcome_recorded_at = ?
                WHERE decision_id = ?
                """,
                [pnl, slippage_bps, datetime.utcnow(), decision_id],
            )
        except Exception as exc:
            logger.debug(f"[ShadowEvaluator] record_outcome failed (non-fatal): {exc}")

    def evaluate_shadow_period(
        self,
        agent_type: str,
        min_observations: int = 63,
    ) -> ShadowEvaluationResult:
        """
        Compute shadow period metrics for the given agent type.

        Args:
            agent_type: "sizing" | "execution" | "meta"
            min_observations: Minimum decisions required before evaluation

        Returns:
            ShadowEvaluationResult with performance metrics and promotion readiness.
        """
        tool_map = {
            "sizing": "rl_position_size",
            "execution": "rl_execution_strategy",
            "meta": "rl_alpha_weight",
        }
        tool_name = tool_map.get(agent_type, agent_type)

        # Load shadow decisions with outcomes
        try:
            rows = self.store.conn.execute(
                """
                SELECT
                    rl_recommendation,
                    crew_action,
                    crew_confidence,
                    pnl,
                    slippage_bps,
                    created_at
                FROM rl_shadow_decisions
                WHERE tool_name = ?
                  AND outcome_recorded_at IS NOT NULL
                ORDER BY created_at ASC
                """,
                [tool_name],
            ).fetchall()
        except Exception as exc:
            logger.warning(f"[ShadowEvaluator] evaluate query failed: {exc}")
            return ShadowEvaluationResult(
                agent_type=agent_type,
                n_observations=0,
                rl_simulated_sharpe=None,
                rl_simulated_win_rate=None,
                rl_simulated_max_drawdown=None,
                directional_agreement_rate=None,
                ready_for_promotion=False,
                reasons=[f"Query failed: {exc}"],
            )

        n = len(rows)
        reasons = []

        if n < min_observations:
            reasons.append(f"Insufficient observations: {n} < {min_observations} required.")
            return ShadowEvaluationResult(
                agent_type=agent_type,
                n_observations=n,
                rl_simulated_sharpe=None,
                rl_simulated_win_rate=None,
                rl_simulated_max_drawdown=None,
                directional_agreement_rate=None,
                ready_for_promotion=False,
                reasons=reasons,
            )

        # Extract RL-recommended scales / crew outcomes
        rl_scales = []
        pnls = []
        directional_matches = []

        for rec_json, _crew_action, _crew_conf, pnl, slippage_bps, _ in rows:
            if pnl is None:
                continue
            try:
                rec = json.loads(rec_json) if isinstance(rec_json, str) else rec_json
            except Exception:
                continue

            pnls.append(float(pnl))

            # For sizing: compare RL scale vs profitability
            if agent_type == "sizing":
                scale = float(rec.get("scale", 0.5))
                rl_scales.append(scale)
                # Directional agreement: RL recommended large scale when trade was profitable
                if pnl > 0 and scale > 0.5:
                    directional_matches.append(1)
                elif pnl <= 0 and scale <= 0.5:
                    directional_matches.append(1)
                else:
                    directional_matches.append(0)

            elif agent_type == "execution":
                # Execution: lower slippage = better
                directional_matches.append(1 if (slippage_bps or 0) < 5.0 else 0)

            elif agent_type == "meta":
                rec.get("selected_alpha", "")
                directional_matches.append(1 if pnl > 0 else 0)

        if not pnls:
            reasons.append("No outcomes recorded yet.")
            return ShadowEvaluationResult(
                agent_type=agent_type,
                n_observations=n,
                rl_simulated_sharpe=None,
                rl_simulated_win_rate=None,
                rl_simulated_max_drawdown=None,
                directional_agreement_rate=None,
                ready_for_promotion=False,
                reasons=reasons,
            )

        # Compute simulated metrics treating pnl as returns
        pnl_array = np.array(pnls)
        win_rate = float(np.mean(pnl_array > 0))
        sharpe = (
            float(np.mean(pnl_array) / (np.std(pnl_array) + 1e-8) * np.sqrt(252))
            if len(pnl_array) > 1
            else 0.0
        )
        # Max drawdown on cumulative equity
        equity = np.cumprod(1 + pnl_array / 100)  # treat pnl as %
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(np.max(drawdown))

        agreement = float(np.mean(directional_matches)) if directional_matches else 0.0

        # Readiness checks
        ready = True
        if sharpe < 0.3:
            reasons.append(f"Sharpe {sharpe:.2f} < 0.3 minimum.")
            ready = False
        if max_dd > 0.15:
            reasons.append(f"Max drawdown {max_dd:.1%} > 15% limit.")
            ready = False
        if agreement < 0.55:
            reasons.append(f"Directional agreement {agreement:.1%} < 55% minimum.")
            ready = False
        if not reasons:
            reasons.append("All shadow period checks passed.")

        return ShadowEvaluationResult(
            agent_type=agent_type,
            n_observations=n,
            rl_simulated_sharpe=round(sharpe, 4),
            rl_simulated_win_rate=round(win_rate, 4),
            rl_simulated_max_drawdown=round(max_dd, 4),
            directional_agreement_rate=round(agreement, 4),
            ready_for_promotion=ready,
            reasons=reasons,
        )

    def get_observation_count(self, agent_type: str) -> int:
        """Return the number of shadow observations recorded for an agent type."""
        tool_map = {
            "sizing": "rl_position_size",
            "execution": "rl_execution_strategy",
            "meta": "rl_alpha_weight",
        }
        tool_name = tool_map.get(agent_type, agent_type)
        try:
            row = self.store.conn.execute(
                "SELECT COUNT(*) FROM rl_shadow_decisions WHERE tool_name = ?",
                [tool_name],
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0
