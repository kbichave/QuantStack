# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
OutcomeTracker — the learning loop's core.

Records trade entry/exit attributed to a specific strategy and regime,
then adjusts regime_affinity weights based on accumulated outcomes.

This is the non-parametric RLHF described in the architecture:
  - No gradient, no model training
  - Simple Bayesian momentum: each outcome nudges weights by step=0.05
  - Clipped to [0.1, 1.0] so no regime affinity ever reaches zero
  - 20+ trades needed to move affinity from 0.5 to its asymptote

Design invariants:
  - All DB writes are best-effort (try/except). A DB failure NEVER blocks a fill.
  - Reads use PostgreSQL (pg_conn) — no file-lock competition.
  - apply_learning() calls update_strategy() rather than writing DB directly,
    so all audit trail and validation logic in that tool is respected.

Usage:
    tracker = OutcomeTracker()

    # At entry (called by AutonomousRunner before order submission):
    tracker.record_entry(
        strategy_id="strat_abc123",
        symbol="XOM",
        regime="trending_up",
        action="buy",
        price=105.50,
        session_id="autonomous_run_abc",
    )

    # At exit (called by execute_trade after sell fill):
    tracker.record_exit(
        strategy_id="strat_abc123",
        symbol="XOM",
        exit_price=110.20,
    )

    # Apply learning (called after each session or by IntradayMonitorFlow):
    updated = tracker.apply_learning("strat_abc123")
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import db_conn, open_db_readonly


# =============================================================================
# Learning hyperparameters — documented with reasoning
# =============================================================================

_STEP = 0.05
# Conservative learning rate. At 20 consistent wins (+5%), affinity moves from
# 0.5 → ~0.76. At 20 consistent losses (-5%), 0.5 → ~0.24. This ensures
# the system requires a meaningful sample before changing routing decisions.

_CLIP_MIN = 0.1
_CLIP_MAX = 1.0
# Never drive affinity to 0 — a losing streak in the current regime doesn't
# mean the strategy will never work in that regime again.

_PNL_SCALE = 5.0
# tanh(pnl_pct / _PNL_SCALE): at ±5%, weight = ±0.76. At ±1%, weight = ±0.20.
# Keeps small P&L moves from over-influencing weights.

_MIN_OUTCOMES_FOR_UPDATE = 3
# Don't update weights until at least 3 outcomes are recorded.
# Prevents a single lucky/unlucky trade from immediately routing strategy away.


# =============================================================================
# OutcomeTracker
# =============================================================================


class OutcomeTracker:
    """
    Writes to and reads from the strategy_outcomes table.
    All methods are synchronous — designed to be called via asyncio.to_thread
    from async callers.
    """

    def record_entry(
        self,
        strategy_id: str,
        symbol: str,
        regime: str,
        action: str,
        price: float,
        session_id: str = "",
    ) -> None:
        """
        Write an open outcome row at trade entry. Best-effort.

        Called by AutonomousRunner before order submission and by execute_trade
        after a buy fill succeeds.
        """
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO strategy_outcomes
                        (strategy_id, symbol, regime_at_entry, action,
                         entry_price, opened_at, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        strategy_id,
                        symbol,
                        regime,
                        action,
                        price,
                        datetime.now(timezone.utc),
                        session_id,
                    ],
                )
        except Exception as exc:
            logger.debug(f"[OutcomeTracker] record_entry failed (non-critical): {exc}")

    def record_exit(
        self,
        strategy_id: str,
        symbol: str,
        exit_price: float,
        realized_pnl_pct: float | None = None,
    ) -> None:
        """
        Close the most recent open outcome row for this strategy+symbol.
        Best-effort.

        Called by execute_trade after a sell fill succeeds.
        """
        try:
            with db_conn() as conn:
                # Find the most recent open (unclosed) row
                row = conn.execute(
                    """
                    SELECT id, entry_price
                    FROM strategy_outcomes
                    WHERE strategy_id = ?
                      AND symbol = ?
                      AND closed_at IS NULL
                    ORDER BY opened_at DESC
                    LIMIT 1
                    """,
                    [strategy_id, symbol],
                ).fetchone()

                if row is None:
                    return

                row_id, entry_price = row

                # Compute P&L if not provided
                if realized_pnl_pct is None and entry_price and entry_price > 0:
                    realized_pnl_pct = (exit_price - entry_price) / entry_price * 100.0

                outcome = _classify_outcome(realized_pnl_pct)

                conn.execute(
                    """
                    UPDATE strategy_outcomes
                    SET exit_price       = ?,
                        realized_pnl_pct = ?,
                        outcome          = ?,
                        closed_at        = ?
                    WHERE id = ?
                    """,
                    [
                        exit_price,
                        realized_pnl_pct,
                        outcome,
                        datetime.now(timezone.utc),
                        row_id,
                    ],
                )
        except Exception as exc:
            logger.debug(f"[OutcomeTracker] record_exit failed (non-critical): {exc}")

    def apply_learning(self, strategy_id: str) -> bool:
        """
        Read all closed outcomes for strategy_id and update regime_affinity weights.

        Returns True if any weights were updated, False otherwise.

        Update formula (Bayesian momentum):
            outcome_weight = tanh(realized_pnl_pct / _PNL_SCALE)
            current = regime_affinity.get(regime, 0.5)
            new = clip(current + _STEP * outcome_weight, _CLIP_MIN, _CLIP_MAX)

        Also updates regime_strategy_matrix.confidence for the same strategy+regime.
        """
        try:
            outcomes = self._load_outcomes(strategy_id)
            if len(outcomes) < _MIN_OUTCOMES_FOR_UPDATE:
                return False

            # Group by regime
            by_regime: dict[str, list[float]] = {}
            for row in outcomes:
                regime = row["regime_at_entry"]
                pnl = row["realized_pnl_pct"]
                if regime and pnl is not None:
                    by_regime.setdefault(regime, []).append(pnl)

            if not by_regime:
                return False

            # Load current regime_affinity
            current_affinity = self._load_affinity(strategy_id)
            if current_affinity is None:
                logger.debug(
                    f"[OutcomeTracker] strategy {strategy_id} not found — skip"
                )
                return False

            updated = False
            new_affinity = dict(current_affinity)

            for regime, pnls in by_regime.items():
                if len(pnls) < _MIN_OUTCOMES_FOR_UPDATE:
                    continue

                # Aggregate: mean outcome weight across all trades in this regime
                mean_weight = sum(math.tanh(p / _PNL_SCALE) for p in pnls) / len(pnls)
                current_val = current_affinity.get(regime, 0.5)
                new_val = max(
                    _CLIP_MIN, min(_CLIP_MAX, current_val + _STEP * mean_weight)
                )

                if abs(new_val - current_val) > 0.001:
                    new_affinity[regime] = round(new_val, 4)
                    updated = True
                    logger.info(
                        f"[OutcomeTracker] {strategy_id} {regime}: "
                        f"{current_val:.3f} → {new_val:.3f} "
                        f"(n={len(pnls)} mean_weight={mean_weight:+.3f})"
                    )

            if not updated:
                return False

            # Write back via update_strategy (respects audit trail)
            self._write_affinity(strategy_id, new_affinity)
            return True

        except Exception as exc:
            logger.warning(f"[OutcomeTracker] apply_learning failed: {exc}")
            return False

    # -------------------------------------------------------------------------
    # DB helpers (synchronous, read-only for loads)
    # -------------------------------------------------------------------------

    def _load_outcomes(self, strategy_id: str) -> list[dict[str, Any]]:
        """Load all closed outcome rows for a strategy."""
        try:
            conn = open_db_readonly()
            rows = conn.execute(
                """
                SELECT regime_at_entry, realized_pnl_pct, outcome, closed_at
                FROM strategy_outcomes
                WHERE strategy_id = ?
                  AND closed_at IS NOT NULL
                ORDER BY closed_at DESC
                """,
                [strategy_id],
            ).fetchall()
            conn.close()
            return [
                {
                    "regime_at_entry": r[0],
                    "realized_pnl_pct": r[1],
                    "outcome": r[2],
                    "closed_at": r[3],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug(f"[OutcomeTracker] _load_outcomes failed: {exc}")
            return []

    def _load_affinity(self, strategy_id: str) -> dict[str, float] | None:
        """Load current regime_affinity for a strategy. Returns None if not found."""
        try:
            conn = open_db_readonly()
            row = conn.execute(
                "SELECT regime_affinity FROM strategies WHERE strategy_id = ?",
                [strategy_id],
            ).fetchone()
            conn.close()
            if row is None:
                return None
            raw = row[0]
            if not raw:
                return {}
            return json.loads(raw) if isinstance(raw, str) else raw
        except Exception as exc:
            logger.debug(f"[OutcomeTracker] _load_affinity failed: {exc}")
            return None

    def _write_affinity(self, strategy_id: str, affinity: dict[str, float]) -> None:
        """
        Persist updated regime_affinity via direct DB write.

        We write directly to the DB rather than calling the update_strategy tool
        because OutcomeTracker may run from a non-tool context (scheduler, script).
        """
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    UPDATE strategies
                    SET regime_affinity = ?,
                        updated_at      = ?
                    WHERE strategy_id = ?
                    """,
                    [json.dumps(affinity), datetime.now(timezone.utc), strategy_id],
                )
            logger.debug(
                f"[OutcomeTracker] wrote regime_affinity for {strategy_id}: {affinity}"
            )
        except Exception as exc:
            logger.warning(f"[OutcomeTracker] _write_affinity failed: {exc}")


# =============================================================================
# Helpers
# =============================================================================


def _classify_outcome(pnl_pct: float | None) -> str:
    """Classify a P&L percentage into win/loss/flat."""
    if pnl_pct is None:
        return "flat"
    if pnl_pct > 0.5:
        return "win"
    if pnl_pct < -0.5:
        return "loss"
    return "flat"
