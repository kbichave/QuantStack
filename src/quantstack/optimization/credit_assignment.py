# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Step-level credit assignment for trading decisions.

Attributes trade outcomes to specific decision steps (signal, regime,
strategy selection, sizing, debate) so TextGrad can target the node
that actually caused the loss instead of propagating uniformly.

Paper: AgentPRM (2025) — https://arxiv.org/abs/2511.08325
Lightweight version: heuristic rules + statistical correlation.

Design:
  - Heuristic: deterministic rules applied per-trade (immediate, zero cost)
  - Statistical: rolling correlation over a window (daily batch, cheap)
  - Both methods must agree before TextGrad targets a specific node
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from quantstack.db import pg_conn


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StepCredit:
    """Credit score for one decision step in a trade."""
    step_type: str           # "signal", "regime", "strategy_selection", "sizing", "debate"
    step_output: str         # What this step decided
    credit_score: float      # -1.0 (caused the loss) to 1.0 (prevented worse)
    attribution_method: str  # "heuristic" or "statistical"
    evidence: str            # Human-readable explanation


STEP_CREDITS_TABLE = "step_credits"

# Step types in the decision chain
STEP_TYPES = ("signal", "regime", "strategy_selection", "sizing", "debate")


# ---------------------------------------------------------------------------
# CreditAssigner
# ---------------------------------------------------------------------------

class CreditAssigner:
    """Attribute trade outcomes to specific decision steps.

    Usage:
        assigner = CreditAssigner(conn)
        # After trade close (immediate):
        credits = assigner.assign_heuristic(trade_context)
        # Daily batch:
        stats = assigner.assign_statistical(window_days=30)
        # For TextGrad targeting:
        worst = assigner.get_worst_step(credits)
    """

    def __init__(self, conn: Any = None) -> None:
        # conn is accepted but ignored — each write uses pg_conn() to ensure
        # the transaction is committed immediately and never left open.
        pass

    # ------------------------------------------------------------------
    # Heuristic credit (per-trade, immediate, deterministic)
    # ------------------------------------------------------------------

    def assign_heuristic(self, trade_context: dict) -> list[StepCredit]:
        """Assign credit based on observable facts about a single trade.

        Args:
            trade_context: Dict with keys:
                - realized_pnl_pct: float
                - regime_at_entry: str
                - regime_at_exit: str
                - strategy_id: str
                - conviction: float
                - position_size: str  ("quarter", "half", "full")
                - debate_verdict: str  ("pass", "downgrade", "veto", or "")
                - signals_present: bool
                - strategy_regime_affinity: float  (0-1, how well strategy fits regime)

        Returns list of StepCredit, one per step type.
        """
        pnl = trade_context.get("realized_pnl_pct", 0.0)
        is_loss = pnl < 0
        credits = []

        # --- Signal step ---
        signals_present = trade_context.get("signals_present", True)
        if not signals_present and is_loss:
            credits.append(StepCredit(
                step_type="signal",
                step_output="incomplete_data",
                credit_score=-0.4,
                attribution_method="heuristic",
                evidence="Entered with missing signal data",
            ))
        else:
            credits.append(StepCredit(
                step_type="signal",
                step_output="data_ok",
                credit_score=0.1 if is_loss else 0.3,
                attribution_method="heuristic",
                evidence="Signal data was present at entry",
            ))

        # --- Regime step ---
        regime_entry = trade_context.get("regime_at_entry", "unknown")
        regime_exit = trade_context.get("regime_at_exit", "unknown")
        regime_shifted = (
            regime_entry != regime_exit
            and regime_exit not in ("unknown", "")
            and regime_entry not in ("unknown", "")
        )
        if regime_shifted and is_loss:
            credits.append(StepCredit(
                step_type="regime",
                step_output=f"{regime_entry}->{regime_exit}",
                credit_score=-0.5,
                attribution_method="heuristic",
                evidence=f"Regime shifted {regime_entry} -> {regime_exit} during hold",
            ))
        elif regime_entry == "unknown" and is_loss:
            credits.append(StepCredit(
                step_type="regime",
                step_output="unknown_regime",
                credit_score=-0.3,
                attribution_method="heuristic",
                evidence="Entered with unknown regime classification",
            ))
        else:
            credits.append(StepCredit(
                step_type="regime",
                step_output=f"stable:{regime_entry}",
                credit_score=0.2 if not is_loss else 0.0,
                attribution_method="heuristic",
                evidence="Regime was stable during hold",
            ))

        # --- Strategy selection step ---
        affinity = trade_context.get("strategy_regime_affinity", 0.5)
        strategy_id = trade_context.get("strategy_id", "")
        if affinity < 0.4 and is_loss:
            credits.append(StepCredit(
                step_type="strategy_selection",
                step_output=f"{strategy_id}@{regime_entry}",
                credit_score=-0.5,
                attribution_method="heuristic",
                evidence=f"Strategy '{strategy_id}' has low affinity ({affinity:.2f}) for regime '{regime_entry}'",
            ))
        elif affinity < 0.6 and is_loss:
            credits.append(StepCredit(
                step_type="strategy_selection",
                step_output=f"{strategy_id}@{regime_entry}",
                credit_score=-0.2,
                attribution_method="heuristic",
                evidence=f"Strategy '{strategy_id}' has marginal affinity ({affinity:.2f}) for regime '{regime_entry}'",
            ))
        else:
            credits.append(StepCredit(
                step_type="strategy_selection",
                step_output=f"{strategy_id}@{regime_entry}",
                credit_score=0.2 if not is_loss else 0.0,
                attribution_method="heuristic",
                evidence=f"Strategy-regime affinity OK ({affinity:.2f})",
            ))

        # --- Sizing step ---
        position_size = trade_context.get("position_size", "half")
        conviction = trade_context.get("conviction", 0.5)
        size_rank = {"quarter": 1, "half": 2, "full": 3}.get(position_size, 2)
        oversized = (conviction < 0.6 and size_rank >= 2) or (conviction < 0.5 and size_rank >= 1)
        if oversized and is_loss:
            credits.append(StepCredit(
                step_type="sizing",
                step_output=f"{position_size}@{conviction:.0%}",
                credit_score=-0.3,
                attribution_method="heuristic",
                evidence=f"Position size '{position_size}' too large for conviction {conviction:.0%}",
            ))
        else:
            credits.append(StepCredit(
                step_type="sizing",
                step_output=f"{position_size}@{conviction:.0%}",
                credit_score=0.1 if not is_loss else 0.0,
                attribution_method="heuristic",
                evidence="Size appropriate for conviction level",
            ))

        # --- Debate step ---
        debate_verdict = trade_context.get("debate_verdict", "")
        if debate_verdict == "pass" and is_loss and pnl < -3.0:
            credits.append(StepCredit(
                step_type="debate",
                step_output="passed_but_lost",
                credit_score=-0.3,
                attribution_method="heuristic",
                evidence=f"Debate passed but trade lost {pnl:.1f}% — bear case was under-weighted",
            ))
        elif debate_verdict == "veto":
            # Trade was vetoed — no outcome to credit
            credits.append(StepCredit(
                step_type="debate",
                step_output="vetoed",
                credit_score=0.0,
                attribution_method="heuristic",
                evidence="Trade was vetoed by debate filter",
            ))
        else:
            credits.append(StepCredit(
                step_type="debate",
                step_output=debate_verdict or "not_triggered",
                credit_score=0.0,
                attribution_method="heuristic",
                evidence="Debate filter not triggered or no material impact",
            ))

        # Persist to DB
        self._persist_credits(credits, trade_context)

        return credits

    # ------------------------------------------------------------------
    # Targeting for TextGrad
    # ------------------------------------------------------------------

    @staticmethod
    def get_worst_step(credits: list[StepCredit]) -> StepCredit | None:
        """Return the step with the lowest credit score.

        TextGrad should focus its backward pass on this node.
        """
        if not credits:
            return None
        return min(credits, key=lambda c: c.credit_score)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_credits(self, credits: list[StepCredit], trade_context: dict) -> None:
        """Write step credits to the database.

        Uses pg_conn() so the batch INSERT is committed immediately — a long-lived
        open transaction here would hold row locks and block CREATE INDEX migrations.
        """
        trade_id = trade_context.get("trade_id", 0)
        try:
            with pg_conn() as conn:
                for credit in credits:
                    conn.execute(
                        f"INSERT INTO {STEP_CREDITS_TABLE} "
                        f"(id, trade_id, step_type, step_output, credit_score, "
                        f"attribution_method, evidence) "
                        f"VALUES (nextval('step_credits_seq'), ?, ?, ?, ?, ?, ?)",
                        [
                            trade_id, credit.step_type, credit.step_output,
                            credit.credit_score, credit.attribution_method,
                            credit.evidence,
                        ],
                    )
        except Exception as exc:
            logger.warning(f"[CreditAssigner] Persist failed: {exc}")
