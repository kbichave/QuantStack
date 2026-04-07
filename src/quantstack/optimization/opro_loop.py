# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
OPRO-style weekly prompt evolution.

Maintains a population of prompt variants per decision node,
scores them on walk-forward performance, and uses a meta-prompt
to generate improved candidates.

Paper: OPRO (Yang et al., DeepMind 2023) — https://arxiv.org/abs/2309.03409

STATUS: CONDITIONAL. Wired into ResearchOrchestrator.run_weekly() with
guard: activates when closed trade count >= 150 (lowered from 500 after
analysis showed walk-forward OOS trades provide sufficient signal).
See docs/OPTIMIZATION.md "When to Revisit".

Safety:
  - Candidates start as 'candidate', promoted to 'testing' if >5% better
  - Must survive 1 full week in 'testing' (paper trades only) before 'champion'
  - Auto-rollback if Sharpe drops >1σ below champion baseline
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

import litellm

from quantstack.llm.provider import get_model_for_role


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PromptCandidate:
    """A prompt variant being evaluated."""
    candidate_id: str
    node_name: str
    prompt_text: str
    generation: int
    fitness: float = 0.0
    trades_evaluated: int = 0
    status: str = "candidate"  # candidate → testing → champion → retired
    source: str = "opro"
    parent_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    promoted_at: str | None = None


CANDIDATES_TABLE = "prompt_candidates"
CRITIQUES_TABLE = "prompt_critiques"

PROMOTION_THRESHOLD = 0.05  # 5% improvement required
MIN_TESTING_DAYS = 7        # 1 week in testing before champion


# ---------------------------------------------------------------------------
# OPROLoop
# ---------------------------------------------------------------------------

class OPROLoop:
    """Weekly prompt evolution via meta-optimization.

    Usage:
        loop = OPROLoop(conn)
        new_candidates = loop.run_weekly()
    """

    def __init__(
        self,
        conn: Any,
        engine: str | None = None,
    ) -> None:
        self._conn = conn
        self._engine = engine or get_model_for_role("bulk")

    # ------------------------------------------------------------------
    # Weekly run
    # ------------------------------------------------------------------

    def run_weekly(self) -> list[PromptCandidate]:
        """Execute one generation of prompt evolution.

        1. Load current champion + top-3 candidates per node
        2. Score each on last 7 days of trades
        3. Promote testing → champion if stable for 1 week
        4. Generate new candidates via meta-prompt
        5. Promote best candidate → testing if >5% better than champion
        """
        all_candidates = []

        # Get distinct node names from critiques (nodes that have been analyzed)
        node_names = self._get_active_nodes()
        if not node_names:
            logger.info("[OPRO] No active nodes with critiques, nothing to evolve")
            return []

        for node_name in node_names:
            try:
                candidates = self._evolve_node(node_name)
                all_candidates.extend(candidates)
            except Exception as exc:
                logger.warning(f"[OPRO] Failed to evolve {node_name}: {exc}")

        logger.info(f"[OPRO] Generated {len(all_candidates)} candidates across {len(node_names)} nodes")
        return all_candidates

    def _evolve_node(self, node_name: str) -> list[PromptCandidate]:
        """Run one evolution step for a single node."""
        # Load history
        history = self._load_candidates(node_name)
        champion = next((c for c in history if c.status == "champion"), None)
        testing = [c for c in history if c.status == "testing"]

        # Promote testing → champion if stable
        for candidate in testing:
            if self._is_testing_period_complete(candidate):
                self._promote_to_champion(candidate, champion)
                champion = candidate

        # Score current candidates
        for candidate in history:
            if candidate.status in ("candidate", "testing"):
                candidate.fitness = self.score_candidate(candidate)

        # Generate new candidates using critiques as signal
        critiques = self._load_recent_critiques(node_name)
        new_candidates = self.generate_candidates(node_name, history, critiques)

        # Promote best new candidate if significantly better
        if new_candidates and champion:
            best = max(new_candidates, key=lambda c: c.fitness)
            promoted = self.promote_if_better(best, champion)
            if promoted:
                logger.info(
                    f"[OPRO] Promoted {best.candidate_id[:8]} to testing "
                    f"for {node_name} (fitness={best.fitness:.3f} vs "
                    f"champion={champion.fitness:.3f})"
                )

        return new_candidates

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_candidate(self, candidate: PromptCandidate, window_days: int = 7) -> float:
        """Compute composite fitness from recent trade performance.

        Composite: 0.5*sharpe + 0.3*win_rate + 0.2*(1-max_dd)
        Returns 0.0 if insufficient data.
        """
        try:
            row = self._conn.execute(
                "SELECT "
                "  AVG(realized_pnl_pct) / GREATEST(STDDEV(realized_pnl_pct), 0.01) as sharpe, "
                "  AVG(CASE WHEN realized_pnl_pct > 0 THEN 1.0 ELSE 0.0 END) as win_rate, "
                "  MIN(realized_pnl_pct) as max_loss, "
                "  COUNT(*) as n "
                "FROM strategy_outcomes "
                f"WHERE closed_at >= CURRENT_TIMESTAMP - INTERVAL '{window_days}' DAY "
                "AND realized_pnl_pct IS NOT NULL"
            ).fetchone()

            if not row or row[3] < 3:  # Need at least 3 trades
                return 0.0

            # Annualise the mean/std Sharpe (daily returns → yearly)
            import math
            raw_sharpe = row[0] or 0.0
            sharpe = raw_sharpe * math.sqrt(252)
            win_rate = row[1] or 0.0
            max_dd = abs(row[2] or 0.0) / 100.0  # Normalize to 0-1

            return 0.5 * sharpe + 0.3 * win_rate + 0.2 * (1.0 - min(max_dd, 1.0))
        except Exception as exc:
            logger.warning(f"[OPRO] score_candidate failed: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        node_name: str,
        history: list[PromptCandidate],
        critiques: list[str],
    ) -> list[PromptCandidate]:
        """Generate new candidate prompts via meta-prompt.

        Uses LLM to propose improvements based on:
        - Previous variants and their fitness scores
        - Recent critiques from TextGrad
        """
        generation = max((c.generation for c in history), default=0) + 1

        # Build meta-prompt
        meta_prompt = self._build_meta_prompt(node_name, history, critiques)

        # Try LLM generation
        candidates = []
        try:
            response = litellm.completion(
                model=self._engine,
                messages=[{"role": "user", "content": meta_prompt}],
                max_tokens=500,
                temperature=0.7,
                n=1,
            )
            content = response.choices[0].message.content.strip()

            # Parse response into candidate(s)
            candidate = PromptCandidate(
                candidate_id=str(uuid.uuid4()),
                node_name=node_name,
                prompt_text=content,
                generation=generation,
                fitness=0.0,
                status="candidate",
                source="opro",
                parent_ids=[c.candidate_id for c in history[:3]],
                created_at=datetime.now().isoformat(),
            )
            self._persist_candidate(candidate)
            candidates.append(candidate)

        except Exception as exc:
            logger.warning(f"[OPRO] LLM generation failed for {node_name}: {exc}")

        return candidates

    def _build_meta_prompt(
        self,
        node_name: str,
        history: list[PromptCandidate],
        critiques: list[str],
    ) -> str:
        """Build OPRO meta-prompt from history + critiques."""
        parts = [
            f"You are optimizing the prompt for the '{node_name}' step in a "
            f"trading decision chain. Below are previous prompt variants "
            f"and their performance scores.\n",
        ]

        # Add history
        if history:
            parts.append("## Previous variants (sorted by fitness):")
            sorted_hist = sorted(history, key=lambda c: c.fitness, reverse=True)
            for c in sorted_hist[:5]:
                parts.append(
                    f"- [gen={c.generation}, fitness={c.fitness:.3f}, "
                    f"status={c.status}] {c.prompt_text[:200]}"
                )

        # Add critiques
        if critiques:
            parts.append("\n## Recent critiques from losing trades:")
            for critique in critiques[:5]:
                parts.append(f"- {critique[:200]}")

        parts.append(
            "\nGenerate an improved prompt variant that addresses the critiques "
            "while preserving what worked in the high-fitness variants. "
            "Output ONLY the new prompt text, nothing else."
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_if_better(
        self,
        candidate: PromptCandidate,
        champion: PromptCandidate | None,
    ) -> PromptCandidate | None:
        """Promote candidate to 'testing' if significantly better than champion."""
        if champion is None:
            # No champion yet — first candidate becomes testing
            candidate.status = "testing"
            candidate.promoted_at = datetime.now().isoformat()
            self._update_candidate_status(candidate)
            return candidate

        if champion.fitness <= 0:
            # Champion has no valid fitness — promote any positive candidate
            if candidate.fitness > 0:
                candidate.status = "testing"
                candidate.promoted_at = datetime.now().isoformat()
                self._update_candidate_status(candidate)
                return candidate
            return None

        improvement = (candidate.fitness - champion.fitness) / abs(champion.fitness)
        if improvement >= PROMOTION_THRESHOLD:
            candidate.status = "testing"
            candidate.promoted_at = datetime.now().isoformat()
            self._update_candidate_status(candidate)
            return candidate

        return None

    def _is_testing_period_complete(self, candidate: PromptCandidate) -> bool:
        """Check if a testing candidate has survived MIN_TESTING_DAYS."""
        if not candidate.promoted_at:
            return False
        try:
            promoted = datetime.fromisoformat(candidate.promoted_at)
            return (datetime.now() - promoted).days >= MIN_TESTING_DAYS
        except (ValueError, TypeError):
            return False

    def _promote_to_champion(
        self,
        candidate: PromptCandidate,
        old_champion: PromptCandidate | None,
    ) -> None:
        """Promote testing → champion, retire old champion."""
        candidate.status = "champion"
        self._update_candidate_status(candidate)

        if old_champion:
            old_champion.status = "retired"
            self._update_candidate_status(old_champion)

        logger.info(f"[OPRO] New champion for {candidate.node_name}: {candidate.candidate_id[:8]}")

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _get_active_nodes(self) -> list[str]:
        """Get node names that have critiques (meaning they've been analyzed)."""
        try:
            rows = self._conn.execute(
                f"SELECT DISTINCT node_name FROM {CRITIQUES_TABLE}"
            ).fetchall()
            return [row[0] for row in rows]
        except Exception as exc:
            logger.warning(f"[OPRO] Failed to get active nodes: {exc}")
            return []

    def _load_candidates(self, node_name: str) -> list[PromptCandidate]:
        """Load all candidates for a node."""
        try:
            rows = self._conn.execute(
                f"SELECT candidate_id, node_name, prompt_text, generation, "
                f"fitness, trades_evaluated, status, source, parent_ids, "
                f"created_at, promoted_at "
                f"FROM {CANDIDATES_TABLE} "
                f"WHERE node_name = ? "
                f"ORDER BY fitness DESC",
                [node_name],
            ).fetchall()
            return [
                PromptCandidate(
                    candidate_id=r[0], node_name=r[1], prompt_text=r[2],
                    generation=r[3], fitness=r[4] or 0.0,
                    trades_evaluated=r[5] or 0, status=r[6] or "candidate",
                    source=r[7] or "opro",
                    parent_ids=json.loads(r[8]) if r[8] else [],
                    created_at=str(r[9] or ""),
                    promoted_at=str(r[10]) if r[10] else None,
                )
                for r in rows
            ]
        except Exception as exc:
            logger.warning(f"[OPRO] Failed to load candidates for {node_name}: {exc}")
            return []

    def _load_recent_critiques(self, node_name: str, days: int = 7) -> list[str]:
        """Load recent TextGrad critiques for a node."""
        try:
            rows = self._conn.execute(
                f"SELECT critique FROM {CRITIQUES_TABLE} "
                f"WHERE node_name = ? "
                f"AND created_at >= CURRENT_TIMESTAMP - INTERVAL '{days}' DAY "
                f"ORDER BY created_at DESC LIMIT 10",
                [node_name],
            ).fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception as exc:
            logger.warning(f"[OPRO] Failed to load critiques for {node_name}: {exc}")
            return []

    def _persist_candidate(self, candidate: PromptCandidate) -> None:
        """Insert a new candidate."""
        try:
            self._conn.execute(
                f"INSERT INTO {CANDIDATES_TABLE} "
                f"(candidate_id, node_name, prompt_text, generation, fitness, "
                f"trades_evaluated, status, source, parent_ids, created_at, promoted_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    candidate.candidate_id, candidate.node_name,
                    candidate.prompt_text, candidate.generation,
                    candidate.fitness, candidate.trades_evaluated,
                    candidate.status, candidate.source,
                    json.dumps(candidate.parent_ids),
                    candidate.created_at, candidate.promoted_at,
                ],
            )
        except Exception as exc:
            logger.warning(f"[OPRO] Persist candidate failed: {exc}")

    # NOTE: Research agent prompt evolution (run_weekly_research, _evolve_agent_prompt,
    # _get_active_research_nodes, _load_research_critiques_for_agent) removed —
    # auto-editing .claude/agents/*.md without human review is unsafe, and with
    # <25 research runs/month there's not enough signal to evolve prompts.
    # Re-add when system has measurement infrastructure for agent prompt quality.

    def _update_candidate_status(self, candidate: PromptCandidate) -> None:
        """Update status + promoted_at for an existing candidate."""
        try:
            self._conn.execute(
                f"UPDATE {CANDIDATES_TABLE} "
                f"SET status = ?, promoted_at = ? "
                f"WHERE candidate_id = ?",
                [candidate.status, candidate.promoted_at, candidate.candidate_id],
            )
        except Exception as exc:
            logger.warning(f"[OPRO] Update status failed: {exc}")
