# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
ReflexionMemory — structured episodic memory with per-decision injection.

Extends the flat lesson strings from ReflectionManager into indexed,
queryable episodes classified by root cause. Each episode includes a
verbal reinforcement and counterfactual that can be injected into
specific decision prompts before trades.

Paper: Reflexion (Shinn et al., NeurIPS 2023) — https://arxiv.org/abs/2303.11366

Key differences from ReflectionManager:
  - Episodes are classified by root cause (regime shift, sizing, timing, etc.)
  - Retrieval filters by (regime, strategy, symbol) via SQL queries
  - inject_into_prompt() prepends lessons into the decision prompt that
    caused the failure, not a flat file
  - Capped at 1000 episodes (old/small losses pruned monthly)

ReflectionManager remains the raw trade journal; ReflexionMemory is the
indexed, classified, retrievable layer on top.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from quantstack.db import pg_conn
from quantstack.shared.models import TradeReflection


# ---------------------------------------------------------------------------
# Root cause taxonomy
# ---------------------------------------------------------------------------

class RootCause(str, Enum):
    """Deterministic classification of why a trade lost money."""
    REGIME_SHIFT = "regime_shift"
    SIZING_ERROR = "sizing_error"
    ENTRY_TIMING = "entry_timing"
    DATA_GAP = "data_gap"
    STRATEGY_MISMATCH = "strategy_mismatch"
    STOP_LOSS_WIDTH = "stop_loss_width"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Episode record
# ---------------------------------------------------------------------------

@dataclass
class ReflexionEpisode:
    """A classified, retrievable lesson from a losing trade."""
    episode_id: str
    trade_id: int
    regime: str
    strategy_id: str
    symbol: str
    pnl_pct: float
    root_cause: RootCause
    verbal_reinforcement: str
    counterfactual: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPISODES_TABLE = "reflexion_episodes"
MAX_EPISODES = 1000
LOSS_THRESHOLD = -1.0  # Only record episodes for losses worse than -1%

_EPISODE_COLUMNS = (
    "episode_id, trade_id, regime, strategy_id, symbol, "
    "pnl_pct, root_cause, verbal_reinforcement, counterfactual, "
    "tags, created_at"
)


# ---------------------------------------------------------------------------
# ReflexionMemory
# ---------------------------------------------------------------------------

class ReflexionMemory:
    """Structured episodic memory with root-cause classification and per-decision injection.

    Usage:
        mem = ReflexionMemory(conn)
        # After a losing trade:
        episode = mem.record_episode(trade_reflection)
        # Before a new trade:
        lessons = mem.get_relevant("trending_up", "regime_momentum_v1", "SPY")
        prompt = mem.inject_into_prompt(base_prompt, lessons)
    """

    def __init__(self, conn: Any = None) -> None:
        # conn is accepted but ignored — each DB operation uses pg_conn() to
        # ensure transactions are committed immediately and never left open.
        pass

    @staticmethod
    def _row_to_episode(row: tuple) -> ReflexionEpisode:
        """Convert a DB row to a ReflexionEpisode."""
        return ReflexionEpisode(
            episode_id=row[0] or "",
            trade_id=row[1] or 0,
            regime=row[2] or "",
            strategy_id=row[3] or "",
            symbol=row[4] or "",
            pnl_pct=row[5] or 0.0,
            root_cause=RootCause(row[6]) if row[6] else RootCause.UNKNOWN,
            verbal_reinforcement=row[7] or "",
            counterfactual=row[8] or "",
            tags=json.loads(row[9]) if row[9] else [],
            created_at=str(row[10] or ""),
        )

    # ------------------------------------------------------------------
    # Root cause classification (deterministic, no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def classify_root_cause(ref: TradeReflection) -> RootCause:
        """Classify the primary root cause of a losing trade.

        Rules applied in priority order (first match wins):
        1. Regime shifted during hold -> REGIME_SHIFT
        2. Severe loss (>5%) with high conviction (>0.7) -> SIZING_ERROR
        3. Signals were empty or missing -> DATA_GAP
        4. Strategy mismatched with regime at entry -> STRATEGY_MISMATCH
        5. Severe loss (>5%) with low conviction -> STOP_LOSS_WIDTH
        6. Moderate loss with signals present -> ENTRY_TIMING
        7. Everything else -> UNKNOWN
        """
        regime_shifted = (
            ref.regime_at_entry != ref.regime_at_exit
            and ref.regime_at_exit not in ("unknown", "")
            and ref.regime_at_entry not in ("unknown", "")
        )
        if regime_shifted:
            return RootCause.REGIME_SHIFT

        if ref.realized_pnl_pct < -5.0 and ref.conviction_at_entry > 0.7:
            return RootCause.SIZING_ERROR

        if not ref.signals_at_entry or ref.signals_at_entry.strip() == "":
            return RootCause.DATA_GAP

        # Strategy-regime mismatch: momentum in ranging, mean-reversion in trending
        strategy_lower = ref.strategy_id.lower()
        regime_lower = ref.regime_at_entry.lower()
        momentum_in_ranging = "momentum" in strategy_lower and "ranging" in regime_lower
        reversion_in_trending = "reversion" in strategy_lower and "trending" in regime_lower
        if momentum_in_ranging or reversion_in_trending:
            return RootCause.STRATEGY_MISMATCH

        if ref.realized_pnl_pct < -5.0:
            return RootCause.STOP_LOSS_WIDTH

        if ref.signals_at_entry:
            return RootCause.ENTRY_TIMING

        return RootCause.UNKNOWN

    # ------------------------------------------------------------------
    # Record episode
    # ------------------------------------------------------------------

    def record_episode(self, ref: TradeReflection, trade_id: int = 0) -> ReflexionEpisode:
        """Create a classified episode from a losing trade reflection.

        Only call for trades with realized_pnl_pct < LOSS_THRESHOLD.
        """
        root_cause = self.classify_root_cause(ref)
        verbal = self._generate_verbal_reinforcement(ref, root_cause)
        counterfactual = self._generate_counterfactual(ref, root_cause)

        tags = [ref.symbol, ref.strategy_id, ref.regime_at_entry, root_cause.value]
        if ref.regime_at_exit and ref.regime_at_exit != "unknown":
            tags.append(ref.regime_at_exit)

        episode = ReflexionEpisode(
            episode_id=str(uuid.uuid4()),
            trade_id=trade_id,
            regime=ref.regime_at_entry,
            strategy_id=ref.strategy_id,
            symbol=ref.symbol,
            pnl_pct=ref.realized_pnl_pct,
            root_cause=root_cause,
            verbal_reinforcement=verbal,
            counterfactual=counterfactual,
            tags=tags,
            created_at=datetime.now().isoformat(),
        )

        # Persist to DB — pg_conn() commits immediately, preventing idle-in-transaction
        try:
            with pg_conn() as conn:
                conn.execute(
                    f"INSERT INTO {EPISODES_TABLE} "
                    f"(episode_id, trade_id, regime, strategy_id, symbol, pnl_pct, "
                    f"root_cause, verbal_reinforcement, counterfactual, tags, created_at) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        episode.episode_id, trade_id, episode.regime,
                        episode.strategy_id, episode.symbol, episode.pnl_pct,
                        episode.root_cause.value, episode.verbal_reinforcement,
                        episode.counterfactual, json.dumps(episode.tags),
                        episode.created_at,
                    ],
                )
        except Exception as exc:
            logger.warning(f"[ReflexionMemory] Failed to persist episode: {exc}")

        # Prune if over capacity
        self._prune_if_needed()

        logger.info(
            f"[ReflexionMemory] {ref.symbol} episode: {root_cause.value} "
            f"({ref.realized_pnl_pct:+.1f}%) \u2014 {verbal[:80]}"
        )
        return episode

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_relevant(
        self,
        regime: str,
        strategy_id: str = "",
        symbol: str = "",
        k: int = 3,
    ) -> list[ReflexionEpisode]:
        """Retrieve the most relevant past episodes for a decision context.

        Filters by regime, then optionally by strategy and symbol.
        Falls back to regime-only if no exact matches found.
        """
        conditions = ["regime = ?"]
        params: list[Any] = [regime]

        if strategy_id:
            conditions.append("strategy_id = ?")
            params.append(strategy_id)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    f"""SELECT {_EPISODE_COLUMNS}
                        FROM {EPISODES_TABLE}
                        WHERE {' AND '.join(conditions)}
                        ORDER BY abs(pnl_pct) DESC
                        LIMIT ?""",
                    params + [k],
                ).fetchall()
        except Exception as exc:
            logger.debug(f"[ReflexionMemory] get_relevant query failed: {exc}")
            rows = []

        # Fallback: relax to regime-only if specific filters returned nothing
        if not rows and (strategy_id or symbol):
            try:
                with pg_conn() as conn:
                    rows = conn.execute(
                        f"""SELECT {_EPISODE_COLUMNS}
                            FROM {EPISODES_TABLE}
                            WHERE regime = ?
                            ORDER BY abs(pnl_pct) DESC
                            LIMIT ?""",
                        [regime, k],
                    ).fetchall()
            except Exception as exc:
                logger.debug(f"[ReflexionMemory] get_relevant fallback failed: {exc}")
                rows = []

        return [self._row_to_episode(row) for row in rows]

    def inject_into_prompt(
        self,
        base_prompt: str,
        episodes: list[ReflexionEpisode],
    ) -> str:
        """Prepend relevant lessons into a decision prompt.

        Adds a structured section before the base prompt so the agent
        conditions its decision on past failures.
        """
        if not episodes:
            return base_prompt

        lines = ["## Lessons from Similar Past Trades", ""]
        for ep in episodes:
            lines.append(
                f"- [{ep.root_cause.value}] {ep.symbol}/{ep.strategy_id} "
                f"({ep.pnl_pct:+.1f}%): {ep.verbal_reinforcement}"
            )
            if ep.counterfactual:
                lines.append(f"  Counterfactual: {ep.counterfactual}")
        lines.append("")

        return "\n".join(lines) + base_prompt

    # ------------------------------------------------------------------
    # Verbal reinforcement generation (deterministic)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_verbal_reinforcement(
        ref: TradeReflection, root_cause: RootCause,
    ) -> str:
        """Generate a concise verbal lesson from the classified root cause."""
        templates = {
            RootCause.REGIME_SHIFT: (
                f"Regime shifted from {ref.regime_at_entry} to {ref.regime_at_exit} "
                f"during hold. {ref.strategy_id} was designed for {ref.regime_at_entry}. "
                f"Check HMM stability before entry; avoid when stability < 0.6."
            ),
            RootCause.SIZING_ERROR: (
                f"Sized too large for the signal quality. Lost {ref.realized_pnl_pct:.1f}% "
                f"with {ref.conviction_at_entry:.0%} conviction. "
                f"Reduce size when conviction is borderline."
            ),
            RootCause.ENTRY_TIMING: (
                f"Entry timing was poor \u2014 signals were present but trade moved against. "
                f"Consider waiting for confirmation bar or volume spike."
            ),
            RootCause.DATA_GAP: (
                f"Entered with incomplete data (missing collectors or signals). "
                f"Never enter a trade when data coverage is below 80%."
            ),
            RootCause.STRATEGY_MISMATCH: (
                f"Strategy '{ref.strategy_id}' does not fit regime '{ref.regime_at_entry}'. "
                f"Check regime-strategy matrix before deployment."
            ),
            RootCause.STOP_LOSS_WIDTH: (
                f"Severe loss ({ref.realized_pnl_pct:.1f}%) suggests stop-loss was too wide "
                f"or missing. Tighten stops for high-vol environments."
            ),
            RootCause.UNKNOWN: (
                f"Loss of {ref.realized_pnl_pct:.1f}% on {ref.symbol}/{ref.strategy_id}. "
                f"No clear root cause identified \u2014 may be normal variance."
            ),
        }
        return templates.get(root_cause, templates[RootCause.UNKNOWN])

    @staticmethod
    def _generate_counterfactual(
        ref: TradeReflection, root_cause: RootCause,
    ) -> str:
        """Generate what should have been done differently."""
        counterfactuals = {
            RootCause.REGIME_SHIFT: "Wait for regime confirmation before entry; add regime-stability filter.",
            RootCause.SIZING_ERROR: "Use quarter-size for borderline conviction (0.50-0.65).",
            RootCause.ENTRY_TIMING: "Wait for price to close above/below signal level (not just touch).",
            RootCause.DATA_GAP: "Skip trade when >1 collector fails or total coverage < 80%.",
            RootCause.STRATEGY_MISMATCH: "Verify regime-strategy affinity >= 0.6 before deploying.",
            RootCause.STOP_LOSS_WIDTH: "Set ATR-based stop at 1.5x ATR; reduce hold period.",
            RootCause.UNKNOWN: "Review trade context manually for non-obvious factors.",
        }
        return counterfactuals.get(root_cause, counterfactuals[RootCause.UNKNOWN])

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune_if_needed(self) -> None:
        """Remove smallest-loss episodes to stay under MAX_EPISODES.

        Keeps severe losses longer (they're more informative).
        Runs entirely in SQL — no in-memory index to rebuild.
        """
        try:
            with pg_conn() as conn:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {EPISODES_TABLE}"
                ).fetchone()[0]
        except Exception:
            return

        if count <= MAX_EPISODES:
            return

        excess = count - MAX_EPISODES
        try:
            with pg_conn() as conn:
                conn.execute(
                    f"""DELETE FROM {EPISODES_TABLE}
                        WHERE episode_id IN (
                            SELECT episode_id FROM {EPISODES_TABLE}
                            ORDER BY abs(pnl_pct) ASC
                            LIMIT ?
                        )""",
                    [excess],
                )
            logger.info(
                f"[ReflexionMemory] Pruned {excess} episodes, "
                f"{MAX_EPISODES} remaining"
            )
        except Exception as exc:
            logger.warning(f"[ReflexionMemory] Prune DB failed: {exc}")
