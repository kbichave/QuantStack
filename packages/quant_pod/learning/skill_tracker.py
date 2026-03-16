"""Agent skill tracking helpers — persisted to DuckDB."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

import numpy as np
from loguru import logger

from quant_pod.knowledge.store import KnowledgeStore


@dataclass
class AgentSkill:
    """
    Lightweight agent performance record.

    Tracks both win-rate metrics (prediction_accuracy, signal_win_rate)
    and signal quality metrics (IC, ICIR).

    Why the distinction matters:
      win_rate / prediction_accuracy conflate signal quality with position
      sizing and execution. A signal with IC=0.03 is genuinely predictive
      even if win_rate is 51% — the IC separates signal from noise.
      ICIR (IC mean / IC std) measures consistency: a low-IC signal that
      is stable beats a high-IC signal that is volatile.
    """

    agent_id: str
    prediction_count: int = 0
    correct_predictions: int = 0
    signal_count: int = 0
    winning_signals: int = 0
    total_signal_pnl: float = 0.0
    # IC observations — each entry is the IC for one completed signal period
    ic_observations: list[float] = field(default_factory=list)

    @property
    def prediction_accuracy(self) -> float:
        return self.correct_predictions / self.prediction_count if self.prediction_count else 0.0

    @property
    def signal_win_rate(self) -> float:
        return self.winning_signals / self.signal_count if self.signal_count else 0.0

    @property
    def avg_signal_pnl(self) -> float:
        return self.total_signal_pnl / self.signal_count if self.signal_count else 0.0

    @property
    def ic(self) -> float:
        """Mean Information Coefficient across all recorded observations."""
        if not self.ic_observations:
            return 0.0
        return float(np.mean(self.ic_observations))

    @property
    def ic_std(self) -> float:
        """Standard deviation of IC observations."""
        if len(self.ic_observations) < 2:
            return 0.0
        return float(np.std(self.ic_observations, ddof=1))

    @property
    def icir(self) -> float:
        """
        IC Information Ratio = IC mean / IC std.

        Measures signal consistency. ICIR > 0.5 is considered good;
        ICIR > 1.0 is institutional-grade.
        """
        std = self.ic_std
        return self.ic / std if std > 1e-9 else 0.0

    def rolling_ic(self, window: int = 30) -> float:
        """Mean IC over the most recent `window` observations."""
        if not self.ic_observations:
            return 0.0
        recent = self.ic_observations[-window:]
        return float(np.mean(recent))

    def ic_trend(self) -> str:
        """
        Detect whether IC is improving, stable, or decaying.

        Compares last-10 IC observations to the prior-10.
        Returns "IMPROVING" | "STABLE" | "DECAYING" | "INSUFFICIENT_DATA".
        """
        if len(self.ic_observations) < 20:
            return "INSUFFICIENT_DATA"
        recent = np.mean(self.ic_observations[-10:])
        prior = np.mean(self.ic_observations[-20:-10])
        diff = recent - prior
        if diff > 0.01:
            return "IMPROVING"
        if diff < -0.01:
            return "DECAYING"
        return "STABLE"


class SkillTracker:
    """Tracks prediction and signal quality per agent — persisted to DuckDB."""

    _lock = Lock()

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        self._skills: dict[str, AgentSkill] = {}
        self._ensure_table()
        self._load_from_db()

    def _ensure_table(self) -> None:
        """Create agent_skills and agent_ic_observations tables if they don't exist."""
        try:
            self.store.conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_skills (
                    agent_id            VARCHAR PRIMARY KEY,
                    prediction_count    INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    signal_count        INTEGER DEFAULT 0,
                    winning_signals     INTEGER DEFAULT 0,
                    total_signal_pnl    DOUBLE DEFAULT 0.0,
                    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Separate table for IC time series — allows rolling queries
            self.store.conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_ic_observations (
                    id          INTEGER PRIMARY KEY,
                    agent_id    VARCHAR NOT NULL,
                    ic_value    DOUBLE NOT NULL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        except Exception as e:
            logger.warning(f"[SKILL] Could not create agent_skills table: {e}")

    def _load_from_db(self) -> None:
        """Load persisted skill records and IC observations from DuckDB on startup."""
        try:
            rows = self.store.conn.execute(
                "SELECT agent_id, prediction_count, correct_predictions, "
                "signal_count, winning_signals, total_signal_pnl "
                "FROM agent_skills"
            ).fetchall()
            for row in rows:
                self._skills[row[0]] = AgentSkill(
                    agent_id=row[0],
                    prediction_count=row[1],
                    correct_predictions=row[2],
                    signal_count=row[3],
                    winning_signals=row[4],
                    total_signal_pnl=row[5],
                )
            if rows:
                logger.info(f"[SKILL] Loaded {len(rows)} agent skill records from DB")
        except Exception as e:
            logger.warning(f"[SKILL] Could not load skill records: {e}")

        # Load IC observations (last 200 per agent — enough for rolling metrics)
        try:
            ic_rows = self.store.conn.execute(
                """
                SELECT agent_id, ic_value
                FROM (
                    SELECT agent_id, ic_value,
                           ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY recorded_at DESC) AS rn
                    FROM agent_ic_observations
                )
                WHERE rn <= 200
                ORDER BY agent_id, rn DESC
                """
            ).fetchall()
            for agent_id, ic_value in ic_rows:
                skill = self._skills.get(agent_id)
                if skill:
                    skill.ic_observations.append(float(ic_value))
                # Reverse so observations are in chronological order
            for skill in self._skills.values():
                skill.ic_observations.reverse()
        except Exception as e:
            logger.debug(f"[SKILL] Could not load IC observations: {e}")

    def _persist(self, skill: AgentSkill) -> None:
        """Upsert a skill record to DuckDB."""
        try:
            existing = self.store.conn.execute(
                "SELECT COUNT(*) FROM agent_skills WHERE agent_id = ?",
                [skill.agent_id],
            ).fetchone()[0]

            if existing:
                self.store.conn.execute(
                    """
                    UPDATE agent_skills
                    SET prediction_count = ?, correct_predictions = ?,
                        signal_count = ?, winning_signals = ?,
                        total_signal_pnl = ?, updated_at = ?
                    WHERE agent_id = ?
                    """,
                    [
                        skill.prediction_count,
                        skill.correct_predictions,
                        skill.signal_count,
                        skill.winning_signals,
                        skill.total_signal_pnl,
                        datetime.now(),
                        skill.agent_id,
                    ],
                )
            else:
                self.store.conn.execute(
                    """
                    INSERT INTO agent_skills
                        (agent_id, prediction_count, correct_predictions,
                         signal_count, winning_signals, total_signal_pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        skill.agent_id,
                        skill.prediction_count,
                        skill.correct_predictions,
                        skill.signal_count,
                        skill.winning_signals,
                        skill.total_signal_pnl,
                        datetime.now(),
                    ],
                )
        except Exception as e:
            logger.warning(f"[SKILL] Could not persist skill for {skill.agent_id}: {e}")

    def _get_skill(self, agent_id: str) -> AgentSkill:
        if agent_id not in self._skills:
            self._skills[agent_id] = AgentSkill(agent_id=agent_id)
        return self._skills[agent_id]

    def update_agent_skill(
        self,
        agent_id: str,
        prediction_correct: bool | None = None,
        signal_pnl: float | None = None,
    ) -> AgentSkill:
        """Update metrics for an agent and persist to DuckDB."""
        with self._lock:
            skill = self._get_skill(agent_id)

            if prediction_correct is not None:
                skill.prediction_count += 1
                if prediction_correct:
                    skill.correct_predictions += 1

            if signal_pnl is not None:
                skill.signal_count += 1
                if signal_pnl > 0:
                    skill.winning_signals += 1
                skill.total_signal_pnl += signal_pnl

            self._skills[agent_id] = skill
            self._persist(skill)
        return skill

    def record_ic(
        self,
        agent_id: str,
        ic_value: float,
    ) -> AgentSkill:
        """
        Record an IC observation for an agent after a signal's outcome is known.

        Call this when you can compute the actual correlation between the
        agent's signal at time t and the realised return at t+k.

        Args:
            agent_id: Agent identifier.
            ic_value: Pearson correlation between the signal and forward return.
                      Should be in [-1, 1]; NaN values are silently dropped.

        Returns:
            Updated AgentSkill.
        """
        import math

        if math.isnan(ic_value) or math.isinf(ic_value):
            logger.debug(f"[SKILL] Skipping NaN/Inf IC for {agent_id}")
            return self._get_skill(agent_id)

        with self._lock:
            skill = self._get_skill(agent_id)
            skill.ic_observations.append(float(ic_value))
            # Keep in-memory list bounded at 500 observations
            if len(skill.ic_observations) > 500:
                skill.ic_observations = skill.ic_observations[-500:]
            self._skills[agent_id] = skill
            self._persist_ic(agent_id, ic_value)

        logger.debug(
            f"[SKILL] IC recorded for {agent_id}: {ic_value:.4f} "
            f"(mean={skill.ic:.4f}, ICIR={skill.icir:.2f}, trend={skill.ic_trend()})"
        )
        return skill

    def _persist_ic(self, agent_id: str, ic_value: float) -> None:
        """Append an IC observation to the agent_ic_observations table."""
        try:
            self.store.conn.execute(
                "INSERT INTO agent_ic_observations (agent_id, ic_value, recorded_at) "
                "VALUES (?, ?, ?)",
                [agent_id, ic_value, datetime.now()],
            )
        except Exception as e:
            logger.debug(f"[SKILL] Could not persist IC for {agent_id}: {e}")

    def get_confidence_adjustment(self, agent_id: str) -> float:
        """
        Return a confidence adjustment factor for an agent.

        Incorporates both win-rate metrics and IC/ICIR:
          - Win rate above 0.5 → modest boost
          - ICIR > 0.5 → additional boost (signal is consistently predictive)
          - IC trend DECAYING → reduce confidence regardless of win rate

        New agents default to 1.0. Range is clamped to [0.5, 1.5].
        """
        skill = self._skills.get(agent_id)
        if skill is None or (skill.prediction_count == 0 and skill.signal_count == 0):
            return 1.0

        adjustment = 1.0

        if skill.prediction_count >= 5:
            adjustment += max(-0.2, min(0.3, skill.prediction_accuracy - 0.5))

        if skill.signal_count >= 2:
            adjustment += max(-0.2, min(0.2, skill.signal_win_rate - 0.5))

        # IC-based adjustment — only when we have enough observations
        if len(skill.ic_observations) >= 10:
            # ICIR > 0.5 → meaningful boost; < 0 → penalise
            icir_adj = max(-0.2, min(0.3, skill.icir * 0.2))
            adjustment += icir_adj

            # Decay penalty — if IC is deteriorating, reduce confidence
            trend = skill.ic_trend()
            if trend == "DECAYING":
                adjustment -= 0.15
                logger.debug(
                    f"[SKILL] {agent_id}: IC trend=DECAYING, penalising confidence by 0.15"
                )

        return max(0.5, min(1.5, adjustment))

    def get_all_skills(self) -> list[AgentSkill]:
        """Return all tracked agent skills."""
        return list(self._skills.values())

    def needs_retraining(self, agent_id: str, min_win_rate: float = 0.52) -> bool:
        """
        True if an agent should be retrained based on win rate OR IC decay.

        Triggers when ANY of:
          - signal_count >= 20 AND win_rate < min_win_rate
          - ic_observations >= 20 AND rolling_ic(30) < 0.0 (signal has no edge)
          - ic_trend() == "DECAYING" AND rolling_ic(30) < 0.01
        """
        skill = self._skills.get(agent_id)
        if skill is None:
            return False

        # Win-rate criterion (existing)
        if skill.signal_count >= 20 and skill.signal_win_rate < min_win_rate:
            return True

        # IC criterion — agent's signal has lost predictive power
        if len(skill.ic_observations) >= 20:
            rolling = skill.rolling_ic(30)
            if rolling < 0.0:
                logger.info(
                    f"[SKILL] {agent_id} flagged for retraining: rolling IC={rolling:.4f} < 0"
                )
                return True
            if skill.ic_trend() == "DECAYING" and rolling < 0.01:
                logger.info(
                    f"[SKILL] {agent_id} flagged for retraining: IC decaying + rolling IC={rolling:.4f}"
                )
                return True

        return False

    def ic_summary(self) -> list[dict]:
        """
        Return per-agent IC summary for the /skills API endpoint.

        Includes IC mean, ICIR, rolling IC (30-day), and trend.
        """
        summary = []
        for skill in self._skills.values():
            summary.append(
                {
                    "agent_id": skill.agent_id,
                    "ic": round(skill.ic, 4),
                    "icir": round(skill.icir, 3),
                    "rolling_ic_30": round(skill.rolling_ic(30), 4),
                    "ic_trend": skill.ic_trend(),
                    "n_ic_observations": len(skill.ic_observations),
                    "prediction_accuracy": round(skill.prediction_accuracy, 3),
                    "signal_win_rate": round(skill.signal_win_rate, 3),
                    "avg_signal_pnl": round(skill.avg_signal_pnl, 2),
                    "needs_retraining": self.needs_retraining(skill.agent_id),
                }
            )
        return sorted(summary, key=lambda x: x["icir"], reverse=True)
