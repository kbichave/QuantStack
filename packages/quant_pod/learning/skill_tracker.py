"""Agent skill tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from quant_pod.knowledge.store import KnowledgeStore


@dataclass
class AgentSkill:
    """Lightweight agent performance record."""

    agent_id: str
    prediction_count: int = 0
    correct_predictions: int = 0
    signal_count: int = 0
    winning_signals: int = 0
    total_signal_pnl: float = 0.0

    @property
    def prediction_accuracy(self) -> float:
        return (
            self.correct_predictions / self.prediction_count
            if self.prediction_count
            else 0.0
        )

    @property
    def signal_win_rate(self) -> float:
        return self.winning_signals / self.signal_count if self.signal_count else 0.0

    @property
    def avg_signal_pnl(self) -> float:
        return self.total_signal_pnl / self.signal_count if self.signal_count else 0.0


class SkillTracker:
    """Tracks prediction and signal quality per agent."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        self._skills: Dict[str, AgentSkill] = {}

    def _get_skill(self, agent_id: str) -> AgentSkill:
        if agent_id not in self._skills:
            self._skills[agent_id] = AgentSkill(agent_id=agent_id)
        return self._skills[agent_id]

    def update_agent_skill(
        self,
        agent_id: str,
        prediction_correct: Optional[bool] = None,
        signal_pnl: Optional[float] = None,
    ) -> AgentSkill:
        """Update metrics for an agent and return the latest snapshot."""
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
        return skill

    def get_confidence_adjustment(self, agent_id: str) -> float:
        """
        Return a confidence adjustment factor for an agent.

        New agents default to 1.0. Agents with strong track records can get
        modest boosts; underperformers are capped to avoid negative swings.
        """
        skill = self._skills.get(agent_id)
        if skill is None or (skill.prediction_count == 0 and skill.signal_count == 0):
            return 1.0

        adjustment = 1.0

        if skill.prediction_count >= 5:
            adjustment += max(-0.2, min(0.5, skill.prediction_accuracy - 0.5))

        if skill.signal_count >= 2:
            adjustment += max(-0.2, min(0.25, skill.signal_win_rate - 0.5))

        return max(0.5, adjustment)
