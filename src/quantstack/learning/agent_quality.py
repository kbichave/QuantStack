"""Agent quality evaluation — degradation detection and confidence formatting.

Standalone module that operates on plain dicts (extracted from SkillTracker)
so it can be tested without database or heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class AgentQualityReport:
    """Result of evaluating a single agent's decision quality."""

    agent_id: str
    win_rate: float
    signal_count: int
    is_degraded: bool
    confidence_label: str  # "reliable", "cautious", "degraded, under investigation"


def evaluate_agent_quality(
    skills: dict[str, Any],
    min_trades: int = 30,
    alert_threshold: float = 0.40,
) -> list[AgentQualityReport]:
    """Evaluate all agents and return a report for each with enough data.

    Args:
        skills: dict of agent_id -> dict with keys ``signal_count``,
                ``winning_signals``, ``confidence_adjustment`` (float in [0.5, 1.5]).
        min_trades: minimum completed signals before an agent is evaluated.
        alert_threshold: win rate strictly below this triggers degradation.

    Returns:
        List of :class:`AgentQualityReport` for agents that meet the
        ``min_trades`` threshold.
    """
    reports: list[AgentQualityReport] = []
    for agent_id, skill in skills.items():
        signal_count = skill.get("signal_count", 0)
        if signal_count < min_trades:
            continue

        winning = skill.get("winning_signals", 0)
        win_rate = winning / signal_count if signal_count > 0 else 0.0
        conf_adj = skill.get("confidence_adjustment", 1.0)

        is_degraded = win_rate < alert_threshold

        # Determine confidence label from the adjustment factor
        if conf_adj >= 1.0:
            label = "reliable"
        elif conf_adj >= 0.8:
            label = "cautious"
        else:
            label = "degraded"

        # Override label when degraded by win-rate criterion
        if is_degraded:
            label = "degraded, under investigation"

        reports.append(
            AgentQualityReport(
                agent_id=agent_id,
                win_rate=round(win_rate, 3),
                signal_count=signal_count,
                is_degraded=is_degraded,
                confidence_label=label,
            )
        )

    return reports


def get_degraded_agents(
    skills: dict[str, Any],
    min_trades: int = 30,
    alert_threshold: float = 0.40,
) -> list[dict[str, Any]]:
    """Return degraded agents with context suitable for research-task queuing.

    Each entry contains ``agent_id``, ``win_rate``, ``signal_count``, and
    ``task_type`` (always ``"agent_prompt_investigation"``).
    """
    reports = evaluate_agent_quality(skills, min_trades, alert_threshold)
    return [
        {
            "agent_id": r.agent_id,
            "win_rate": r.win_rate,
            "signal_count": r.signal_count,
            "task_type": "agent_prompt_investigation",
        }
        for r in reports
        if r.is_degraded
    ]


def format_agent_confidence(skills: dict[str, Any], min_signals: int = 5) -> str:
    """Format per-agent confidence for injection into daily-plan prompts.

    Returns an empty string when no agents have enough signals.
    """
    entries: list[str] = []
    for agent_id, skill in skills.items():
        if skill.get("signal_count", 0) < min_signals:
            continue
        conf = skill.get("confidence_adjustment", 1.0)
        if conf >= 1.0:
            label = "reliable"
        elif conf >= 0.8:
            label = "cautious"
        else:
            label = "degraded, under investigation"
        entries.append(f"{agent_id}={conf:.1f} ({label})")

    if not entries:
        return ""
    return "Agent confidence: " + ", ".join(entries) + "."
