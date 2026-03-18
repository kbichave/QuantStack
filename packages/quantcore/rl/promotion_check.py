"""
RL Shadow Mode Promotion Check — convenience function for /review sessions.

Checks all RL agent types that have been in shadow mode, evaluates their
performance via ShadowEvaluator + PromotionGate, and returns a consolidated
promotion recommendation.

Usage:
    from quantcore.rl.promotion_check import rl_promotion_check
    result = rl_promotion_check()
    for rec in result["agents"]:
        print(f"{rec['agent_type']}: {rec['recommendation']}")
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger


_AGENT_TYPES = ["sizing", "execution", "meta"]

# Minimum shadow period in days before promotion is considered
MIN_SHADOW_DAYS = 30


def rl_promotion_check(
    min_shadow_days: int = MIN_SHADOW_DAYS,
) -> dict[str, Any]:
    """
    Check all RL agents for promotion readiness from shadow mode.

    For each agent type (sizing, execution, meta):
      1. Check if shadow mode observations exist (>30 days of data).
      2. Run ShadowEvaluator to compute simulated performance.
      3. Run PromotionGate to apply go/no-go criteria.
      4. Return a consolidated recommendation.

    Args:
        min_shadow_days: Minimum days in shadow mode before evaluation
            (default 30). Overrides the per-agent-type minimums in
            PromotionGate for the duration check only.

    Returns:
        {
            "checked_at": "2026-03-18T...",
            "agents": [
                {
                    "agent_type": "sizing",
                    "n_observations": 75,
                    "shadow_days": 45,
                    "recommendation": "promote" | "continue_shadow" | "no_data",
                    "ready": True | False,
                    "sharpe": 0.82,
                    "win_rate": 0.58,
                    "max_drawdown": 0.08,
                    "reasons": ["All shadow period checks passed."],
                },
                ...
            ],
            "any_ready": True | False,
        }
    """
    results: list[dict[str, Any]] = []

    try:
        from quant_pod.knowledge.store import KnowledgeStore

        store = KnowledgeStore()
    except Exception as exc:
        logger.warning(f"[rl_promotion_check] KnowledgeStore unavailable: {exc}")
        return {
            "checked_at": datetime.utcnow().isoformat(),
            "agents": [
                {
                    "agent_type": at,
                    "n_observations": 0,
                    "shadow_days": 0,
                    "recommendation": "no_data",
                    "ready": False,
                    "reasons": [f"KnowledgeStore unavailable: {exc}"],
                }
                for at in _AGENT_TYPES
            ],
            "any_ready": False,
        }

    try:
        from quantcore.rl.promotion_gate import PromotionGate
        from quantcore.rl.shadow_mode import ShadowEvaluator

        evaluator = ShadowEvaluator(store)
        gate = PromotionGate()
    except Exception as exc:
        logger.warning(f"[rl_promotion_check] RL modules unavailable: {exc}")
        return {
            "checked_at": datetime.utcnow().isoformat(),
            "agents": [
                {
                    "agent_type": at,
                    "n_observations": 0,
                    "shadow_days": 0,
                    "recommendation": "no_data",
                    "ready": False,
                    "reasons": [f"RL modules unavailable: {exc}"],
                }
                for at in _AGENT_TYPES
            ],
            "any_ready": False,
        }

    for agent_type in _AGENT_TYPES:
        try:
            n_obs = evaluator.get_observation_count(agent_type)

            if n_obs == 0:
                results.append({
                    "agent_type": agent_type,
                    "n_observations": 0,
                    "shadow_days": 0,
                    "recommendation": "no_data",
                    "ready": False,
                    "reasons": ["No shadow observations recorded."],
                })
                continue

            # Approximate shadow days from observation count
            # (1 observation per trading day is the expected cadence)
            shadow_days = n_obs  # conservative: 1 obs = 1 day

            if shadow_days < min_shadow_days:
                results.append({
                    "agent_type": agent_type,
                    "n_observations": n_obs,
                    "shadow_days": shadow_days,
                    "recommendation": "continue_shadow",
                    "ready": False,
                    "reasons": [
                        f"Only {shadow_days} days in shadow mode; "
                        f"need {min_shadow_days} minimum."
                    ],
                })
                continue

            # Full evaluation
            shadow_result = evaluator.evaluate_shadow_period(
                agent_type,
                min_observations=gate.MIN_OBSERVATIONS.get(agent_type, 63),
            )

            promotion_result = gate.evaluate(
                agent_type=agent_type,
                shadow_result=shadow_result,
            )

            recommendation = "promote" if promotion_result.passes else "continue_shadow"

            results.append({
                "agent_type": agent_type,
                "n_observations": shadow_result.n_observations,
                "shadow_days": shadow_days,
                "recommendation": recommendation,
                "ready": promotion_result.passes,
                "sharpe": shadow_result.rl_simulated_sharpe,
                "win_rate": shadow_result.rl_simulated_win_rate,
                "max_drawdown": shadow_result.rl_simulated_max_drawdown,
                "agreement": shadow_result.directional_agreement_rate,
                "reasons": shadow_result.reasons,
                "gate_checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "value": c.value,
                        "threshold": c.threshold,
                        "message": c.message,
                    }
                    for c in promotion_result.checks
                ],
            })

        except Exception as exc:
            logger.warning(
                f"[rl_promotion_check] Failed for {agent_type}: {exc}"
            )
            results.append({
                "agent_type": agent_type,
                "n_observations": 0,
                "shadow_days": 0,
                "recommendation": "no_data",
                "ready": False,
                "reasons": [f"Evaluation failed: {exc}"],
            })

    return {
        "checked_at": datetime.utcnow().isoformat(),
        "agents": results,
        "any_ready": any(r["ready"] for r in results),
    }
