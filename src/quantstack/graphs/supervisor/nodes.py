"""Node functions for the Supervisor Graph.

Each node is an async function: (SupervisorState) -> dict
The return dict contains only the state fields the node updates.
"""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from quantstack.graphs.config import AgentConfig
from quantstack.graphs.state import SupervisorState

logger = logging.getLogger(__name__)


def _build_system_message(config: AgentConfig) -> SystemMessage:
    """Build a system message from an AgentConfig."""
    return SystemMessage(content=(
        f"You are a {config.role}.\n\n"
        f"Goal: {config.goal}\n\n"
        f"Background: {config.backstory}\n\n"
        "Always respond with valid JSON."
    ))


def make_health_check(llm: BaseChatModel, config: AgentConfig):
    """Create the health_check node function."""

    async def health_check(state: SupervisorState) -> dict[str, Any]:
        """Check system health: heartbeats, services, data freshness."""
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Cycle {state['cycle_number']}: Run a system health check. "
                    "Check service heartbeats, reachability, data freshness, "
                    "and API rate limits. Classify each as healthy/degraded/critical. "
                    "Return JSON with per-service status."
                )),
            ])
            try:
                health_status = json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                health_status = {"raw_response": response.content, "parse_error": True}
            return {"health_status": health_status}
        except Exception as exc:
            logger.error("health_check failed: %s", exc)
            return {
                "health_status": {"error": str(exc), "overall": "unknown"},
                "errors": [f"health_check: {exc}"],
            }

    return health_check


def make_diagnose_issues(llm: BaseChatModel, config: AgentConfig):
    """Create the diagnose_issues node function."""

    async def diagnose_issues(state: SupervisorState) -> dict[str, Any]:
        """Diagnose root causes of any degraded/critical services."""
        health = state.get("health_status", {})
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"System health status:\n{json.dumps(health, indent=2)}\n\n"
                    "Diagnose any issues found. For each degraded/critical service, "
                    "identify root cause and recommend a recovery action. "
                    "If all healthy, return an empty array. "
                    'Return JSON: [{"service": ..., "diagnosis": ..., "recommended_action": ...}]'
                )),
            ])
            try:
                issues = json.loads(response.content)
                if not isinstance(issues, list):
                    issues = [issues] if issues else []
            except (json.JSONDecodeError, TypeError):
                issues = []
            return {"diagnosed_issues": issues}
        except Exception as exc:
            logger.error("diagnose_issues failed: %s", exc)
            return {
                "diagnosed_issues": [],
                "errors": [f"diagnose_issues: {exc}"],
            }

    return diagnose_issues


def make_execute_recovery(llm: BaseChatModel, config: AgentConfig):
    """Create the execute_recovery node function."""

    async def execute_recovery(state: SupervisorState) -> dict[str, Any]:
        """Execute recovery actions for diagnosed issues."""
        issues = state.get("diagnosed_issues", [])
        if not issues:
            return {"recovery_actions": []}

        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Diagnosed issues:\n{json.dumps(issues, indent=2)}\n\n"
                    "Execute recovery actions from the playbook. "
                    "For each issue, execute the recommended action and report results. "
                    'Return JSON: [{"action": ..., "target": ..., "result": ...}]'
                )),
            ])
            try:
                actions = json.loads(response.content)
                if not isinstance(actions, list):
                    actions = [actions] if actions else []
            except (json.JSONDecodeError, TypeError):
                actions = []
            return {"recovery_actions": actions}
        except Exception as exc:
            logger.error("execute_recovery failed: %s", exc)
            return {
                "recovery_actions": [],
                "errors": [f"execute_recovery: {exc}"],
            }

    return execute_recovery


def make_strategy_lifecycle(llm: BaseChatModel, config: AgentConfig):
    """Create the strategy_lifecycle node function."""

    async def strategy_lifecycle(state: SupervisorState) -> dict[str, Any]:
        """Evaluate forward-testing strategies for promotion/retirement."""
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Cycle {state['cycle_number']}: Review all forward_testing strategies. "
                    "For each, evaluate performance evidence (P&L, win rate, drawdown, "
                    "trade count, duration) and decide: promote, extend, retire, or no_change. "
                    'Return JSON: [{"strategy_id": ..., "decision": ..., "reasoning": ...}]'
                )),
            ])
            try:
                actions = json.loads(response.content)
                if not isinstance(actions, list):
                    actions = [actions] if actions else []
            except (json.JSONDecodeError, TypeError):
                actions = []
            return {"strategy_lifecycle_actions": actions}
        except Exception as exc:
            logger.error("strategy_lifecycle failed: %s", exc)
            return {
                "strategy_lifecycle_actions": [],
                "errors": [f"strategy_lifecycle: {exc}"],
            }

    return strategy_lifecycle


def make_scheduled_tasks(llm: BaseChatModel, config: AgentConfig):
    """Create the scheduled_tasks node function."""

    async def scheduled_tasks(state: SupervisorState) -> dict[str, Any]:
        """Check and fire due scheduled tasks."""
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Cycle {state['cycle_number']}: Check scheduled tasks. "
                    "Check due tasks: weekly community-intel scan, monthly execution audit, "
                    "30-min data freshness check, daily preflight, daily digest. "
                    "Fire coordination events for due tasks. "
                    'Return JSON: [{"task": ..., "was_due": true/false, "fired": true/false}]'
                )),
            ])
            try:
                results = json.loads(response.content)
                if not isinstance(results, list):
                    results = [results] if results else []
            except (json.JSONDecodeError, TypeError):
                results = []
            return {"scheduled_task_results": results}
        except Exception as exc:
            logger.error("scheduled_tasks failed: %s", exc)
            return {
                "scheduled_task_results": [],
                "errors": [f"scheduled_tasks: {exc}"],
            }

    return scheduled_tasks
