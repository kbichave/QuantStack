"""Supervisor pipeline graph builder.

Builds a 6-node linear StateGraph:
  START -> health_check -> diagnose_issues -> execute_recovery
        -> strategy_lifecycle -> scheduled_tasks -> eod_data_sync -> END
"""

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from quantstack.graphs.config_watcher import ConfigWatcher
from quantstack.graphs.state import SupervisorState
from quantstack.llm.provider import get_chat_model
from quantstack.tools.registry import get_tools_for_agent

from .nodes import (
    make_diagnose_issues,
    make_eod_data_sync,
    make_execute_recovery,
    make_health_check,
    make_scheduled_tasks,
    make_strategy_lifecycle,
)

logger = logging.getLogger(__name__)


def _bind_tools_to_llm(llm, config):
    """Bind tools from agent config to the LLM. Returns (llm_with_tools, tool_list)."""
    if not config.tools:
        return llm, []
    try:
        tools = get_tools_for_agent(config.tools)
        bound_llm = llm.bind_tools(tools)
        logger.info(
            "Agent '%s' bound %d tools: %s",
            config.name, len(tools), [t.name for t in tools],
        )
        return bound_llm, tools
    except (KeyError, Exception) as exc:
        logger.warning(
            "Failed to bind tools for agent '%s': %s — running without tools",
            config.name, exc,
        )
        return llm, []


def build_supervisor_graph(
    config_watcher: ConfigWatcher,
    checkpointer: BaseCheckpointSaver,
):
    """Build the supervisor pipeline graph with tool-enabled agents."""
    health_cfg = config_watcher.get_config("health_monitor")
    healer_cfg = config_watcher.get_config("self_healer")
    promoter_cfg = config_watcher.get_config("strategy_promoter")

    light_llm = get_chat_model(health_cfg.llm_tier)
    medium_llm = get_chat_model(promoter_cfg.llm_tier)

    # Bind tools per agent
    health_llm, health_tools = _bind_tools_to_llm(light_llm, health_cfg)
    healer_llm, healer_tools = _bind_tools_to_llm(light_llm, healer_cfg)
    promoter_llm, promoter_tools = _bind_tools_to_llm(medium_llm, promoter_cfg)

    graph = StateGraph(SupervisorState)

    graph.add_node("health_check", make_health_check(health_llm, health_cfg, health_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("diagnose_issues", make_diagnose_issues(healer_llm, healer_cfg, healer_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("execute_recovery", make_execute_recovery(healer_llm, healer_cfg, healer_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("strategy_lifecycle", make_strategy_lifecycle(promoter_llm, promoter_cfg, promoter_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("scheduled_tasks", make_scheduled_tasks(health_llm, health_cfg, health_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("eod_data_sync", make_eod_data_sync())

    graph.add_edge(START, "health_check")
    graph.add_edge("health_check", "diagnose_issues")
    graph.add_edge("diagnose_issues", "execute_recovery")
    graph.add_edge("execute_recovery", "strategy_lifecycle")
    graph.add_edge("strategy_lifecycle", "scheduled_tasks")
    graph.add_edge("scheduled_tasks", "eod_data_sync")
    graph.add_edge("eod_data_sync", END)

    return graph.compile(checkpointer=checkpointer)
