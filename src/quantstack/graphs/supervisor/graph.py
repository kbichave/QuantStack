"""Supervisor pipeline graph builder.

Builds a 5-node linear StateGraph:
  START -> health_check -> diagnose_issues -> execute_recovery
        -> strategy_lifecycle -> scheduled_tasks -> END
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from quantstack.graphs.config_watcher import ConfigWatcher
from quantstack.graphs.state import SupervisorState
from quantstack.llm.provider import get_chat_model

from .nodes import (
    make_diagnose_issues,
    make_execute_recovery,
    make_health_check,
    make_scheduled_tasks,
    make_strategy_lifecycle,
)


def build_supervisor_graph(
    config_watcher: ConfigWatcher,
    checkpointer: BaseCheckpointSaver,
):
    """Build the supervisor pipeline graph.

    Reads agent configs from config_watcher, creates ChatModel instances
    per agent tier, binds them to node functions, and compiles the graph.
    """
    health_cfg = config_watcher.get_config("health_monitor")
    healer_cfg = config_watcher.get_config("self_healer")
    promoter_cfg = config_watcher.get_config("strategy_promoter")

    light_llm = get_chat_model(health_cfg.llm_tier)
    medium_llm = get_chat_model(promoter_cfg.llm_tier)

    graph = StateGraph(SupervisorState)

    graph.add_node(
        "health_check",
        make_health_check(light_llm, health_cfg),
        retry=RetryPolicy(max_attempts=2),
    )
    graph.add_node(
        "diagnose_issues",
        make_diagnose_issues(light_llm, healer_cfg),
        retry=RetryPolicy(max_attempts=3),
    )
    graph.add_node(
        "execute_recovery",
        make_execute_recovery(light_llm, healer_cfg),
        retry=RetryPolicy(max_attempts=3),
    )
    graph.add_node(
        "strategy_lifecycle",
        make_strategy_lifecycle(medium_llm, promoter_cfg),
        retry=RetryPolicy(max_attempts=3),
    )
    graph.add_node(
        "scheduled_tasks",
        make_scheduled_tasks(light_llm, health_cfg),
        retry=RetryPolicy(max_attempts=2),
    )

    graph.add_edge(START, "health_check")
    graph.add_edge("health_check", "diagnose_issues")
    graph.add_edge("diagnose_issues", "execute_recovery")
    graph.add_edge("execute_recovery", "strategy_lifecycle")
    graph.add_edge("strategy_lifecycle", "scheduled_tasks")
    graph.add_edge("scheduled_tasks", END)

    return graph.compile(checkpointer=checkpointer)
