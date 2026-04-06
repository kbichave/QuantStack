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
from quantstack.graphs.tool_binding import bind_tools_to_llm

from .nodes import (
    make_diagnose_issues,
    make_eod_data_sync,
    make_execute_recovery,
    make_health_check,
    make_scheduled_tasks,
    make_strategy_lifecycle,
    make_strategy_pipeline,
)

logger = logging.getLogger(__name__)


def build_supervisor_graph(
    config_watcher: ConfigWatcher,
    checkpointer: BaseCheckpointSaver,
):
    """Build the supervisor pipeline graph with tool-enabled agents."""
    health_cfg = config_watcher.get_config("health_monitor")
    healer_cfg = config_watcher.get_config("self_healer")
    promoter_cfg = config_watcher.get_config("strategy_promoter")

    # community_intel lives in the research YAML — load it directly
    from pathlib import Path
    from quantstack.graphs.config import load_agent_configs

    research_yaml = Path(__file__).resolve().parent.parent / "research" / "config" / "agents.yaml"
    research_configs = load_agent_configs(research_yaml)
    community_cfg = research_configs["community_intel"]

    # Per-agent LLM instances (each agent gets its own tier + thinking config)
    health_llm_base = get_chat_model(health_cfg.llm_tier, thinking=health_cfg.thinking)
    healer_llm_base = get_chat_model(healer_cfg.llm_tier, thinking=healer_cfg.thinking)
    promoter_llm_base = get_chat_model(promoter_cfg.llm_tier, thinking=promoter_cfg.thinking)
    community_llm_base = get_chat_model(community_cfg.llm_tier, thinking=community_cfg.thinking)

    # Bind tools per agent
    health_llm, health_tools, _ = bind_tools_to_llm(health_llm_base, health_cfg)
    healer_llm, healer_tools, _ = bind_tools_to_llm(healer_llm_base, healer_cfg)
    promoter_llm, promoter_tools, _ = bind_tools_to_llm(promoter_llm_base, promoter_cfg)
    community_llm, community_tools, _ = bind_tools_to_llm(community_llm_base, community_cfg)

    graph = StateGraph(SupervisorState)

    graph.add_node("health_check", make_health_check(health_llm, health_cfg, health_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("diagnose_issues", make_diagnose_issues(healer_llm, healer_cfg, healer_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("execute_recovery", make_execute_recovery(healer_llm, healer_cfg, healer_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("strategy_pipeline", make_strategy_pipeline(), retry=RetryPolicy(max_attempts=2))
    graph.add_node("strategy_lifecycle", make_strategy_lifecycle(promoter_llm, promoter_cfg, promoter_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("scheduled_tasks", make_scheduled_tasks(
        health_llm, health_cfg, health_tools,
        community_llm=community_llm, community_cfg=community_cfg, community_tools=community_tools,
    ), retry=RetryPolicy(max_attempts=2))
    graph.add_node("eod_data_sync", make_eod_data_sync())

    graph.add_edge(START, "health_check")
    graph.add_edge("health_check", "diagnose_issues")
    graph.add_edge("diagnose_issues", "execute_recovery")
    graph.add_edge("execute_recovery", "strategy_pipeline")
    graph.add_edge("strategy_pipeline", "strategy_lifecycle")
    graph.add_edge("strategy_lifecycle", "scheduled_tasks")
    graph.add_edge("scheduled_tasks", "eod_data_sync")
    graph.add_edge("eod_data_sync", END)

    return graph.compile(checkpointer=checkpointer)
