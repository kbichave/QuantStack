"""Research pipeline graph builder.

Builds an 8-node StateGraph with one conditional edge:
  START -> context_load -> domain_selection -> hypothesis_generation
        -> signal_validation -> [passed?] -> backtest_validation
        -> ml_experiment -> strategy_registration -> knowledge_update -> END
                           |-> [failed] -> END
"""

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from quantstack.graphs.config_watcher import ConfigWatcher
from quantstack.graphs.state import ResearchState
from quantstack.llm.provider import get_chat_model
from quantstack.tools.registry import get_tools_for_agent

from .nodes import (
    make_backtest_validation,
    make_context_load,
    make_domain_selection,
    make_hypothesis_generation,
    make_knowledge_update,
    make_ml_experiment,
    make_signal_validation,
    make_strategy_registration,
    route_after_validation,
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


def build_research_graph(
    config_watcher: ConfigWatcher,
    checkpointer: BaseCheckpointSaver,
):
    """Build the research pipeline graph.

    Reads agent configs from config_watcher, creates ChatModel instances
    per agent tier, binds YAML-configured tools to each LLM, and compiles.
    """
    quant_cfg = config_watcher.get_config("quant_researcher")
    ml_cfg = config_watcher.get_config("ml_scientist")

    heavy_llm = get_chat_model(quant_cfg.llm_tier)
    ml_llm = get_chat_model(ml_cfg.llm_tier)

    # Bind tools per agent
    quant_llm, quant_tools = _bind_tools_to_llm(heavy_llm, quant_cfg)
    ml_bound_llm, ml_tools = _bind_tools_to_llm(ml_llm, ml_cfg)

    graph = StateGraph(ResearchState)

    # Tool nodes (deterministic, retry once)
    graph.add_node("context_load", make_context_load(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("signal_validation", make_signal_validation(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("backtest_validation", make_backtest_validation(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("strategy_registration", make_strategy_registration(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("knowledge_update", make_knowledge_update(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))

    # Agent nodes (LLM reasoning with tools, retry up to 2 times)
    graph.add_node("domain_selection", make_domain_selection(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("hypothesis_generation", make_hypothesis_generation(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("ml_experiment", make_ml_experiment(ml_bound_llm, ml_cfg, ml_tools), retry=RetryPolicy(max_attempts=3))

    # Linear edges
    graph.add_edge(START, "context_load")
    graph.add_edge("context_load", "domain_selection")
    graph.add_edge("domain_selection", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "signal_validation")

    # Conditional edge after signal_validation
    graph.add_conditional_edges(
        "signal_validation",
        route_after_validation,
        {"backtest_validation": "backtest_validation", END: END},
    )

    # Post-validation linear pipeline
    graph.add_edge("backtest_validation", "ml_experiment")
    graph.add_edge("ml_experiment", "strategy_registration")
    graph.add_edge("strategy_registration", "knowledge_update")
    graph.add_edge("knowledge_update", END)

    return graph.compile(checkpointer=checkpointer)
