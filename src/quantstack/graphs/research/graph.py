"""Research pipeline graph builder.

Builds a StateGraph with self-critique loop (WI-8) and optional fan-out (WI-7):
  START -> context_load -> domain_selection -> hypothesis_generation
        -> hypothesis_critique -> [confidence check]
           -> [loop back to hypothesis_generation if low confidence]
           -> signal_validation (or fan-out workers if WI-7 enabled) -> ...

Fan-out (RESEARCH_FAN_OUT_ENABLED=true):
  ... -> fan_out_hypotheses -> validate_symbol (parallel) -> filter_results
       -> strategy_registration -> knowledge_update -> END

Sequential (RESEARCH_FAN_OUT_ENABLED=false, default):
  ... -> signal_validation -> [passed?] -> backtest_validation
       -> ml_experiment -> strategy_registration -> knowledge_update -> END
"""

import logging
import os

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from quantstack.graphs.config_watcher import ConfigWatcher
from quantstack.graphs.state import ResearchState
from quantstack.llm.provider import get_chat_model
from quantstack.graphs.tool_binding import bind_tools_to_llm

from .nodes import (
    fan_out_hypotheses,
    make_backtest_validation,
    make_context_load,
    make_domain_selection,
    make_filter_results,
    make_hypothesis_critique,
    make_hypothesis_generation,
    make_knowledge_update,
    make_ml_experiment,
    make_signal_validation,
    make_strategy_registration,
    make_validate_symbol,
    route_after_hypothesis,
    route_after_hypothesis_fanout,
    route_after_validation,
)

logger = logging.getLogger(__name__)


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
    critic_cfg = config_watcher.get_config("hypothesis_critic")

    # Per-agent LLM instances (each agent gets its own tier + thinking config)
    quant_llm_base = get_chat_model(quant_cfg.llm_tier, thinking=quant_cfg.thinking)
    ml_llm_base = get_chat_model(ml_cfg.llm_tier, thinking=ml_cfg.thinking)
    critic_llm_base = get_chat_model(critic_cfg.llm_tier, thinking=critic_cfg.thinking)

    # Bind tools per agent
    quant_llm, quant_tools, _ = bind_tools_to_llm(quant_llm_base, quant_cfg)
    ml_bound_llm, ml_tools, _ = bind_tools_to_llm(ml_llm_base, ml_cfg)
    critic_llm, critic_tools, _ = bind_tools_to_llm(critic_llm_base, critic_cfg)

    fan_out_enabled = os.environ.get("RESEARCH_FAN_OUT_ENABLED", "false").lower() == "true"

    graph = StateGraph(ResearchState)

    # Common nodes (always present)
    graph.add_node("context_load", make_context_load(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("domain_selection", make_domain_selection(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("hypothesis_generation", make_hypothesis_generation(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("hypothesis_critique", make_hypothesis_critique(critic_llm, critic_cfg, critic_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("strategy_registration", make_strategy_registration(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("knowledge_update", make_knowledge_update(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))

    # Linear edges: START -> context_load -> domain_selection -> hypothesis_generation -> hypothesis_critique
    graph.add_edge(START, "context_load")
    graph.add_edge("context_load", "domain_selection")
    graph.add_edge("domain_selection", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "hypothesis_critique")

    if fan_out_enabled:
        # Fan-out path: workers + filter_results
        graph.add_node("validate_symbol", make_validate_symbol(
            quant_llm, ml_bound_llm, quant_cfg, ml_cfg, quant_tools, ml_tools
        ))
        graph.add_node("filter_results", make_filter_results())

        # After critique: fan out to parallel workers or loop back
        graph.add_conditional_edges(
            "hypothesis_critique",
            route_after_hypothesis_fanout,
            {"hypothesis_generation": "hypothesis_generation"},
        )
        graph.add_edge("filter_results", "strategy_registration")
    else:
        # Sequential path (original)
        graph.add_node("signal_validation", make_signal_validation(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))
        graph.add_node("backtest_validation", make_backtest_validation(quant_llm, quant_cfg, quant_tools), retry=RetryPolicy(max_attempts=2))
        graph.add_node("ml_experiment", make_ml_experiment(ml_bound_llm, ml_cfg, ml_tools), retry=RetryPolicy(max_attempts=3))

        # After critique: forward to signal_validation or loop back
        graph.add_conditional_edges(
            "hypothesis_critique",
            route_after_hypothesis,
            {"signal_validation": "signal_validation", "hypothesis_generation": "hypothesis_generation"},
        )

        # Conditional edge after signal_validation
        graph.add_conditional_edges(
            "signal_validation",
            route_after_validation,
            {"backtest_validation": "backtest_validation", END: END},
        )

        graph.add_edge("backtest_validation", "ml_experiment")
        graph.add_edge("ml_experiment", "strategy_registration")

    # Common tail
    graph.add_edge("strategy_registration", "knowledge_update")
    graph.add_edge("knowledge_update", END)

    return graph.compile(checkpointer=checkpointer)
