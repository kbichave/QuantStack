"""Trading pipeline graph builder.

Builds a 12-node StateGraph with parallel branches and risk gate:
  START -> safety_check -> [halted?] -> END
                        -> plan_day
                            |-> position_review -> execute_exits -> merge_parallel
                            |-> entry_scan --------------------->  merge_parallel
                        -> risk_sizing -> [SafetyGate] -> portfolio_review
                                                        -> analyze_options
                                                        -> execute_entries
                                                        -> reflect -> END
                                       |-> [rejected] -> END
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from quantstack.graphs.config_watcher import ConfigWatcher
from quantstack.graphs.state import TradingState
from quantstack.llm.provider import get_chat_model

from .nodes import (
    make_daily_plan,
    make_entry_scan,
    make_execute_entries,
    make_execute_exits,
    make_options_analysis,
    make_portfolio_review,
    make_position_review,
    make_reflection,
    make_risk_sizing,
    make_safety_check,
    merge_parallel,
)


def _safety_check_router(state: TradingState) -> str:
    """Route to END if system is halted, daily_plan otherwise."""
    decisions = state.get("decisions", [])
    for d in reversed(decisions):
        if d.get("node") == "safety_check":
            if d.get("halted", False):
                return "halt"
            break
    # Also check errors from safety_check for crash-halts
    errors = state.get("errors", [])
    if any("safety_check" in e for e in errors):
        return "halt"
    return "continue"


def _risk_gate_router(state: TradingState) -> str:
    """Route based on SafetyGate verdict.

    If ANY verdict is rejected, the entire batch is rejected.
    Conservative by design — partial approval is not supported
    because correlated entries could collectively breach limits.
    """
    verdicts = state.get("risk_verdicts", [])
    if not verdicts:
        return "rejected"

    for verdict in verdicts:
        if not verdict.get("approved", False):
            return "rejected"

    return "approved"


def build_trading_graph(
    config_watcher: ConfigWatcher,
    checkpointer: BaseCheckpointSaver,
):
    """Build the trading pipeline graph.

    Reads agent configs from config_watcher, creates ChatModel instances
    per agent tier, binds them to node functions, and compiles the graph.
    """
    planner_cfg = config_watcher.get_config("daily_planner")
    monitor_cfg = config_watcher.get_config("position_monitor")
    debater_cfg = config_watcher.get_config("trade_debater")
    risk_cfg = config_watcher.get_config("risk_analyst")
    fm_cfg = config_watcher.get_config("fund_manager")
    options_cfg = config_watcher.get_config("options_analyst")
    reflector_cfg = config_watcher.get_config("trade_reflector")

    medium_llm = get_chat_model(planner_cfg.llm_tier)
    heavy_llm = get_chat_model(debater_cfg.llm_tier)

    graph = StateGraph(TradingState)

    # Critical node (no retry — fail fast)
    graph.add_node("safety_check", make_safety_check(medium_llm, planner_cfg))

    # Agent nodes (retry up to 2 times for transient LLM failures)
    # Node names must differ from TradingState keys (LangGraph constraint),
    # so daily_plan/options_analysis/reflection are renamed to avoid collision.
    graph.add_node("plan_day", make_daily_plan(medium_llm, planner_cfg), retry=RetryPolicy(max_attempts=3))
    graph.add_node("position_review", make_position_review(medium_llm, monitor_cfg), retry=RetryPolicy(max_attempts=3))
    graph.add_node("entry_scan", make_entry_scan(heavy_llm, debater_cfg), retry=RetryPolicy(max_attempts=3))
    graph.add_node("portfolio_review", make_portfolio_review(heavy_llm, fm_cfg), retry=RetryPolicy(max_attempts=3))
    graph.add_node("analyze_options", make_options_analysis(heavy_llm, options_cfg), retry=RetryPolicy(max_attempts=3))
    graph.add_node("reflect", make_reflection(medium_llm, reflector_cfg), retry=RetryPolicy(max_attempts=3))

    # Tool nodes (retry once for deterministic failures)
    graph.add_node("execute_exits", make_execute_exits(medium_llm, monitor_cfg), retry=RetryPolicy(max_attempts=2))
    graph.add_node("execute_entries", make_execute_entries(medium_llm, fm_cfg), retry=RetryPolicy(max_attempts=2))

    # Risk sizing (SafetyGate portion has no retry — critical)
    graph.add_node("risk_sizing", make_risk_sizing(medium_llm, risk_cfg))

    # Join node (no retry needed)
    graph.add_node("merge_parallel", merge_parallel)

    # --- Edges ---

    # START -> safety_check
    graph.add_edge(START, "safety_check")

    # Conditional: halted -> END, healthy -> plan_day
    graph.add_conditional_edges(
        "safety_check",
        _safety_check_router,
        {"continue": "plan_day", "halt": END},
    )

    # Parallel branches from plan_day
    graph.add_edge("plan_day", "position_review")
    graph.add_edge("plan_day", "entry_scan")

    # Position review branch
    graph.add_edge("position_review", "execute_exits")
    graph.add_edge("execute_exits", "merge_parallel")

    # Entry scan joins at merge
    graph.add_edge("entry_scan", "merge_parallel")

    # Post-merge linear pipeline
    graph.add_edge("merge_parallel", "risk_sizing")

    # Risk gate conditional edge (mandatory, no bypass)
    graph.add_conditional_edges(
        "risk_sizing",
        _risk_gate_router,
        {"approved": "portfolio_review", "rejected": END},
    )

    # Post-risk-gate linear pipeline
    graph.add_edge("portfolio_review", "analyze_options")
    graph.add_edge("analyze_options", "execute_entries")
    graph.add_edge("execute_entries", "reflect")
    graph.add_edge("reflect", END)

    return graph.compile(checkpointer=checkpointer)
