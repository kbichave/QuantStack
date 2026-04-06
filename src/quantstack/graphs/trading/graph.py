"""Trading pipeline graph builder.

Builds a 16-node StateGraph with parallel branches and risk gate:
  START -> data_refresh -> safety_check -> [halted?] -> END
                                        -> market_intel -> plan_day
                            |-> position_review -> execute_exits -> merge_parallel
                            |-> entry_scan --------------------->  merge_parallel
                        -> risk_sizing -> [SafetyGate] -> portfolio_construction
                                                         |-> portfolio_review -----> merge_pre_execution
                                                         |-> analyze_options -----> merge_pre_execution
                                                        -> execute_entries -> reflect -> END
                                       |-> [rejected] -> END
"""

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from quantstack.graphs.config_watcher import ConfigWatcher
from quantstack.graphs.state import TradingState
from quantstack.llm.provider import get_chat_model
from quantstack.graphs.tool_binding import bind_tools_to_llm

from .nodes import (
    make_daily_plan,
    make_data_refresh,
    make_earnings_analysis,
    make_entry_scan,
    make_execute_entries,
    make_execute_exits,
    make_market_intel,
    make_options_analysis,
    make_portfolio_construction,
    make_portfolio_review,
    make_position_review,
    make_reflection,
    make_risk_sizing,
    make_safety_check,
    merge_parallel,
    merge_pre_execution,
)

logger = logging.getLogger(__name__)


def _safety_check_router(state: TradingState) -> str:
    """Route to END if system is halted, daily_plan otherwise."""
    decisions = state.get("decisions", [])
    for d in reversed(decisions):
        if d.get("node") == "safety_check":
            if d.get("halted", False):
                return "halt"
            break
    errors = state.get("errors", [])
    if any("safety_check" in e for e in errors):
        return "halt"
    return "continue"


def _earnings_router(state: TradingState) -> str:
    """Route to earnings_analysis if any symbols have earnings within 14 days."""
    earnings_symbols = state.get("earnings_symbols", [])
    if earnings_symbols:
        return "has_earnings"
    return "no_earnings"


def _risk_gate_router(state: TradingState) -> str:
    """Route based on whether risk_sizing produced alpha signals.

    SafetyGate now runs inside portfolio_construction (post-optimization).
    This router simply checks whether there are candidates to optimize.
    If risk_sizing produced no alpha_signals, skip to END (no candidates).
    """
    alpha_signals = state.get("alpha_signals", [])
    if not alpha_signals:
        return "rejected"
    return "approved"


def build_trading_graph(
    config_watcher: ConfigWatcher,
    checkpointer: BaseCheckpointSaver,
):
    """Build the trading pipeline graph.

    Reads agent configs from config_watcher, creates ChatModel instances
    per agent tier, binds YAML-configured tools to each LLM, and compiles
    the graph.
    """
    planner_cfg = config_watcher.get_config("daily_planner")
    monitor_cfg = config_watcher.get_config("position_monitor")
    exit_cfg = config_watcher.get_config("exit_evaluator")
    debater_cfg = config_watcher.get_config("trade_debater")
    fm_cfg = config_watcher.get_config("fund_manager")
    options_cfg = config_watcher.get_config("options_analyst")
    reflector_cfg = config_watcher.get_config("trade_reflector")
    market_intel_cfg = config_watcher.get_config("market_intel")
    earnings_cfg = config_watcher.get_config("earnings_analyst")

    # Per-agent LLM instances (each agent gets its own tier + thinking config)
    planner_llm_base = get_chat_model(planner_cfg.llm_tier, thinking=planner_cfg.thinking)
    monitor_llm_base = get_chat_model(monitor_cfg.llm_tier, thinking=monitor_cfg.thinking)
    exit_llm_base = get_chat_model(exit_cfg.llm_tier, thinking=exit_cfg.thinking)
    debater_llm_base = get_chat_model(debater_cfg.llm_tier, thinking=debater_cfg.thinking)
    fm_llm_base = get_chat_model(fm_cfg.llm_tier, thinking=fm_cfg.thinking)
    options_llm_base = get_chat_model(options_cfg.llm_tier, thinking=options_cfg.thinking)
    reflector_llm_base = get_chat_model(reflector_cfg.llm_tier, thinking=reflector_cfg.thinking)
    mi_llm_base = get_chat_model(market_intel_cfg.llm_tier, thinking=market_intel_cfg.thinking)
    earnings_llm_base = get_chat_model(earnings_cfg.llm_tier, thinking=earnings_cfg.thinking)

    # Bind tools per agent from YAML config
    planner_llm, planner_tools, _ = bind_tools_to_llm(planner_llm_base, planner_cfg)
    monitor_llm, monitor_tools, _ = bind_tools_to_llm(monitor_llm_base, monitor_cfg)
    exit_llm, exit_tools, _ = bind_tools_to_llm(exit_llm_base, exit_cfg)
    debater_llm, debater_tools, _ = bind_tools_to_llm(debater_llm_base, debater_cfg)
    fm_llm, fm_tools, _ = bind_tools_to_llm(fm_llm_base, fm_cfg)
    options_llm, options_tools, _ = bind_tools_to_llm(options_llm_base, options_cfg)
    reflector_llm, reflector_tools, _ = bind_tools_to_llm(reflector_llm_base, reflector_cfg)
    mi_llm, mi_tools, _ = bind_tools_to_llm(mi_llm_base, market_intel_cfg)
    earnings_llm, earnings_tools, _ = bind_tools_to_llm(earnings_llm_base, earnings_cfg)

    graph = StateGraph(TradingState)

    # Data refresh (deterministic, no LLM — runs every cycle to keep market data fresh)
    graph.add_node("data_refresh", make_data_refresh())

    # Critical node (no retry — fail fast)
    graph.add_node("safety_check", make_safety_check(planner_llm_base, planner_cfg))

    # Pre-market intelligence (runs between safety_check and plan_day)
    graph.add_node("market_intel", make_market_intel(mi_llm, market_intel_cfg, mi_tools), retry=RetryPolicy(max_attempts=2))

    # Agent nodes with tool access (retry up to 2 times for transient LLM failures)
    graph.add_node("plan_day", make_daily_plan(planner_llm, planner_cfg, planner_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("position_review", make_position_review(
        monitor_llm, monitor_cfg, monitor_tools,
        exit_llm=exit_llm, exit_config=exit_cfg, exit_tools=exit_tools,
    ), retry=RetryPolicy(max_attempts=3))
    graph.add_node("entry_scan", make_entry_scan(debater_llm, debater_cfg, debater_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("portfolio_review", make_portfolio_review(fm_llm, fm_cfg, fm_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("analyze_options", make_options_analysis(options_llm, options_cfg, options_tools), retry=RetryPolicy(max_attempts=3))
    graph.add_node("reflect", make_reflection(reflector_llm, reflector_cfg, reflector_tools), retry=RetryPolicy(max_attempts=3))

    # Tool/execution nodes
    graph.add_node("execute_exits", make_execute_exits(monitor_llm, monitor_cfg, monitor_tools), retry=RetryPolicy(max_attempts=2))
    graph.add_node("execute_entries", make_execute_entries(fm_llm, fm_cfg, fm_tools), retry=RetryPolicy(max_attempts=2))

    # Earnings analysis (conditional — only when earnings_symbols non-empty)
    graph.add_node("earnings_analysis", make_earnings_analysis(earnings_llm, earnings_cfg, earnings_tools), retry=RetryPolicy(max_attempts=2))

    # Risk sizing (deterministic, no LLM — SafetyGate runs in portfolio_construction)
    graph.add_node("risk_sizing", make_risk_sizing())

    # Portfolio construction (deterministic, no LLM)
    graph.add_node("portfolio_construction", make_portfolio_construction())

    # Join nodes (no retry needed)
    graph.add_node("merge_parallel", merge_parallel)
    graph.add_node("merge_pre_execution", merge_pre_execution)

    # --- Edges ---
    graph.add_edge(START, "data_refresh")
    graph.add_edge("data_refresh", "safety_check")
    graph.add_conditional_edges(
        "safety_check",
        _safety_check_router,
        {"continue": "market_intel", "halt": END},
    )

    # market_intel → plan_day
    graph.add_edge("market_intel", "plan_day")

    # Parallel branches from plan_day
    graph.add_edge("plan_day", "position_review")
    graph.add_edge("plan_day", "entry_scan")

    # Position review branch
    graph.add_edge("position_review", "execute_exits")
    graph.add_edge("execute_exits", "merge_parallel")

    # Entry scan → earnings routing
    graph.add_conditional_edges(
        "entry_scan",
        _earnings_router,
        {"has_earnings": "earnings_analysis", "no_earnings": "merge_parallel"},
    )
    graph.add_edge("earnings_analysis", "merge_parallel")

    # Post-merge linear pipeline
    graph.add_edge("merge_parallel", "risk_sizing")

    # Risk gate conditional edge (mandatory, no bypass)
    graph.add_conditional_edges(
        "risk_sizing",
        _risk_gate_router,
        {"approved": "portfolio_construction", "rejected": END},
    )

    # Post-risk-gate: optimizer -> parallel (fund manager + options) -> execution
    graph.add_edge("portfolio_construction", "portfolio_review")
    graph.add_edge("portfolio_construction", "analyze_options")
    graph.add_edge("portfolio_review", "merge_pre_execution")
    graph.add_edge("analyze_options", "merge_pre_execution")
    graph.add_edge("merge_pre_execution", "execute_entries")
    graph.add_edge("execute_entries", "reflect")
    graph.add_edge("reflect", END)

    return graph.compile(checkpointer=checkpointer)
