"""Central tool registry mapping string names to tool objects.

Graph builders look up tools by name from agent YAML configs, enabling
YAML-driven tool assignment without hardcoded imports in graph code.

Only LLM-facing tools appear here. Node-callable functions are imported
directly by node implementations.
"""

from langchain_core.tools import BaseTool

# Signal & analysis
from quantstack.tools.langchain.signal_tools import signal_brief, multi_signal_brief

# Data
from quantstack.tools.langchain.data_tools import (
    check_listing_status,
    fetch_market_data,
    fetch_fundamentals,
    fetch_earnings_data,
    get_commodity_signals,
    get_earnings_momentum,
    get_forex_rates,
    get_put_call_ratio,
    load_market_data,
    list_stored_symbols,
    get_company_facts,
    get_analyst_estimates,
    screen_stocks,
)

# Portfolio
from quantstack.tools.langchain.portfolio_tools import fetch_portfolio

# Options
from quantstack.tools.langchain.options_tools import (
    fetch_options_chain,
    compute_greeks,
    price_option,
    compute_implied_vol,
    analyze_option_structure,
    get_iv_surface,
    score_trade_structure,
    simulate_trade_outcome,
)

# Risk
from quantstack.tools.langchain.risk_tools import (
    compute_risk_metrics,
    compute_position_size,
    compute_var,
    stress_test_portfolio,
    check_risk_limits,
    compute_max_drawdown,
)

# Execution
from quantstack.tools.langchain.execution_tools import (
    execute_order,
    close_position,
    get_fills,
    get_audit_trail,
    update_position_stops,
    check_broker_connection,
)

# Backtesting
from quantstack.tools.langchain.backtest_tools import (
    run_backtest,
    run_walkforward,
    run_backtest_options,
)

# ML
from quantstack.tools.langchain.ml_tools import (
    train_model,
    compute_features,
    predict_ml_signal,
    get_ml_model_status,
    check_concept_drift,
)

# Intelligence
from quantstack.tools.langchain.intelligence_tools import web_search

# Knowledge / Learning
from quantstack.tools.langchain.learning_tools import (
    search_knowledge_base,
    promote_strategy,
    retire_strategy,
    get_strategy_performance,
    validate_strategy,
)

# Strategy
from quantstack.tools.langchain.strategy_tools import (
    fetch_strategy_registry,
    register_strategy,
    get_strategy,
    update_strategy,
)

# Calendar
from quantstack.tools.langchain.calendar_tools import get_trading_calendar


def _try_import(module: str, names: list[str]) -> dict[str, BaseTool]:
    """Best-effort import of tools from a module. Returns {name: tool} for successful imports."""
    result = {}
    try:
        mod = __import__(module, fromlist=names)
        for name in names:
            obj = getattr(mod, name, None)
            if obj is not None:
                result[name] = obj
    except ImportError:
        pass
    return result


# System (supervisor graph)
_system_tools = _try_import("quantstack.tools.langchain.system_tools", [
    "check_system_status", "check_heartbeat", "get_recent_decisions",
])

# Batch 1 new modules (created by agents)
_alert_tools = _try_import("quantstack.tools.langchain.alert_tools", [
    "create_equity_alert", "get_equity_alerts", "update_alert_status",
    "create_exit_signal", "add_alert_update",
])
_attribution_tools = _try_import("quantstack.tools.langchain.attribution_tools", [
    "get_daily_equity", "get_strategy_pnl",
])
_capitulation_tools = _try_import("quantstack.tools.langchain.capitulation_tools", [
    "get_capitulation_score",
])
_cross_domain_tools = _try_import("quantstack.tools.langchain.cross_domain_tools", [
    "get_cross_domain_intel",
])
_decoder_tools = _try_import("quantstack.tools.langchain.decoder_tools", [
    "decode_strategy", "decode_from_trades",
])
_feedback_tools = _try_import("quantstack.tools.langchain.feedback_tools", [
    "get_fill_quality", "get_position_monitor",
])
_institutional_tools = _try_import("quantstack.tools.langchain.institutional_tools", [
    "get_institutional_accumulation",
])
_intraday_tools = _try_import("quantstack.tools.langchain.intraday_tools", [
    "get_intraday_status", "get_tca_report", "get_algo_recommendation",
])
_macro_tools = _try_import("quantstack.tools.langchain.macro_tools", [
    "get_credit_market_signals", "get_market_breadth",
])

# Batch 2 new modules
_nlp_tools = _try_import("quantstack.tools.langchain.nlp_tools", [
    "analyze_text_sentiment",
])
_options_exec_tools = _try_import("quantstack.tools.langchain.options_execution_tools", [
    "execute_options_trade",
])
_meta_tools = _try_import("quantstack.tools.langchain.meta_tools", [
    "get_regime_strategies", "set_regime_allocation", "resolve_portfolio_conflicts",
    "get_strategy_gaps", "promote_draft_strategies", "check_strategy_rules",
])
_finrl_tools = _try_import("quantstack.tools.langchain.finrl_tools", [
    "finrl_create_environment", "finrl_train_model", "finrl_train_ensemble",
    "finrl_evaluate_model", "finrl_predict", "finrl_list_models",
    "finrl_compare_models", "finrl_get_model_status", "finrl_promote_model",
    "finrl_screen_stocks", "finrl_screen_options",
])
_qc_acquisition_tools = _try_import("quantstack.tools.langchain.qc_acquisition_tools", [
    "acquire_historical_data", "register_ticker",
])
_qc_backtesting_tools = _try_import("quantstack.tools.langchain.qc_backtesting_tools", [
    "run_backtest_template", "get_backtest_metrics", "run_walkforward_template",
    "run_purged_cv",
])
_qc_indicator_tools = _try_import("quantstack.tools.langchain.qc_indicator_tools", [
    "compute_technical_indicators", "list_available_indicators",
    "compute_feature_matrix", "compute_quantagent_features",
])
_qc_research_tools = _try_import("quantstack.tools.langchain.qc_research_tools", [
    "run_adf_test", "compute_alpha_decay", "compute_information_coefficient",
    "run_monte_carlo", "validate_signal", "diagnose_signal", "detect_leakage",
    "check_lookahead_bias", "fit_garch_model", "forecast_volatility",
    "compute_deflated_sharpe_ratio", "run_combinatorial_purged_cv",
    "compute_probability_of_overfitting",
])
# EWF Elliott Wave tools
_ewf_tools = _try_import("quantstack.tools.langchain.ewf_tools", [
    "get_ewf_analysis",
    "get_ewf_blue_box_setups",
])


TOOL_REGISTRY: dict[str, BaseTool] = {
    # Signal & analysis
    "signal_brief": signal_brief,
    "multi_signal_brief": multi_signal_brief,
    # Data
    "fetch_market_data": fetch_market_data,
    "fetch_fundamentals": fetch_fundamentals,
    "fetch_earnings_data": fetch_earnings_data,
    "load_market_data": load_market_data,
    "list_stored_symbols": list_stored_symbols,
    "get_company_facts": get_company_facts,
    "get_analyst_estimates": get_analyst_estimates,
    "screen_stocks": screen_stocks,
    "get_put_call_ratio": get_put_call_ratio,
    "get_earnings_momentum": get_earnings_momentum,
    "get_commodity_signals": get_commodity_signals,
    "get_forex_rates": get_forex_rates,
    "check_listing_status": check_listing_status,
    # Portfolio
    "fetch_portfolio": fetch_portfolio,
    # Options
    "fetch_options_chain": fetch_options_chain,
    "compute_greeks": compute_greeks,
    "price_option": price_option,
    "compute_implied_vol": compute_implied_vol,
    "analyze_option_structure": analyze_option_structure,
    "get_iv_surface": get_iv_surface,
    "score_trade_structure": score_trade_structure,
    "simulate_trade_outcome": simulate_trade_outcome,
    # Risk
    "compute_risk_metrics": compute_risk_metrics,
    "compute_position_size": compute_position_size,
    "compute_var": compute_var,
    "stress_test_portfolio": stress_test_portfolio,
    "check_risk_limits": check_risk_limits,
    "compute_max_drawdown": compute_max_drawdown,
    # Execution
    "execute_order": execute_order,
    "close_position": close_position,
    "get_fills": get_fills,
    "get_audit_trail": get_audit_trail,
    "update_position_stops": update_position_stops,
    "check_broker_connection": check_broker_connection,
    # Backtesting
    "run_backtest": run_backtest,
    "run_walkforward": run_walkforward,
    "run_backtest_options": run_backtest_options,
    # ML
    "train_model": train_model,
    "compute_features": compute_features,
    "predict_ml_signal": predict_ml_signal,
    "get_ml_model_status": get_ml_model_status,
    "check_concept_drift": check_concept_drift,
    # Intelligence
    "web_search": web_search,
    # Knowledge / Learning
    "search_knowledge_base": search_knowledge_base,
    "promote_strategy": promote_strategy,
    "retire_strategy": retire_strategy,
    "get_strategy_performance": get_strategy_performance,
    "validate_strategy": validate_strategy,
    # Strategy
    "fetch_strategy_registry": fetch_strategy_registry,
    "register_strategy": register_strategy,
    "get_strategy": get_strategy,
    "update_strategy": update_strategy,
    # Calendar
    "get_trading_calendar": get_trading_calendar,
}

# Merge in dynamically-imported tools from new modules
for _tools_dict in [
    _system_tools, _alert_tools, _attribution_tools, _capitulation_tools,
    _cross_domain_tools, _decoder_tools, _feedback_tools, _institutional_tools,
    _intraday_tools, _macro_tools, _nlp_tools, _options_exec_tools, _meta_tools,
    _finrl_tools, _qc_acquisition_tools, _qc_backtesting_tools,
    _qc_indicator_tools, _qc_research_tools, _ewf_tools,
]:
    TOOL_REGISTRY.update(_tools_dict)


from quantstack.graphs.tool_search_compat import TOOL_SEARCH_TOOL, tool_to_anthropic_dict


# Client-side search tool schema — used by non-Anthropic providers (Bedrock, etc.)
# that don't support defer_loading. The agent executor handles this tool specially.
CLIENT_SEARCH_TOOL_NAME = "search_available_tools"


def search_deferred_tools(
    query: str,
    deferred_names: set[str],
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Client-side keyword search over deferred tool descriptions.

    Returns up to max_results tool summaries sorted by relevance (keyword match count).
    """
    query_lower = query.lower()
    query_words = query_lower.split()

    scored: list[tuple[int, str, str]] = []
    for name in deferred_names:
        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            continue
        desc = tool.description.lower()
        # Score: number of query words that appear in the description
        score = sum(1 for w in query_words if w in desc)
        # Bonus for exact phrase match
        if query_lower in desc:
            score += len(query_words)
        if score > 0:
            scored.append((score, name, tool.description))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"name": name, "description": desc[:200]}
        for _, name, desc in scored[:max_results]
    ]


def get_tools_for_agent(tool_names: list[str] | tuple[str, ...]) -> list[BaseTool]:
    """Resolve a list of tool name strings to tool objects.

    Raises KeyError with a clear message if any tool name is not found
    in the registry.
    """
    tools = []
    for name in tool_names:
        if name not in TOOL_REGISTRY:
            available = sorted(TOOL_REGISTRY.keys())
            raise KeyError(
                f"Tool '{name}' not found in TOOL_REGISTRY. "
                f"Available tools: {available}"
            )
        tools.append(TOOL_REGISTRY[name])
    return tools


def get_tools_for_agent_with_search(
    tool_names: list[str] | tuple[str, ...],
    always_loaded: list[str] | tuple[str, ...],
) -> tuple[list[dict], list[BaseTool]]:
    """Partition tools into deferred/always-loaded and return API-ready dicts.

    Args:
        tool_names: All tool names configured for the agent (from agents.yaml tools:).
        always_loaded: Subset of tool_names that must never be deferred
            (from agents.yaml always_loaded_tools:).

    Returns:
        tools_for_api: List of tool schema dicts with defer_loading flags,
            plus the BM25 search tool definition. Pass to llm.bind(tools=...).
        tools_for_execution: All BaseTool objects for tool_names. The agent
            executor needs these to run any tool the LLM discovers via search.
    """
    always_loaded_set = set(always_loaded)
    tool_names_set = set(tool_names)
    invalid = always_loaded_set - tool_names_set
    if invalid:
        raise ValueError(
            f"always_loaded contains tools not in tool_names: {sorted(invalid)}"
        )

    tools_for_execution = get_tools_for_agent(tool_names)
    tool_map = {t.name: t for t in tools_for_execution}

    tools_for_api: list[dict] = []
    for name in tool_names:
        defer = name not in always_loaded_set
        tools_for_api.append(tool_to_anthropic_dict(tool_map[name], defer=defer))

    tools_for_api.append(TOOL_SEARCH_TOOL)

    return tools_for_api, tools_for_execution
