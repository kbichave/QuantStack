"""Central tool registry mapping string names to tool objects.

Graph builders look up tools by name from agent YAML configs, enabling
YAML-driven tool assignment without hardcoded imports in graph code.

Only LLM-facing tools appear here. Node-callable functions are imported
directly by node implementations.
"""

import json

from langchain_core.tools import BaseTool, tool as tool_decorator

from quantstack.tools.langchain.signal_tools import signal_brief, multi_signal_brief
from quantstack.tools.langchain.data_tools import (
    fetch_market_data,
    fetch_fundamentals,
    fetch_earnings_data,
)
from quantstack.tools.langchain.portfolio_tools import fetch_portfolio
from quantstack.tools.langchain.options_tools import fetch_options_chain, compute_greeks
from quantstack.tools.langchain.risk_tools import compute_risk_metrics, compute_position_size
from quantstack.tools.langchain.execution_tools import execute_order
from quantstack.tools.langchain.backtest_tools import run_backtest
from quantstack.tools.langchain.ml_tools import train_model, compute_features
from quantstack.tools.langchain.intelligence_tools import web_search
from quantstack.tools.langchain.learning_tools import search_knowledge_base
from quantstack.tools.langchain.strategy_tools import fetch_strategy_registry


# Supervisor-specific: LLM-facing wrappers for system functions

@tool_decorator
async def check_system_status() -> str:
    """Check overall system health including services, kill switch, and data freshness."""
    from quantstack.tools.functions.system_functions import check_system_status as _fn
    result = await _fn()
    return json.dumps(result, default=str)


@tool_decorator
async def check_heartbeat(service: str) -> str:
    """Check heartbeat freshness for a service (trading-graph, research-graph)."""
    from quantstack.tools.functions.system_functions import check_heartbeat as _fn
    result = await _fn(service=service)
    return json.dumps(result, default=str)


TOOL_REGISTRY: dict[str, BaseTool] = {
    # Signal & analysis
    "signal_brief": signal_brief,
    "multi_signal_brief": multi_signal_brief,
    # Data
    "fetch_market_data": fetch_market_data,
    "fetch_fundamentals": fetch_fundamentals,
    "fetch_earnings_data": fetch_earnings_data,
    # Portfolio
    "fetch_portfolio": fetch_portfolio,
    # Options
    "fetch_options_chain": fetch_options_chain,
    "compute_greeks": compute_greeks,
    # Risk
    "compute_risk_metrics": compute_risk_metrics,
    "compute_position_size": compute_position_size,
    # Execution
    "execute_order": execute_order,
    # Backtesting
    "run_backtest": run_backtest,
    # ML
    "train_model": train_model,
    "compute_features": compute_features,
    # Intelligence
    "web_search": web_search,
    # Knowledge
    "search_knowledge_base": search_knowledge_base,
    # Strategy
    "fetch_strategy_registry": fetch_strategy_registry,
    # Supervisor
    "check_system_status": check_system_status,
    "check_heartbeat": check_heartbeat,
}


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
