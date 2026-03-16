# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI Tools for Trading Crew.

Provides tool configurations for trading crew agents using the existing
MCP bridge tools from quant_pod.tools.mcp_bridge.

NO FALLBACKS - If tools are not available, raise an error.
Agents MUST have their tools to function.

Usage:
    from quant_pod.crews.tools import (
        get_analyst_tools,
        get_risk_tools,
        get_execution_tools,
        get_all_tools,
    )

    # Get tools for a specific analyst type
    trend_tools = get_analyst_tools("trend")

    # Get all tools for the crew
    all_tools = get_all_tools()
"""

from typing import List, Optional

from quant_pod.crewai_compat import BaseTool
from loguru import logger

# RL tools (optional — degrade gracefully when torch/checkpoints unavailable)
try:
    from quantcore.rl.rl_tools import (
        get_rl_tools as _get_rl_tools,
        rl_position_size_tool,
        rl_execution_strategy_tool,
        rl_alpha_weight_tool,
    )
    from quantcore.rl.config import RLProductionConfig, get_rl_config
    RL_TOOLS_AVAILABLE = True
except ImportError:
    RL_TOOLS_AVAILABLE = False
    logger.debug("RL tools not available (quantcore.rl not importable).")


# =============================================================================
# TOOL IMPORTS from MCP Bridge
# =============================================================================

from quant_pod.tools.mcp_bridge import (
    # Market Data Tools
    fetch_market_data_tool,
    load_market_data_tool,
    list_stored_symbols_tool,
    get_symbol_snapshot_tool,
    # Technical Analysis Tools
    compute_indicators_tool,
    compute_all_features_tool,
    list_available_indicators_tool,
    # Risk Tools
    compute_position_size_tool,
    compute_var_tool,
    check_risk_limits_tool,
    stress_test_portfolio_tool,
    compute_max_drawdown_tool,
    compute_portfolio_stats_tool,
    analyze_liquidity_tool,
    # Market/Regime Tools
    get_market_regime_snapshot_tool,
    analyze_volume_profile_tool,
    get_trading_calendar_tool,
    get_event_calendar_tool,
    # Trade Tools
    generate_trade_template_tool,
    validate_trade_tool,
    score_trade_structure_tool,
    simulate_trade_outcome_tool,
    # Backtesting Tools
    run_backtest_tool,
    run_monte_carlo_tool,
    # Statistical Tools
    run_adf_test_tool,
    compute_alpha_decay_tool,
    compute_information_coefficient_tool,
    # Options Tools
    price_option_tool,
    compute_greeks_tool,
    compute_implied_vol_tool,
    analyze_option_structure_tool,
    compute_option_chain_tool,
)

# Tools are available - no fallback needed
MCP_TOOLS_AVAILABLE = True
logger.info("MCP Bridge tools loaded successfully")


# =============================================================================
# TOOL COLLECTIONS by Role
# =============================================================================


def get_market_data_tools() -> List[BaseTool]:
    """Get tools for loading and fetching market data."""
    return [
        load_market_data_tool(),
        fetch_market_data_tool(),
        get_symbol_snapshot_tool(),
        list_stored_symbols_tool(),
    ]


def get_technical_analysis_tools() -> List[BaseTool]:
    """Get tools for technical analysis (indicators, features)."""
    return [
        compute_indicators_tool(),
        compute_all_features_tool(),
        list_available_indicators_tool(),
        get_market_regime_snapshot_tool(),
        analyze_volume_profile_tool(),
        get_symbol_snapshot_tool(),
    ]


def get_analyst_tools(analyst_type: str = "general") -> List[BaseTool]:
    """
    Get tools for a specific analyst type.

    Args:
        analyst_type: One of "trend", "momentum", "volatility", "structure", "general"

    Returns:
        List of relevant tools for that analyst
    """
    # Base tools all analysts need
    base_tools = [
        compute_indicators_tool(),
        get_symbol_snapshot_tool(),
    ]

    if analyst_type == "trend":
        return base_tools + [
            get_market_regime_snapshot_tool(),
            compute_all_features_tool(),
        ]

    elif analyst_type == "momentum":
        return base_tools + [
            compute_all_features_tool(),
        ]

    elif analyst_type == "volatility":
        return base_tools + [
            compute_var_tool(),
            get_market_regime_snapshot_tool(),
        ]

    elif analyst_type == "structure":
        return base_tools + [
            analyze_volume_profile_tool(),
            compute_all_features_tool(),
        ]

    else:  # general
        return base_tools + [
            compute_all_features_tool(),
            get_market_regime_snapshot_tool(),
        ]


def get_risk_tools() -> List[BaseTool]:
    """Get tools for risk management analysis."""
    return [
        compute_position_size_tool(),
        compute_var_tool(),
        check_risk_limits_tool(),
        stress_test_portfolio_tool(),
        compute_max_drawdown_tool(),
        compute_portfolio_stats_tool(),
        analyze_liquidity_tool(),
    ]


def get_execution_tools() -> List[BaseTool]:
    """Get tools for trade execution decisions."""
    return [
        generate_trade_template_tool(),
        validate_trade_tool(),
        score_trade_structure_tool(),
        simulate_trade_outcome_tool(),
        compute_position_size_tool(),
        get_trading_calendar_tool(),
    ]


def get_validation_tools() -> List[BaseTool]:
    """Get tools for strategy validation and backtesting."""
    return [
        run_backtest_tool(),
        run_monte_carlo_tool(),
    ]


def get_statistical_tools() -> List[BaseTool]:
    """Get tools for statistical analysis."""
    return [
        run_adf_test_tool(),
        compute_alpha_decay_tool(),
        compute_information_coefficient_tool(),
    ]


def get_options_tools() -> List[BaseTool]:
    """Get tools for options analysis."""
    return [
        price_option_tool(),
        compute_greeks_tool(),
        compute_implied_vol_tool(),
        analyze_option_structure_tool(),
        compute_option_chain_tool(),
    ]


def get_rl_tools(config: "Optional[RLProductionConfig]" = None) -> List[BaseTool]:
    """
    Get RL agent tools for the trading crew.

    Returns tools for: position sizing (PPO), execution strategy (DQN),
    and alpha weighting (contextual bandit).

    All tools start in shadow mode — output is tagged [SHADOW – not yet validated]
    until the agent passes PromotionGate checks. The LLM agent can read the
    recommendation and decide whether to follow it.

    Only tools whose enable_* flag is True in RLProductionConfig are returned.
    Falls back to [] if RL tools are not available (graceful degradation).
    """
    if not RL_TOOLS_AVAILABLE:
        return []
    try:
        cfg = config or get_rl_config()
        return _get_rl_tools(cfg)
    except Exception as exc:
        logger.warning(f"[RL] Failed to load RL tools (non-fatal): {exc}")
        return []


def get_all_tools() -> List[BaseTool]:
    """Get all available tools for the trading crew."""
    # Deduplicate tools (some appear in multiple categories)
    tool_instances = {}

    for tool_list in [
        get_market_data_tools(),
        get_technical_analysis_tools(),
        get_risk_tools(),
        get_execution_tools(),
        get_statistical_tools(),
        get_options_tools(),
    ]:
        for tool in tool_list:
            tool_instances[tool.name] = tool

    return list(tool_instances.values())


# =============================================================================
# MCPS DSL Configuration (for native CrewAI MCP integration)
# =============================================================================


def get_quantcore_mcps_config() -> dict:
    """
    Get MCP server configuration for CrewAI's native mcps integration.

    This can be used with the `mcps` parameter on Agent if CrewAI's
    direct MCP integration is preferred over BaseTool wrappers.

    Returns:
        Dict with MCP server configuration
    """
    return {
        "quantcore": {
            "command": "python",
            "args": ["-m", "quantcore.mcp.server"],
            "env": {},
        }
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MCP_TOOLS_AVAILABLE",
    "RL_TOOLS_AVAILABLE",
    "get_rl_tools",
    "get_market_data_tools",
    "get_technical_analysis_tools",
    "get_analyst_tools",
    "get_risk_tools",
    "get_execution_tools",
    "get_validation_tools",
    "get_statistical_tools",
    "get_options_tools",
    "get_all_tools",
    "get_quantcore_mcps_config",
    # Individual tool factories for direct use
    "fetch_market_data_tool",
    "load_market_data_tool",
    "list_stored_symbols_tool",
    "get_symbol_snapshot_tool",
    "compute_indicators_tool",
    "compute_all_features_tool",
    "list_available_indicators_tool",
    "compute_position_size_tool",
    "compute_var_tool",
    "check_risk_limits_tool",
    "stress_test_portfolio_tool",
    "compute_max_drawdown_tool",
    "compute_portfolio_stats_tool",
    "analyze_liquidity_tool",
    "get_market_regime_snapshot_tool",
    "analyze_volume_profile_tool",
    "get_trading_calendar_tool",
    "get_event_calendar_tool",
    "generate_trade_template_tool",
    "validate_trade_tool",
    "score_trade_structure_tool",
    "simulate_trade_outcome_tool",
    "run_backtest_tool",
    "run_monte_carlo_tool",
    "run_adf_test_tool",
    "compute_alpha_decay_tool",
    "compute_information_coefficient_tool",
    "price_option_tool",
    "compute_greeks_tool",
    "compute_implied_vol_tool",
    "analyze_option_structure_tool",
    "compute_option_chain_tool",
]
