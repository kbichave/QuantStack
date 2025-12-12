# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tools module for CrewAI agent tools.

This module exports all available tools for use by CrewAI agents.

Tool Categories:
- eTrade MCP: Trading and account management
- QuantCore MCP: Technical analysis, backtesting, options, risk management
- Knowledge: Agent knowledge sharing (observations, signals, scenarios)
- Memory: Mem0 semantic memory integration
- Alpha Vantage: News and calendar data
"""

from quant_pod.tools.mcp_bridge import (
    # Bridge class
    MCPBridge,
    get_bridge,
    # eTrade tools
    get_quote_tool,
    get_option_chains_tool,
    preview_order_tool,
    place_order_tool,
    get_positions_tool,
    get_account_balance_tool,
    # QuantCore - Market Data
    fetch_market_data_tool,
    load_market_data_tool,
    list_stored_symbols_tool,
    get_symbol_snapshot_tool,
    # QuantCore - Technical Analysis
    compute_indicators_tool,
    compute_all_features_tool,
    list_available_indicators_tool,
    # QuantCore - Backtesting
    run_backtest_tool,
    get_backtest_metrics_tool,
    run_monte_carlo_tool,
    run_walkforward_tool,
    # QuantCore - Statistical Analysis
    run_adf_test_tool,
    compute_alpha_decay_tool,
    compute_information_coefficient_tool,
    validate_signal_tool,
    diagnose_signal_tool,
    # QuantCore - Options
    price_option_tool,
    compute_greeks_tool,
    compute_implied_vol_tool,
    analyze_option_structure_tool,
    compute_option_chain_tool,
    compute_multi_leg_price_tool,
    # QuantCore - Risk Management
    compute_position_size_tool,
    compute_max_drawdown_tool,
    compute_portfolio_stats_tool,
    compute_var_tool,
    stress_test_portfolio_tool,
    check_risk_limits_tool,
    analyze_liquidity_tool,
    # QuantCore - Market/Regime
    get_market_regime_snapshot_tool,
    analyze_volume_profile_tool,
    get_trading_calendar_tool,
    get_event_calendar_tool,
    # QuantCore - Trade
    generate_trade_template_tool,
    validate_trade_tool,
    score_trade_structure_tool,
    simulate_trade_outcome_tool,
    run_screener_tool,
)
from quant_pod.tools.knowledge_tools import (
    save_observation_tool,
    get_observations_tool,
    save_signal_tool,
    get_signals_tool,
    save_wave_scenario_tool,
    get_wave_scenarios_tool,
)
from quant_pod.tools.memory_tools import (
    store_memory_tool,
    search_memory_tool,
    get_recent_memory_tool,
)
from quant_pod.tools.alphavantage_tools import (
    fetch_news_sentiment_tool,
    fetch_earnings_calendar_tool,
    fetch_upcoming_earnings_tool,
    fetch_ipo_calendar_tool,
    fetch_company_overview_tool,
)

__all__ = [
    # Bridge
    "MCPBridge",
    "get_bridge",
    # eTrade MCP tools
    "get_quote_tool",
    "get_option_chains_tool",
    "preview_order_tool",
    "place_order_tool",
    "get_positions_tool",
    "get_account_balance_tool",
    # QuantCore - Market Data tools
    "fetch_market_data_tool",
    "load_market_data_tool",
    "list_stored_symbols_tool",
    "get_symbol_snapshot_tool",
    # QuantCore - Technical Analysis tools
    "compute_indicators_tool",
    "compute_all_features_tool",
    "list_available_indicators_tool",
    # QuantCore - Backtesting tools
    "run_backtest_tool",
    "get_backtest_metrics_tool",
    "run_monte_carlo_tool",
    "run_walkforward_tool",
    # QuantCore - Statistical Analysis tools
    "run_adf_test_tool",
    "compute_alpha_decay_tool",
    "compute_information_coefficient_tool",
    "validate_signal_tool",
    "diagnose_signal_tool",
    # QuantCore - Options tools
    "price_option_tool",
    "compute_greeks_tool",
    "compute_implied_vol_tool",
    "analyze_option_structure_tool",
    "compute_option_chain_tool",
    "compute_multi_leg_price_tool",
    # QuantCore - Risk Management tools
    "compute_position_size_tool",
    "compute_max_drawdown_tool",
    "compute_portfolio_stats_tool",
    "compute_var_tool",
    "stress_test_portfolio_tool",
    "check_risk_limits_tool",
    "analyze_liquidity_tool",
    # QuantCore - Market/Regime tools
    "get_market_regime_snapshot_tool",
    "analyze_volume_profile_tool",
    "get_trading_calendar_tool",
    "get_event_calendar_tool",
    # QuantCore - Trade tools
    "generate_trade_template_tool",
    "validate_trade_tool",
    "score_trade_structure_tool",
    "simulate_trade_outcome_tool",
    "run_screener_tool",
    # Knowledge tools
    "save_observation_tool",
    "get_observations_tool",
    "save_signal_tool",
    "get_signals_tool",
    "save_wave_scenario_tool",
    "get_wave_scenarios_tool",
    # Memory tools
    "store_memory_tool",
    "search_memory_tool",
    "get_recent_memory_tool",
    # Alpha Vantage tools
    "fetch_news_sentiment_tool",
    "fetch_earnings_calendar_tool",
    "fetch_upcoming_earnings_tool",
    "fetch_ipo_calendar_tool",
    "fetch_company_overview_tool",
]
