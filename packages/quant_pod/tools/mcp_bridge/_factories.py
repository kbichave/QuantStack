# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Factory functions that return singleton-like tool instances."""

from .tools_analysis import (
    AnalyzeOptionStructureTool,
    ComputeAllFeaturesTool,
    ComputeAlphaDecayTool,
    ComputeGreeksTool,
    ComputeImpliedVolTool,
    ComputeIndicatorsTool,
    ComputeInformationCoefficientTool,
    ComputeMultiLegPriceTool,
    ComputeOptionChainTool,
    DiagnoseSignalTool,
    GetBacktestMetricsTool,
    ListAvailableIndicatorsTool,
    PriceOptionTool,
    RunADFTestTool,
    RunBacktestTool,
    RunMonteCarloTool,
    RunWalkForwardTool,
    ValidateSignalTool,
)
from .tools_data import (
    FetchMarketDataTool,
    GetSymbolSnapshotTool,
    ListStoredSymbolsTool,
    LoadMarketDataTool,
)
from .tools_etrade import (
    GetAccountBalanceTool,
    GetOptionChainsTool,
    GetPositionsTool,
    GetQuoteTool,
    PlaceOrderTool,
    PreviewOrderTool,
)
from .tools_market import (
    AnalyzeVolumeProfileTool,
    GenerateTradeTemplateTool,
    GetEventCalendarTool,
    GetMarketRegimeSnapshotTool,
    GetTradingCalendarTool,
    RunScreenerTool,
    ScoreTradeStructureTool,
    SimulateTradeOutcomeTool,
    ValidateTradeTool,
)
from .tools_risk import (
    AnalyzeLiquidityTool,
    CheckRiskLimitsTool,
    ComputeMaxDrawdownTool,
    ComputePortfolioStatsTool,
    ComputePositionSizeTool,
    ComputeVaRTool,
    StressTestPortfolioTool,
)


# =============================================================================
# TOOL FACTORY FUNCTIONS - eTrade
# =============================================================================


def get_quote_tool() -> GetQuoteTool:
    """Get the quote tool instance."""
    return GetQuoteTool()


def get_option_chains_tool() -> GetOptionChainsTool:
    """Get the option chains tool instance."""
    return GetOptionChainsTool()


def preview_order_tool() -> PreviewOrderTool:
    """Get the preview order tool instance."""
    return PreviewOrderTool()


def place_order_tool() -> PlaceOrderTool:
    """Get the place order tool instance."""
    return PlaceOrderTool()


def get_positions_tool() -> GetPositionsTool:
    """Get the positions tool instance."""
    return GetPositionsTool()


def get_account_balance_tool() -> GetAccountBalanceTool:
    """Get the account balance tool instance."""
    return GetAccountBalanceTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Market Data
# =============================================================================


def fetch_market_data_tool() -> FetchMarketDataTool:
    """Get the fetch market data tool instance."""
    return FetchMarketDataTool()


def load_market_data_tool() -> LoadMarketDataTool:
    """Get the load market data tool instance."""
    return LoadMarketDataTool()


def list_stored_symbols_tool() -> ListStoredSymbolsTool:
    """Get the list stored symbols tool instance."""
    return ListStoredSymbolsTool()


def get_symbol_snapshot_tool() -> GetSymbolSnapshotTool:
    """Get the symbol snapshot tool instance."""
    return GetSymbolSnapshotTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Technical Analysis
# =============================================================================


def compute_indicators_tool() -> ComputeIndicatorsTool:
    """Get the indicators tool instance."""
    return ComputeIndicatorsTool()


def compute_all_features_tool() -> ComputeAllFeaturesTool:
    """Get the all features tool instance."""
    return ComputeAllFeaturesTool()


def list_available_indicators_tool() -> ListAvailableIndicatorsTool:
    """Get the list indicators tool instance."""
    return ListAvailableIndicatorsTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Backtesting
# =============================================================================


def run_backtest_tool() -> RunBacktestTool:
    """Get the backtest tool instance."""
    return RunBacktestTool()


def get_backtest_metrics_tool() -> GetBacktestMetricsTool:
    """Get the backtest metrics tool instance."""
    return GetBacktestMetricsTool()


def run_monte_carlo_tool() -> RunMonteCarloTool:
    """Get the Monte Carlo tool instance."""
    return RunMonteCarloTool()


def run_walkforward_tool() -> RunWalkForwardTool:
    """Get the walk-forward tool instance."""
    return RunWalkForwardTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Statistical
# =============================================================================


def run_adf_test_tool() -> RunADFTestTool:
    """Get the ADF test tool instance."""
    return RunADFTestTool()


def compute_alpha_decay_tool() -> ComputeAlphaDecayTool:
    """Get the alpha decay tool instance."""
    return ComputeAlphaDecayTool()


def compute_information_coefficient_tool() -> ComputeInformationCoefficientTool:
    """Get the IC tool instance."""
    return ComputeInformationCoefficientTool()


def validate_signal_tool() -> ValidateSignalTool:
    """Get the validate signal tool instance."""
    return ValidateSignalTool()


def diagnose_signal_tool() -> DiagnoseSignalTool:
    """Get the diagnose signal tool instance."""
    return DiagnoseSignalTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Options
# =============================================================================


def price_option_tool() -> PriceOptionTool:
    """Get the price option tool instance."""
    return PriceOptionTool()


def compute_greeks_tool() -> ComputeGreeksTool:
    """Get the Greeks tool instance."""
    return ComputeGreeksTool()


def compute_implied_vol_tool() -> ComputeImpliedVolTool:
    """Get the implied vol tool instance."""
    return ComputeImpliedVolTool()


def analyze_option_structure_tool() -> AnalyzeOptionStructureTool:
    """Get the option structure analysis tool instance."""
    return AnalyzeOptionStructureTool()


def compute_option_chain_tool() -> ComputeOptionChainTool:
    """Get the option chain tool instance."""
    return ComputeOptionChainTool()


def compute_multi_leg_price_tool() -> ComputeMultiLegPriceTool:
    """Get the multi-leg price tool instance."""
    return ComputeMultiLegPriceTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Risk Management
# =============================================================================


def compute_position_size_tool() -> ComputePositionSizeTool:
    """Get the position size tool instance."""
    return ComputePositionSizeTool()


def compute_max_drawdown_tool() -> ComputeMaxDrawdownTool:
    """Get the max drawdown tool instance."""
    return ComputeMaxDrawdownTool()


def compute_portfolio_stats_tool() -> ComputePortfolioStatsTool:
    """Get the portfolio stats tool instance."""
    return ComputePortfolioStatsTool()


def compute_var_tool() -> ComputeVaRTool:
    """Get the VaR tool instance."""
    return ComputeVaRTool()


def stress_test_portfolio_tool() -> StressTestPortfolioTool:
    """Get the stress test tool instance."""
    return StressTestPortfolioTool()


def check_risk_limits_tool() -> CheckRiskLimitsTool:
    """Get the risk limits tool instance."""
    return CheckRiskLimitsTool()


def analyze_liquidity_tool() -> AnalyzeLiquidityTool:
    """Get the liquidity analysis tool instance."""
    return AnalyzeLiquidityTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Market/Regime
# =============================================================================


def get_market_regime_snapshot_tool() -> GetMarketRegimeSnapshotTool:
    """Get the market regime snapshot tool instance."""
    return GetMarketRegimeSnapshotTool()


def analyze_volume_profile_tool() -> AnalyzeVolumeProfileTool:
    """Get the volume profile tool instance."""
    return AnalyzeVolumeProfileTool()


def get_trading_calendar_tool() -> GetTradingCalendarTool:
    """Get the trading calendar tool instance."""
    return GetTradingCalendarTool()


def get_event_calendar_tool() -> GetEventCalendarTool:
    """Get the event calendar tool instance."""
    return GetEventCalendarTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Trade
# =============================================================================


def generate_trade_template_tool() -> GenerateTradeTemplateTool:
    """Get the trade template tool instance."""
    return GenerateTradeTemplateTool()


def validate_trade_tool() -> ValidateTradeTool:
    """Get the validate trade tool instance."""
    return ValidateTradeTool()


def score_trade_structure_tool() -> ScoreTradeStructureTool:
    """Get the score trade tool instance."""
    return ScoreTradeStructureTool()


def simulate_trade_outcome_tool() -> SimulateTradeOutcomeTool:
    """Get the simulate trade tool instance."""
    return SimulateTradeOutcomeTool()


def run_screener_tool() -> RunScreenerTool:
    """Get the screener tool instance."""
    return RunScreenerTool()
