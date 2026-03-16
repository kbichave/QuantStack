# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic input schemas for all MCP Bridge tool classes."""

from pydantic import BaseModel, Field


# =============================================================================
# INPUT SCHEMAS - eTrade Tools
# =============================================================================


class QuoteInput(BaseModel):
    """Input for get_quote tool."""

    symbols: str = Field(..., description="Comma-separated list of symbols (e.g., 'SPY,AAPL,MSFT')")


class OptionChainInput(BaseModel):
    """Input for get_option_chains tool."""

    symbol: str = Field(..., description="Underlying symbol")
    expiration_date: str | None = Field(None, description="Expiration date (YYYY-MM-DD)")
    strike_price_near: float | None = Field(None, description="Center strikes around this price")
    no_of_strikes: int = Field(10, description="Number of strikes to return")


class PreviewOrderInput(BaseModel):
    """Input for preview_order tool."""

    account_id_key: str = Field(..., description="Account ID key")
    symbol: str = Field(..., description="Symbol to trade")
    action: str = Field(..., description="BUY, SELL, BUY_TO_OPEN, etc.")
    quantity: int = Field(..., description="Number of shares/contracts")
    order_type: str = Field("LIMIT", description="MARKET, LIMIT, STOP")
    limit_price: float | None = Field(None, description="Limit price")


class PlaceOrderInput(BaseModel):
    """Input for place_order tool."""

    account_id_key: str = Field(..., description="Account ID key")
    symbol: str = Field(..., description="Symbol to trade")
    action: str = Field(..., description="BUY, SELL, BUY_TO_OPEN, etc.")
    quantity: int = Field(..., description="Number of shares/contracts")
    order_type: str = Field("LIMIT", description="MARKET, LIMIT, STOP")
    limit_price: float | None = Field(None, description="Limit price")
    preview_id: str | None = Field(None, description="Preview ID from preview_order")


class PositionsInput(BaseModel):
    """Input for get_positions tool."""

    account_id_key: str = Field(..., description="Account ID key")
    symbol: str | None = Field(None, description="Optional symbol filter")


class BalanceInput(BaseModel):
    """Input for get_account_balance tool."""

    account_id_key: str = Field(..., description="Account ID key")


# =============================================================================
# INPUT SCHEMAS - QuantCore Market Data Tools
# =============================================================================


class FetchMarketDataInput(BaseModel):
    """Input for fetch_market_data tool."""

    symbol: str = Field(..., description="Stock/ETF symbol (e.g., 'SPY', 'AAPL')")
    timeframe: str = Field("daily", description="Data frequency: daily, 1h, 4h, weekly")
    outputsize: str = Field("compact", description="'compact' (100 bars) or 'full' (20+ years)")


class LoadMarketDataInput(BaseModel):
    """Input for load_market_data tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Timeframe: 1h, 4h, daily, weekly")
    start_date: str | None = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: str | None = Field(None, description="End date (YYYY-MM-DD)")


class EmptyInput(BaseModel):
    """Input schema for tools that take no parameters."""

    pass


class MarketRegimeSnapshotInput(BaseModel):
    """Input for get_market_regime_snapshot tool."""

    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class SymbolInput(BaseModel):
    """Input for symbol-based tools."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Timeframe: 1h, 4h, daily, weekly")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class IndicatorsInput(BaseModel):
    """Input for compute_indicators tool."""

    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field("daily", description="Timeframe: 1h, 4h, daily, weekly")
    indicators: list[str] | None = Field(
        None, description="List of indicators (RSI, MACD, ATR, etc.)"
    )
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


# =============================================================================
# INPUT SCHEMAS - QuantCore Backtesting Tools
# =============================================================================


class BacktestInput(BaseModel):
    """Input for run_backtest tool."""

    symbol: str = Field(..., description="Symbol to backtest")
    strategy_type: str = Field(
        "mean_reversion", description="mean_reversion, trend_following, momentum"
    )
    timeframe: str = Field("daily", description="Timeframe")
    initial_capital: float = Field(100000, description="Starting capital")
    position_size_pct: float = Field(10, description="Position size as % of equity")
    stop_loss_atr: float = Field(2, description="Stop loss in ATR multiples")
    take_profit_atr: float = Field(3, description="Take profit in ATR multiples")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class BacktestMetricsInput(BaseModel):
    """Input for get_backtest_metrics tool."""

    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Risk-adjusted return metric")
    max_drawdown: float = Field(..., description="Maximum peak-to-trough decline")
    win_rate: float = Field(..., description="Percentage of winning trades")
    total_trades: int = Field(..., description="Total number of trades")


class MonteCarloInput(BaseModel):
    """Input for run_monte_carlo tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Data timeframe")
    n_simulations: int = Field(1000, description="Number of simulations to run")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class WalkForwardInput(BaseModel):
    """Input for run_walkforward tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Timeframe")
    n_splits: int = Field(5, description="Number of train/test splits")
    train_pct: float = Field(0.7, description="Percentage for training")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


# =============================================================================
# INPUT SCHEMAS - QuantCore Statistical Tools
# =============================================================================


class ADFTestInput(BaseModel):
    """Input for run_adf_test tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Data timeframe")
    column: str = Field("close", description="Column to test: close, returns, spread")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class AlphaDecayInput(BaseModel):
    """Input for compute_alpha_decay tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Data timeframe")
    signal_column: str = Field("rsi_14", description="Feature to analyze as signal")
    max_lag: int = Field(20, description="Maximum forward lag to analyze")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class ICInput(BaseModel):
    """Input for compute_information_coefficient tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Data timeframe")
    signal_column: str = Field("rsi_14", description="Feature to analyze")
    forward_return_periods: int = Field(5, description="Forward return horizon in bars")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class ValidateSignalInput(BaseModel):
    """Input for validate_signal tool."""

    symbol: str = Field(..., description="Stock symbol")
    signal_column: str = Field(..., description="Signal column to validate")
    timeframe: str = Field("daily", description="Timeframe")


class DiagnoseSignalInput(BaseModel):
    """Input for diagnose_signal tool."""

    symbol: str = Field(..., description="Stock symbol")
    signal_column: str = Field(..., description="Signal column to diagnose")
    timeframe: str = Field("daily", description="Timeframe")


# =============================================================================
# INPUT SCHEMAS - QuantCore Options Tools
# =============================================================================


class PriceOptionInput(BaseModel):
    """Input for price_option tool."""

    spot: float = Field(..., description="Current stock price")
    strike: float = Field(..., description="Option strike price")
    time_to_expiry: float = Field(
        ..., description="Time to expiration in years (e.g., 0.25 for 3 months)"
    )
    volatility: float = Field(..., description="Annualized volatility (e.g., 0.20 for 20%)")
    risk_free_rate: float = Field(0.05, description="Risk-free interest rate")
    option_type: str = Field("call", description="'call' or 'put'")
    dividend_yield: float = Field(0, description="Continuous dividend yield")


class ComputeGreeksInput(BaseModel):
    """Input for compute_greeks tool."""

    spot: float = Field(..., description="Current stock price")
    strike: float = Field(..., description="Option strike price")
    time_to_expiry: float = Field(..., description="Time to expiration in years")
    volatility: float = Field(..., description="Annualized volatility")
    risk_free_rate: float = Field(0.05, description="Risk-free interest rate")
    option_type: str = Field("call", description="'call' or 'put'")
    dividend_yield: float = Field(0, description="Continuous dividend yield")


class ImpliedVolInput(BaseModel):
    """Input for compute_implied_vol tool."""

    option_price: float = Field(..., description="Market option price")
    spot: float = Field(..., description="Current stock price")
    strike: float = Field(..., description="Strike price")
    time_to_expiry: float = Field(..., description="Time to expiry in years")
    risk_free_rate: float = Field(0.05, description="Risk-free rate")
    option_type: str = Field("call", description="'call' or 'put'")


class AnalyzeOptionStructureInput(BaseModel):
    """Input for analyze_option_structure tool."""

    structure_type: str = Field(..., description="VERTICAL_SPREAD, IRON_CONDOR, STRADDLE, etc.")
    legs: str = Field(..., description="JSON array of legs with strike, option_type, quantity")
    spot: float = Field(..., description="Current spot price")
    volatility: float = Field(0.25, description="Implied volatility")
    time_to_expiry: float = Field(0.083, description="Time to expiry in years")


class ComputeOptionChainInput(BaseModel):
    """Input for compute_option_chain tool."""

    symbol: str = Field(..., description="Underlying symbol")
    spot_price: float = Field(..., description="Current spot price")
    volatility: float = Field(0.25, description="Implied volatility")
    days_to_expiry: int = Field(30, description="Days until expiration")
    num_strikes: int = Field(10, description="Number of strikes each side of ATM")


class MultiLegPriceInput(BaseModel):
    """Input for compute_multi_leg_price tool."""

    legs: str = Field(
        ..., description="JSON array of legs with strike, option_type, quantity, price"
    )
    spot: float = Field(..., description="Current spot price")


# =============================================================================
# INPUT SCHEMAS - QuantCore Risk Management Tools
# =============================================================================


class PositionSizeInput(BaseModel):
    """Input for compute_position_size tool."""

    equity: float = Field(..., description="Total account equity")
    entry_price: float = Field(..., description="Planned entry price")
    stop_loss_price: float = Field(..., description="Stop loss price level")
    risk_per_trade_pct: float = Field(1, description="Percentage of equity to risk per trade")
    max_position_pct: float = Field(20, description="Maximum position as % of equity")
    alignment_score: float = Field(1, description="Cross-timeframe alignment score (0-1)")


class MaxDrawdownInput(BaseModel):
    """Input for compute_max_drawdown tool."""

    equity_curve: list[float] = Field(..., description="List of equity values over time")


class PortfolioStatsInput(BaseModel):
    """Input for compute_portfolio_stats tool."""

    returns: list[float] = Field(..., description="List of period returns")
    risk_free_rate: float = Field(0.02, description="Annual risk-free rate")


class VaRInput(BaseModel):
    """Input for compute_var tool."""

    returns: list[float] = Field(..., description="Historical returns")
    confidence_level: float = Field(0.95, description="VaR confidence level")
    portfolio_value: float = Field(100000, description="Portfolio value")


class StressTestInput(BaseModel):
    """Input for stress_test_portfolio tool."""

    positions: str = Field(
        ..., description="JSON array of positions with symbol, quantity, delta, gamma"
    )
    scenarios: str = Field(
        ..., description="JSON array of scenarios with name, price_change, vol_change"
    )


class RiskLimitsInput(BaseModel):
    """Input for check_risk_limits tool."""

    portfolio_delta: float = Field(..., description="Net portfolio delta")
    portfolio_gamma: float = Field(..., description="Net portfolio gamma")
    portfolio_vega: float = Field(..., description="Net portfolio vega")
    max_delta: float = Field(100, description="Maximum allowed delta")
    max_gamma: float = Field(50, description="Maximum allowed gamma")
    max_vega: float = Field(5000, description="Maximum allowed vega")


class LiquidityInput(BaseModel):
    """Input for analyze_liquidity tool."""

    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field("daily", description="Timeframe")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


# =============================================================================
# INPUT SCHEMAS - QuantCore Market/Regime Tools
# =============================================================================


class SymbolSnapshotInput(BaseModel):
    """Input for get_symbol_snapshot tool."""

    symbol: str = Field(..., description="Stock symbol")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class VolumeProfileInput(BaseModel):
    """Input for analyze_volume_profile tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Timeframe")
    num_bins: int = Field(20, description="Number of price bins")
    end_date: str | None = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class TradingCalendarInput(BaseModel):
    """Input for get_trading_calendar tool."""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class EventCalendarInput(BaseModel):
    """Input for get_event_calendar tool."""

    symbol: str | None = Field(None, description="Optional symbol filter")
    days_ahead: int = Field(7, description="Days to look ahead")


# =============================================================================
# INPUT SCHEMAS - QuantCore Trade Tools
# =============================================================================


class TradeTemplateInput(BaseModel):
    """Input for generate_trade_template tool."""

    symbol: str = Field(..., description="Stock symbol")
    direction: str = Field(..., description="LONG or SHORT")
    structure_type: str = Field("VERTICAL_SPREAD", description="Option structure type")
    max_risk: float = Field(500, description="Maximum dollar risk")


class ValidateTradeInput(BaseModel):
    """Input for validate_trade tool."""

    symbol: str = Field(..., description="Stock symbol")
    direction: str = Field(..., description="LONG or SHORT")
    entry_price: float = Field(..., description="Entry price")
    stop_loss: float = Field(..., description="Stop loss price")
    position_size: float = Field(..., description="Position size in dollars")


class ScoreTradeInput(BaseModel):
    """Input for score_trade_structure tool."""

    structure_type: str = Field(..., description="Option structure type")
    max_profit: float = Field(..., description="Maximum profit potential")
    max_loss: float = Field(..., description="Maximum loss potential")
    probability_of_profit: float = Field(..., description="POP estimate (0-1)")
    days_to_expiry: int = Field(..., description="DTE")


class SimulateTradeInput(BaseModel):
    """Input for simulate_trade_outcome tool."""

    symbol: str = Field(..., description="Stock symbol")
    entry_price: float = Field(..., description="Entry price")
    stop_loss: float = Field(..., description="Stop loss")
    take_profit: float = Field(..., description="Take profit")
    position_size: int = Field(..., description="Number of shares")
    days_to_hold: int = Field(20, description="Holding period")


class ScreenerInput(BaseModel):
    """Input for run_screener tool."""

    min_price: float = Field(10, description="Minimum stock price")
    max_price: float = Field(500, description="Maximum stock price")
    min_volume: int = Field(1000000, description="Minimum daily volume")
    rsi_oversold: float = Field(30, description="RSI oversold threshold")
    rsi_overbought: float = Field(70, description="RSI overbought threshold")
