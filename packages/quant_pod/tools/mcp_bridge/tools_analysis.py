# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Technical analysis, backtesting, statistical, and options tool classes."""

import json

from pydantic import BaseModel

from quant_pod.crewai_compat import BaseTool

from ._bridge import _run_async, get_bridge
from ._schemas import (
    ADFTestInput,
    AlphaDecayInput,
    AnalyzeOptionStructureInput,
    BacktestInput,
    BacktestMetricsInput,
    ComputeGreeksInput,
    ComputeOptionChainInput,
    DiagnoseSignalInput,
    EmptyInput,
    ICInput,
    ImpliedVolInput,
    IndicatorsInput,
    MonteCarloInput,
    MultiLegPriceInput,
    PriceOptionInput,
    SymbolInput,
    ValidateSignalInput,
    WalkForwardInput,
)


# =============================================================================
# QUANTCORE TECHNICAL ANALYSIS TOOL CLASSES
# =============================================================================


class ComputeIndicatorsTool(BaseTool):
    """Tool to compute technical indicators."""

    name: str = "compute_indicators"
    description: str = (
        "Compute technical indicators (RSI, MACD, ATR, Bollinger Bands, etc.) for a symbol."
    )
    args_schema: type[BaseModel] = IndicatorsInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        indicators: list[str] | None = None,
        end_date: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_technical_indicators",
                symbol=symbol,
                timeframe=timeframe,
                indicators=indicators,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeAllFeaturesTool(BaseTool):
    """Tool to compute all 200+ features."""

    name: str = "compute_all_features"
    description: str = "Compute all available features (200+) including trend, momentum, volatility, volume, and patterns."
    args_schema: type[BaseModel] = SymbolInput

    def _run(self, symbol: str, timeframe: str = "daily", end_date: str | None = None) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_all_features",
                symbol=symbol,
                timeframe=timeframe,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ListAvailableIndicatorsTool(BaseTool):
    """Tool to list all available indicators."""

    name: str = "list_available_indicators"
    description: str = "List all available technical indicators with their descriptions."
    args_schema: type[BaseModel] = EmptyInput

    def _run(self) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore("list_available_indicators")

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE BACKTESTING TOOL CLASSES
# =============================================================================


class RunBacktestTool(BaseTool):
    """Tool to run backtests."""

    name: str = "run_backtest"
    description: str = "Run a backtest on historical data to validate a trading strategy."
    args_schema: type[BaseModel] = BacktestInput

    def _run(
        self,
        symbol: str,
        strategy_type: str = "mean_reversion",
        timeframe: str = "daily",
        initial_capital: float = 100000,
        position_size_pct: float = 10,
        stop_loss_atr: float = 2,
        take_profit_atr: float = 3,
        end_date: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_backtest",
                symbol=symbol,
                strategy_type=strategy_type,
                timeframe=timeframe,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                stop_loss_atr=stop_loss_atr,
                take_profit_atr=take_profit_atr,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetBacktestMetricsTool(BaseTool):
    """Tool to analyze backtest metrics."""

    name: str = "get_backtest_metrics"
    description: str = "Analyze and interpret backtest metrics to assess strategy quality."
    args_schema: type[BaseModel] = BacktestMetricsInput

    def _run(
        self,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_trades: int,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_backtest_metrics",
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class RunMonteCarloTool(BaseTool):
    """Tool to run Monte Carlo simulations."""

    name: str = "run_monte_carlo"
    description: str = (
        "Run Monte Carlo simulation to test strategy robustness under varying conditions."
    )
    args_schema: type[BaseModel] = MonteCarloInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        n_simulations: int = 1000,
        end_date: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_monte_carlo",
                symbol=symbol,
                timeframe=timeframe,
                n_simulations=n_simulations,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class RunWalkForwardTool(BaseTool):
    """Tool to run walk-forward optimization."""

    name: str = "run_walkforward"
    description: str = "Run walk-forward optimization to validate strategy out-of-sample."
    args_schema: type[BaseModel] = WalkForwardInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        n_splits: int = 5,
        train_pct: float = 0.7,
        end_date: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_walkforward",
                symbol=symbol,
                timeframe=timeframe,
                n_splits=n_splits,
                train_pct=train_pct,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE STATISTICAL ANALYSIS TOOL CLASSES
# =============================================================================


class RunADFTestTool(BaseTool):
    """Tool to run Augmented Dickey-Fuller stationarity test."""

    name: str = "run_adf_test"
    description: str = "Run ADF test to check if a time series is stationary (mean-reverting). p-value < 0.05 indicates stationarity."
    args_schema: type[BaseModel] = ADFTestInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        column: str = "close",
        end_date: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_adf_test",
                symbol=symbol,
                timeframe=timeframe,
                column=column,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeAlphaDecayTool(BaseTool):
    """Tool to analyze signal alpha decay."""

    name: str = "compute_alpha_decay"
    description: str = "Analyze how a trading signal's predictive power decays over time. Returns optimal holding period."
    args_schema: type[BaseModel] = AlphaDecayInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        signal_column: str = "rsi_14",
        max_lag: int = 20,
        end_date: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_alpha_decay",
                symbol=symbol,
                timeframe=timeframe,
                signal_column=signal_column,
                max_lag=max_lag,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeInformationCoefficientTool(BaseTool):
    """Tool to compute Information Coefficient."""

    name: str = "compute_information_coefficient"
    description: str = (
        "Compute IC between a signal and forward returns. IC > 0.05 is generally meaningful."
    )
    args_schema: type[BaseModel] = ICInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        signal_column: str = "rsi_14",
        forward_return_periods: int = 5,
        end_date: str | None = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_information_coefficient",
                symbol=symbol,
                timeframe=timeframe,
                signal_column=signal_column,
                forward_return_periods=forward_return_periods,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ValidateSignalTool(BaseTool):
    """Tool to validate a trading signal."""

    name: str = "validate_signal"
    description: str = (
        "Validate a trading signal for statistical significance and predictive power."
    )
    args_schema: type[BaseModel] = ValidateSignalInput

    def _run(self, symbol: str, signal_column: str, timeframe: str = "daily") -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "validate_signal",
                symbol=symbol,
                signal_column=signal_column,
                timeframe=timeframe,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class DiagnoseSignalTool(BaseTool):
    """Tool to diagnose signal issues."""

    name: str = "diagnose_signal"
    description: str = "Diagnose potential issues with a trading signal (noise, lag, correlation)."
    args_schema: type[BaseModel] = DiagnoseSignalInput

    def _run(self, symbol: str, signal_column: str, timeframe: str = "daily") -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "diagnose_signal",
                symbol=symbol,
                signal_column=signal_column,
                timeframe=timeframe,
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE OPTIONS TOOL CLASSES
# =============================================================================


class PriceOptionTool(BaseTool):
    """Tool to price options using Black-Scholes."""

    name: str = "price_option"
    description: str = (
        "Calculate option price using Black-Scholes-Merton model. Returns price and Greeks."
    )
    args_schema: type[BaseModel] = PriceOptionInput

    def _run(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
        dividend_yield: float = 0,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "price_option",
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
                dividend_yield=dividend_yield,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeGreeksTool(BaseTool):
    """Tool to compute option Greeks."""

    name: str = "compute_greeks"
    description: str = (
        "Compute option Greeks (Delta, Gamma, Theta, Vega, Rho) with interpretations."
    )
    args_schema: type[BaseModel] = ComputeGreeksInput

    def _run(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
        dividend_yield: float = 0,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_greeks",
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
                dividend_yield=dividend_yield,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeImpliedVolTool(BaseTool):
    """Tool to compute implied volatility."""

    name: str = "compute_implied_vol"
    description: str = (
        "Calculate implied volatility from market option price using Newton-Raphson method."
    )
    args_schema: type[BaseModel] = ImpliedVolInput

    def _run(
        self,
        option_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_implied_vol",
                option_price=option_price,
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class AnalyzeOptionStructureTool(BaseTool):
    """Tool to analyze option structures."""

    name: str = "analyze_option_structure"
    description: str = (
        "Analyze an options structure (spread, condor, etc.) for P&L, breakevens, and Greeks."
    )
    args_schema: type[BaseModel] = AnalyzeOptionStructureInput

    def _run(
        self,
        structure_type: str,
        legs: str,
        spot: float,
        volatility: float = 0.25,
        time_to_expiry: float = 0.083,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            legs_list = json.loads(legs)
            return await bridge.call_quantcore(
                "analyze_option_structure",
                structure_type=structure_type,
                legs=legs_list,
                spot=spot,
                volatility=volatility,
                time_to_expiry=time_to_expiry,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeOptionChainTool(BaseTool):
    """Tool to compute theoretical option chain."""

    name: str = "compute_option_chain"
    description: str = (
        "Compute a theoretical option chain with prices and Greeks for multiple strikes."
    )
    args_schema: type[BaseModel] = ComputeOptionChainInput

    def _run(
        self,
        symbol: str,
        spot_price: float,
        volatility: float = 0.25,
        days_to_expiry: int = 30,
        num_strikes: int = 10,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_option_chain",
                symbol=symbol,
                spot_price=spot_price,
                volatility=volatility,
                days_to_expiry=days_to_expiry,
                num_strikes=num_strikes,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeMultiLegPriceTool(BaseTool):
    """Tool to compute multi-leg option structure price."""

    name: str = "compute_multi_leg_price"
    description: str = (
        "Calculate net debit/credit and combined Greeks for a multi-leg options trade."
    )
    args_schema: type[BaseModel] = MultiLegPriceInput

    def _run(self, legs: str, spot: float) -> str:
        async def _exec():
            bridge = get_bridge()
            legs_list = json.loads(legs)
            return await bridge.call_quantcore("compute_multi_leg_price", legs=legs_list, spot=spot)

        return json.dumps(_run_async(_exec()), indent=2)
