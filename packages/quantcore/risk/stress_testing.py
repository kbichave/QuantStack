"""
Monte Carlo Stress Testing and VaR Calculation for Options Portfolios.

Provides:
- GBM price path simulation
- Portfolio stress testing across scenarios
- Historical, Parametric, and Monte Carlo VaR
- Expected Shortfall (CVaR)
- Predefined historical stress scenarios
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger

from quantcore.options.models import OptionsPosition, OptionType
from quantcore.options.pricing import black_scholes_price, black_scholes_greeks


# Predefined historical stress scenarios
# Format: (price_change_pct, vol_change_pct)
STRESS_SCENARIOS = {
    "2008_lehman": (-0.40, +0.80),  # -40% price, +80% vol
    "2020_covid_crash": (-0.35, +1.00),  # -35% price, +100% vol
    "2010_flash_crash": (-0.10, +0.50),  # -10% price, +50% vol
    "2015_china_fears": (-0.12, +0.40),  # -12% price, +40% vol
    "2018_volmageddon": (-0.05, +1.50),  # -5% price, +150% vol (VIX spike)
    "2022_rate_shock": (-0.25, +0.30),  # -25% price, +30% vol
    "vol_spike_only": (0.0, +0.50),  # No price change, +50% vol
    "slow_decline": (-0.20, +0.20),  # -20% price, +20% vol
    "rally": (+0.15, -0.20),  # +15% price, -20% vol
    "sideways_crush": (0.0, -0.30),  # No movement, -30% vol
}


@dataclass
class StressResult:
    """Result of stress test on a single position."""

    scenario_name: str
    initial_value: float
    stressed_value: float
    pnl: float
    pnl_pct: float
    new_delta: float
    new_gamma: float
    new_vega: float
    new_theta: float


@dataclass
class PortfolioStressResult:
    """Aggregate stress test results."""

    scenario_name: str
    total_initial_value: float
    total_stressed_value: float
    total_pnl: float
    total_pnl_pct: float
    position_results: List[StressResult]
    worst_position: str
    best_position: str


@dataclass
class VaRResult:
    """Value at Risk result."""

    var_95: float
    var_99: float
    cvar_95: float  # Expected Shortfall
    cvar_99: float
    max_loss: float
    mean_loss: float
    loss_std: float
    simulation_count: int


class MonteCarloSimulator:
    """
    Monte Carlo simulator for price paths using Geometric Brownian Motion.

    The GBM model:
        dS = μSdt + σSdW

    Which gives:
        S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of price paths to simulate
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_paths(
        self,
        S0: float,
        volatility: float,
        drift: float = 0.0,
        time_horizon_days: int = 21,
        steps_per_day: int = 1,
    ) -> np.ndarray:
        """
        Generate GBM price paths.

        Args:
            S0: Initial price
            volatility: Annualized volatility
            drift: Annualized drift (default 0 for risk-neutral)
            time_horizon_days: Simulation horizon in days
            steps_per_day: Time steps per day

        Returns:
            Array of shape (n_simulations, n_steps + 1) with price paths
        """
        T = time_horizon_days / 252  # Convert to years
        n_steps = time_horizon_days * steps_per_day
        dt = T / n_steps

        # Generate random increments
        dW = np.random.randn(self.n_simulations, n_steps) * np.sqrt(dt)

        # GBM formula
        drift_term = (drift - 0.5 * volatility**2) * dt
        vol_term = volatility * dW

        # Cumulative sum for log prices
        log_returns = drift_term + vol_term
        log_prices = np.cumsum(log_returns, axis=1)

        # Add initial price
        log_prices = np.column_stack(
            [
                np.zeros(self.n_simulations),
                log_prices,
            ]
        )

        # Convert to prices
        prices = S0 * np.exp(log_prices)

        return prices

    def simulate_terminal_prices(
        self,
        S0: float,
        volatility: float,
        drift: float = 0.0,
        time_horizon_days: int = 21,
    ) -> np.ndarray:
        """
        Generate terminal prices only (faster for VaR).

        Args:
            S0: Initial price
            volatility: Annualized volatility
            drift: Annualized drift
            time_horizon_days: Horizon in days

        Returns:
            Array of terminal prices (n_simulations,)
        """
        T = time_horizon_days / 252

        # Direct GBM terminal value
        Z = np.random.randn(self.n_simulations)
        terminal_prices = S0 * np.exp(
            (drift - 0.5 * volatility**2) * T + volatility * np.sqrt(T) * Z
        )

        return terminal_prices

    def simulate_correlated(
        self,
        S0: Dict[str, float],
        volatilities: Dict[str, float],
        correlation_matrix: np.ndarray,
        time_horizon_days: int = 21,
    ) -> Dict[str, np.ndarray]:
        """
        Generate correlated price paths for multiple assets.

        Args:
            S0: Dict of symbol -> initial price
            volatilities: Dict of symbol -> volatility
            correlation_matrix: Correlation matrix
            time_horizon_days: Horizon in days

        Returns:
            Dict of symbol -> terminal prices array
        """
        symbols = list(S0.keys())
        n_assets = len(symbols)
        T = time_horizon_days / 252

        # Cholesky decomposition for correlated normals
        L = np.linalg.cholesky(correlation_matrix)

        # Generate independent normals
        Z = np.random.randn(self.n_simulations, n_assets)

        # Apply correlation
        correlated_Z = Z @ L.T

        # Generate terminal prices for each asset
        result = {}
        for i, symbol in enumerate(symbols):
            s0 = S0[symbol]
            vol = volatilities[symbol]

            terminal = s0 * np.exp(
                -0.5 * vol**2 * T + vol * np.sqrt(T) * correlated_Z[:, i]
            )
            result[symbol] = terminal

        return result


class VaRCalculator:
    """
    Value at Risk calculation using multiple methods.

    Methods:
    - Historical: Uses actual return distribution
    - Parametric: Assumes Gaussian distribution
    - Monte Carlo: Uses simulated price paths
    """

    def historical_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Calculate Historical VaR from return distribution.

        Args:
            returns: Series of historical returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            horizon_days: VaR horizon in days

        Returns:
            VaR as positive number (potential loss)
        """
        if returns.empty:
            return 0.0

        # Get the quantile
        quantile = 1 - confidence
        var_1day = -returns.quantile(quantile)

        # Scale for horizon (square root of time)
        var_horizon = var_1day * np.sqrt(horizon_days)

        return var_horizon

    def parametric_var(
        self,
        portfolio_value: float,
        volatility: float,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Calculate Parametric (Gaussian) VaR.

        VaR = V * σ * Z * sqrt(T)

        Args:
            portfolio_value: Current portfolio value
            volatility: Annualized portfolio volatility
            confidence: Confidence level
            horizon_days: VaR horizon in days

        Returns:
            VaR as positive number (potential loss)
        """
        z_score = norm.ppf(confidence)
        daily_vol = volatility / np.sqrt(252)

        var = portfolio_value * daily_vol * z_score * np.sqrt(horizon_days)

        return var

    def monte_carlo_var(
        self,
        pnl_simulations: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate VaR from Monte Carlo simulations.

        Args:
            pnl_simulations: Array of simulated P&L values
            confidence: Confidence level

        Returns:
            VaR as positive number (potential loss)
        """
        if len(pnl_simulations) == 0:
            return 0.0

        quantile = 1 - confidence
        var = -np.percentile(pnl_simulations, quantile * 100)

        return var

    def expected_shortfall(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR).

        CVaR is the average loss beyond VaR - a more conservative
        measure that accounts for tail risk.

        Args:
            returns: Series of historical returns
            confidence: Confidence level

        Returns:
            Expected Shortfall as positive number
        """
        if returns.empty:
            return 0.0

        var = self.historical_var(returns, confidence)

        # Average of losses beyond VaR
        tail_losses = returns[returns <= -var]

        if tail_losses.empty:
            return var

        cvar = -tail_losses.mean()

        return cvar

    def expected_shortfall_mc(
        self,
        pnl_simulations: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Expected Shortfall from Monte Carlo simulations.

        Args:
            pnl_simulations: Array of simulated P&L values
            confidence: Confidence level

        Returns:
            Expected Shortfall as positive number
        """
        if len(pnl_simulations) == 0:
            return 0.0

        quantile = 1 - confidence
        var = -np.percentile(pnl_simulations, quantile * 100)

        # Average of losses beyond VaR
        tail_losses = pnl_simulations[pnl_simulations <= -var]

        if len(tail_losses) == 0:
            return var

        cvar = -np.mean(tail_losses)

        return cvar

    def full_var_analysis(
        self,
        returns: pd.Series,
        portfolio_value: float,
        volatility: float,
        horizon_days: int = 1,
    ) -> VaRResult:
        """
        Comprehensive VaR analysis using multiple methods.

        Args:
            returns: Historical returns
            portfolio_value: Current value
            volatility: Portfolio volatility
            horizon_days: VaR horizon

        Returns:
            VaRResult with all metrics
        """
        return VaRResult(
            var_95=self.historical_var(returns, 0.95, horizon_days),
            var_99=self.historical_var(returns, 0.99, horizon_days),
            cvar_95=self.expected_shortfall(returns, 0.95),
            cvar_99=self.expected_shortfall(returns, 0.99),
            max_loss=-returns.min() if not returns.empty else 0.0,
            mean_loss=(
                -returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0.0
            ),
            loss_std=(
                returns[returns < 0].std() if len(returns[returns < 0]) > 1 else 0.0
            ),
            simulation_count=len(returns),
        )


class PortfolioStressTester:
    """
    Stress testing for options portfolios.

    Evaluates portfolio P&L under various market scenarios
    including historical crisis events.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
    ):
        """
        Initialize stress tester.

        Args:
            risk_free_rate: Risk-free rate for option repricing
        """
        self.risk_free_rate = risk_free_rate
        self.simulator = MonteCarloSimulator()
        self.var_calc = VaRCalculator()

    def stress_position(
        self,
        position: OptionsPosition,
        spot_price: float,
        current_iv: float,
        price_change_pct: float,
        vol_change_pct: float,
        time_decay_days: int = 0,
    ) -> StressResult:
        """
        Apply stress scenario to a single position.

        Args:
            position: Options position
            spot_price: Current underlying price
            current_iv: Current implied volatility
            price_change_pct: Price change as decimal (e.g., -0.20 for -20%)
            vol_change_pct: Vol change as decimal
            time_decay_days: Days of theta decay to apply

        Returns:
            StressResult with P&L and new Greeks
        """
        # Calculate initial value
        initial_value = self._price_position(position, spot_price, current_iv, 0)

        # Calculate stressed parameters
        stressed_price = spot_price * (1 + price_change_pct)
        stressed_iv = current_iv * (1 + vol_change_pct)
        stressed_iv = max(0.01, min(5.0, stressed_iv))  # Clip to reasonable range

        # Calculate stressed value
        stressed_value = self._price_position(
            position, stressed_price, stressed_iv, time_decay_days
        )

        # Calculate new Greeks
        greeks = self._position_greeks(
            position, stressed_price, stressed_iv, time_decay_days
        )

        pnl = stressed_value - initial_value
        pnl_pct = pnl / abs(initial_value) if initial_value != 0 else 0.0

        return StressResult(
            scenario_name="",
            initial_value=initial_value,
            stressed_value=stressed_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            new_delta=greeks.get("delta", 0),
            new_gamma=greeks.get("gamma", 0),
            new_vega=greeks.get("vega", 0),
            new_theta=greeks.get("theta", 0),
        )

    def _price_position(
        self,
        position: OptionsPosition,
        spot: float,
        iv: float,
        days_passed: int,
    ) -> float:
        """Calculate position value."""
        total = 0.0

        for leg in position.legs:
            contract = leg.contract
            T = max(0.001, (contract.days_to_expiry - days_passed) / 365)

            price = black_scholes_price(
                S=spot,
                K=contract.strike,
                T=T,
                r=self.risk_free_rate,
                sigma=iv,
                option_type=contract.option_type,
            )

            # Value per contract (x100 multiplier)
            leg_value = price * 100 * leg.quantity
            if not leg.is_long:
                leg_value = -leg_value

            total += leg_value

        return total

    def _position_greeks(
        self,
        position: OptionsPosition,
        spot: float,
        iv: float,
        days_passed: int,
    ) -> Dict[str, float]:
        """Calculate aggregate position Greeks."""
        delta = 0.0
        gamma = 0.0
        vega = 0.0
        theta = 0.0

        for leg in position.legs:
            contract = leg.contract
            T = max(0.001, (contract.days_to_expiry - days_passed) / 365)

            greeks = black_scholes_greeks(
                S=spot,
                K=contract.strike,
                T=T,
                r=self.risk_free_rate,
                sigma=iv,
                option_type=contract.option_type,
            )

            multiplier = leg.quantity * 100
            if not leg.is_long:
                multiplier = -multiplier

            delta += greeks.delta * multiplier
            gamma += greeks.gamma * multiplier
            vega += greeks.vega * multiplier
            theta += greeks.theta * multiplier

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
        }

    def run_scenario(
        self,
        positions: Dict[str, OptionsPosition],
        spot_prices: Dict[str, float],
        ivs: Dict[str, float],
        scenario_name: str,
        price_change_pct: float,
        vol_change_pct: float,
        time_decay_days: int = 0,
    ) -> PortfolioStressResult:
        """
        Run a stress scenario across the portfolio.

        Args:
            positions: Dict of symbol -> OptionsPosition
            spot_prices: Dict of symbol -> current price
            ivs: Dict of symbol -> current IV
            scenario_name: Name of the scenario
            price_change_pct: Price change to apply
            vol_change_pct: Vol change to apply
            time_decay_days: Theta decay days

        Returns:
            PortfolioStressResult with aggregate metrics
        """
        position_results = []
        total_initial = 0.0
        total_stressed = 0.0

        for symbol, position in positions.items():
            spot = spot_prices.get(symbol, 100.0)
            iv = ivs.get(symbol, 0.30)

            result = self.stress_position(
                position=position,
                spot_price=spot,
                current_iv=iv,
                price_change_pct=price_change_pct,
                vol_change_pct=vol_change_pct,
                time_decay_days=time_decay_days,
            )

            result.scenario_name = scenario_name
            position_results.append(result)

            total_initial += result.initial_value
            total_stressed += result.stressed_value

        total_pnl = total_stressed - total_initial
        total_pnl_pct = total_pnl / abs(total_initial) if total_initial != 0 else 0.0

        # Find worst and best positions
        if position_results:
            worst_idx = min(
                range(len(position_results)), key=lambda i: position_results[i].pnl
            )
            best_idx = max(
                range(len(position_results)), key=lambda i: position_results[i].pnl
            )
            worst_symbol = list(positions.keys())[worst_idx]
            best_symbol = list(positions.keys())[best_idx]
        else:
            worst_symbol = ""
            best_symbol = ""

        return PortfolioStressResult(
            scenario_name=scenario_name,
            total_initial_value=total_initial,
            total_stressed_value=total_stressed,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            position_results=position_results,
            worst_position=worst_symbol,
            best_position=best_symbol,
        )

    def run_all_scenarios(
        self,
        positions: Dict[str, OptionsPosition],
        spot_prices: Dict[str, float],
        ivs: Dict[str, float],
        scenarios: Optional[Dict[str, Tuple[float, float]]] = None,
        time_decay_days: int = 0,
    ) -> pd.DataFrame:
        """
        Run all stress scenarios and summarize results.

        Args:
            positions: Portfolio positions
            spot_prices: Current spot prices
            ivs: Current implied volatilities
            scenarios: Custom scenarios (uses STRESS_SCENARIOS if None)
            time_decay_days: Days of theta decay

        Returns:
            DataFrame with scenario results
        """
        if scenarios is None:
            scenarios = STRESS_SCENARIOS

        results = []

        for name, (price_change, vol_change) in scenarios.items():
            result = self.run_scenario(
                positions=positions,
                spot_prices=spot_prices,
                ivs=ivs,
                scenario_name=name,
                price_change_pct=price_change,
                vol_change_pct=vol_change,
                time_decay_days=time_decay_days,
            )

            results.append(
                {
                    "scenario": name,
                    "price_change": f"{price_change:+.1%}",
                    "vol_change": f"{vol_change:+.1%}",
                    "initial_value": result.total_initial_value,
                    "stressed_value": result.total_stressed_value,
                    "pnl": result.total_pnl,
                    "pnl_pct": result.total_pnl_pct,
                    "worst_position": result.worst_position,
                }
            )

        return pd.DataFrame(results).sort_values("pnl")

    def monte_carlo_stress(
        self,
        positions: Dict[str, OptionsPosition],
        spot_prices: Dict[str, float],
        ivs: Dict[str, float],
        n_simulations: int = 10000,
        horizon_days: int = 21,
        include_vol_shock: bool = True,
    ) -> VaRResult:
        """
        Run Monte Carlo stress test for portfolio VaR.

        Args:
            positions: Portfolio positions
            spot_prices: Current prices
            ivs: Current IVs
            n_simulations: Number of simulations
            horizon_days: Time horizon
            include_vol_shock: Whether to simulate vol changes

        Returns:
            VaRResult with risk metrics
        """
        self.simulator.n_simulations = n_simulations

        # Simulate terminal prices for each underlying
        pnl_simulations = np.zeros(n_simulations)

        for symbol, position in positions.items():
            spot = spot_prices.get(symbol, 100.0)
            iv = ivs.get(symbol, 0.30)

            # Simulate prices
            terminal_prices = self.simulator.simulate_terminal_prices(
                S0=spot,
                volatility=iv,
                drift=0.0,
                time_horizon_days=horizon_days,
            )

            # Simulate vol changes if requested
            if include_vol_shock:
                # Vol tends to increase when price drops (negative correlation)
                returns = terminal_prices / spot - 1
                vol_shocks = -0.5 * returns  # Rough leverage effect
                vol_shocks = np.clip(vol_shocks, -0.5, 1.0)
                simulated_ivs = iv * (1 + vol_shocks)
            else:
                simulated_ivs = np.full(n_simulations, iv)

            # Calculate initial value
            initial_value = self._price_position(position, spot, iv, 0)

            # Calculate terminal values for each simulation
            for i in range(n_simulations):
                terminal_value = self._price_position(
                    position,
                    terminal_prices[i],
                    simulated_ivs[i],
                    horizon_days,
                )
                pnl_simulations[i] += terminal_value - initial_value

        # Calculate VaR metrics
        return VaRResult(
            var_95=self.var_calc.monte_carlo_var(pnl_simulations, 0.95),
            var_99=self.var_calc.monte_carlo_var(pnl_simulations, 0.99),
            cvar_95=self.var_calc.expected_shortfall_mc(pnl_simulations, 0.95),
            cvar_99=self.var_calc.expected_shortfall_mc(pnl_simulations, 0.99),
            max_loss=-np.min(pnl_simulations),
            mean_loss=(
                -np.mean(pnl_simulations[pnl_simulations < 0])
                if np.any(pnl_simulations < 0)
                else 0
            ),
            loss_std=(
                np.std(pnl_simulations[pnl_simulations < 0])
                if np.any(pnl_simulations < 0)
                else 0
            ),
            simulation_count=n_simulations,
        )
