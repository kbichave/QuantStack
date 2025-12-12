"""
Alpha Decay Analysis.

Analyzes the decay of predictive signals over time:
- Information Coefficient (IC) decay curves
- Signal half-life estimation
- Turnover analysis
- Alpha capacity estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy import stats
from loguru import logger


@dataclass
class AlphaDecayResult:
    """Result of alpha decay analysis."""

    half_life: float  # Periods until IC drops to 50%
    decay_rate: float  # Exponential decay constant
    ic_by_lag: Dict[int, float]  # IC at each lag
    optimal_holding_period: int  # Lag with max cumulative IC
    turnover: float  # Average daily turnover
    capacity_estimate: float  # Estimated alpha capacity in dollars


class AlphaDecayAnalyzer:
    """
    Analyzes how trading signals decay over time.

    Key metrics:
    - IC decay curve: How correlation with future returns decays
    - Half-life: Time until signal loses half its predictive power
    - Optimal holding period: Lag that maximizes risk-adjusted return
    """

    def __init__(self, max_lag: int = 20):
        """
        Initialize analyzer.

        Args:
            max_lag: Maximum forward lag to analyze
        """
        self.max_lag = max_lag
        self.results: Optional[AlphaDecayResult] = None

    def analyze(
        self,
        signal: pd.Series,
        returns: pd.Series,
        volume: Optional[pd.Series] = None,
        price: Optional[pd.Series] = None,
    ) -> AlphaDecayResult:
        """
        Run full alpha decay analysis.

        Args:
            signal: Predictive signal series (z-scores or similar)
            returns: Forward return series
            volume: Trading volume (for capacity analysis)
            price: Price series (for capacity analysis)

        Returns:
            AlphaDecayResult with all metrics
        """
        # 1. Compute IC at each lag
        ic_by_lag = self._compute_ic_curve(signal, returns)

        # 2. Fit exponential decay and estimate half-life
        half_life, decay_rate = self._fit_decay_curve(ic_by_lag)

        # 3. Find optimal holding period
        optimal_lag = self._find_optimal_holding_period(ic_by_lag)

        # 4. Compute turnover
        turnover = self._compute_turnover(signal)

        # 5. Estimate capacity
        capacity = (
            self._estimate_capacity(signal, volume, price)
            if volume is not None
            else np.nan
        )

        self.results = AlphaDecayResult(
            half_life=half_life,
            decay_rate=decay_rate,
            ic_by_lag=ic_by_lag,
            optimal_holding_period=optimal_lag,
            turnover=turnover,
            capacity_estimate=capacity,
        )

        return self.results

    def _compute_ic_curve(
        self,
        signal: pd.Series,
        returns: pd.Series,
    ) -> Dict[int, float]:
        """Compute Information Coefficient at each forward lag."""
        ic_by_lag = {}

        common_idx = signal.index.intersection(returns.index)
        signal = signal.loc[common_idx]
        returns = returns.loc[common_idx]

        for lag in range(1, self.max_lag + 1):
            lagged_returns = returns.shift(-lag)
            valid = ~(signal.isna() | lagged_returns.isna())

            if valid.sum() < 30:
                ic_by_lag[lag] = np.nan
                continue

            ic, _ = stats.spearmanr(signal[valid], lagged_returns[valid])
            ic_by_lag[lag] = ic

        return ic_by_lag

    def _fit_decay_curve(
        self,
        ic_by_lag: Dict[int, float],
    ) -> Tuple[float, float]:
        """
        Fit exponential decay curve to IC values.

        IC(t) = IC(0) * exp(-lambda * t)
        Half-life = ln(2) / lambda
        """
        lags = np.array([k for k, v in ic_by_lag.items() if not np.isnan(v)])
        ics = np.array([ic_by_lag[k] for k in lags])

        if len(lags) < 3:
            return np.nan, np.nan

        # Only fit if IC starts positive (typical for predictive signals)
        if ics[0] <= 0:
            # Flip sign for decay fitting
            ics = -ics

        def exp_decay(t, ic0, lam):
            return ic0 * np.exp(-lam * t)

        try:
            # Initial guess
            p0 = [ics[0], 0.1]
            popt, _ = curve_fit(exp_decay, lags, ics, p0=p0, maxfev=5000)
            ic0, decay_rate = popt

            # Half-life
            half_life = np.log(2) / decay_rate if decay_rate > 0 else np.inf

            return half_life, decay_rate

        except Exception as e:
            logger.warning(f"Decay curve fitting failed: {e}")
            return np.nan, np.nan

    def _find_optimal_holding_period(
        self,
        ic_by_lag: Dict[int, float],
    ) -> int:
        """
        Find holding period that maximizes cumulative IC.

        Accounts for the fact that longer holding = more return capture
        but also more signal decay.
        """
        cumulative_ic = {}
        running_sum = 0

        for lag in sorted(ic_by_lag.keys()):
            ic = ic_by_lag[lag]
            if np.isnan(ic):
                continue
            running_sum += ic
            # Adjust for holding period (diminishing marginal IC)
            cumulative_ic[lag] = running_sum / np.sqrt(lag)

        if not cumulative_ic:
            return 1

        return max(cumulative_ic, key=cumulative_ic.get)

    def _compute_turnover(self, signal: pd.Series) -> float:
        """
        Compute average daily turnover from signal changes.

        Turnover = mean(|signal_t - signal_{t-1}|) / mean(|signal|)
        """
        signal = signal.dropna()

        if len(signal) < 2:
            return np.nan

        signal_change = signal.diff().abs()
        avg_change = signal_change.mean()
        avg_signal = signal.abs().mean()

        if avg_signal == 0:
            return np.nan

        return avg_change / avg_signal

    def _estimate_capacity(
        self,
        signal: pd.Series,
        volume: pd.Series,
        price: pd.Series,
    ) -> float:
        """
        Estimate alpha capacity using volume-based heuristics.

        Capacity ≈ Average participation rate * ADV * average holding

        Rule of thumb: Can trade ~1-5% of ADV without significant impact
        """
        # Align series
        common_idx = signal.index.intersection(volume.index).intersection(price.index)

        if len(common_idx) < 20:
            return np.nan

        volume = volume.loc[common_idx]
        price = price.loc[common_idx]

        # Average Daily Volume in dollars
        adv_dollars = (volume * price).mean()

        # Conservative participation rate (1%)
        participation = 0.01

        # Capacity per day
        capacity_per_day = adv_dollars * participation

        # Annualized capacity (assuming 252 trading days)
        annual_capacity = capacity_per_day * 252

        return annual_capacity

    def plot_decay_curve(
        self,
        ax=None,
        title: str = "Alpha Decay Curve",
    ):
        """
        Plot IC decay curve with fitted exponential.

        Args:
            ax: Matplotlib axis (creates new figure if None)
            title: Plot title
        """
        import matplotlib.pyplot as plt

        if self.results is None:
            raise ValueError("Run analyze() first")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ic_by_lag = self.results.ic_by_lag
        lags = sorted(ic_by_lag.keys())
        ics = [ic_by_lag[k] for k in lags]

        # Plot actual ICs
        ax.bar(lags, ics, alpha=0.7, label="Actual IC")

        # Plot fitted decay curve
        if not np.isnan(self.results.decay_rate):
            ic0 = ics[0]
            fitted_lags = np.linspace(1, max(lags), 100)
            fitted_ics = ic0 * np.exp(-self.results.decay_rate * fitted_lags)
            ax.plot(
                fitted_lags,
                fitted_ics,
                "r-",
                linewidth=2,
                label=f"Fitted decay (t½={self.results.half_life:.1f})",
            )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Forward Lag (periods)")
        ax.set_ylabel("Information Coefficient (IC)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def generate_report(self) -> str:
        """Generate text report of alpha decay analysis."""
        if self.results is None:
            return "No results. Run analyze() first."

        r = self.results

        report = f"""
Alpha Decay Analysis Report
===========================

IC Decay Metrics:
  - Half-life: {r.half_life:.2f} periods
  - Decay rate (lambda): {r.decay_rate:.4f}
  - Optimal holding period: {r.optimal_holding_period} periods

IC by Lag:
"""
        for lag, ic in sorted(r.ic_by_lag.items()):
            report += f"  Lag {lag:2d}: IC = {ic:+.4f}\n"

        report += f"""
Turnover:
  - Average daily turnover: {r.turnover:.2%}

Capacity:
  - Estimated annual capacity: ${r.capacity_estimate:,.0f}

Interpretation:
  - Signal decays to 50% strength after {r.half_life:.1f} periods
  - Recommended rebalancing frequency: every {r.optimal_holding_period} periods
  - {"High turnover - watch transaction costs" if r.turnover > 0.3 else "Moderate turnover"}
"""
        return report


def compute_signal_autocorrelation(
    signal: pd.Series,
    max_lag: int = 20,
) -> Dict[int, float]:
    """
    Compute signal autocorrelation at various lags.

    High autocorrelation = slow-moving signal = lower turnover
    """
    autocorr = {}
    signal = signal.dropna()

    for lag in range(1, max_lag + 1):
        if len(signal) > lag:
            autocorr[lag] = signal.autocorr(lag=lag)
        else:
            autocorr[lag] = np.nan

    return autocorr


def estimate_trading_frequency(
    half_life: float,
    turnover: float,
    target_tc_drag: float = 0.01,  # 1% annual TC drag
    round_trip_cost: float = 0.001,  # 10 bps round trip
) -> Dict[str, float]:
    """
    Estimate optimal trading frequency given alpha decay and costs.

    Balances:
    - More frequent trading captures more alpha
    - More frequent trading incurs more costs

    Args:
        half_life: Signal half-life in periods
        turnover: Average turnover per period
        target_tc_drag: Maximum acceptable TC drag (annual)
        round_trip_cost: Cost per round trip trade

    Returns:
        Dictionary with optimal frequency and metrics
    """
    # Annual trading cost at current turnover
    periods_per_year = 252
    annual_turnover = turnover * periods_per_year
    annual_tc = annual_turnover * round_trip_cost

    # Maximum turnover for target TC drag
    max_turnover = target_tc_drag / round_trip_cost / periods_per_year

    # Optimal rebalancing frequency (based on half-life)
    # Trade more frequently than half-life but not too frequently
    optimal_frequency = min(half_life / 2, 1 / max_turnover)

    return {
        "annual_turnover": annual_turnover,
        "annual_tc_cost": annual_tc,
        "max_turnover_for_target": max_turnover,
        "optimal_rebalance_periods": optimal_frequency,
        "optimal_trades_per_year": periods_per_year / optimal_frequency,
    }
