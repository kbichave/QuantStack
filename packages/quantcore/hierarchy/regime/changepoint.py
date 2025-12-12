"""
Bayesian changepoint detection for regime shifts.

Implements online Bayesian changepoint detection algorithm.
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class ChangePoint:
    """Detected changepoint."""

    timestamp: pd.Timestamp
    index: int
    probability: float
    run_length: int
    magnitude: float  # Magnitude of change


@dataclass
class ChangepointResult:
    """Result from changepoint detection."""

    changepoints: List[ChangePoint]
    current_run_length: int
    run_length_probabilities: np.ndarray
    regime_change_probability: float
    time_since_last_change: int


class BayesianChangepointDetector:
    """
    Bayesian online changepoint detection.

    Implements the algorithm from Adams & MacKay (2007):
    "Bayesian Online Changepoint Detection"

    Uses a predictive distribution to detect regime shifts in:
    - Price returns
    - Volatility
    - Correlation structure

    Key features:
    - Online detection (processes data sequentially)
    - Probabilistic output (changepoint probability)
    - Adjustable sensitivity (hazard rate)
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 250,  # Expected 1 regime change per year
        mean_prior_mu: float = 0.0,
        mean_prior_var: float = 1.0,
        obs_var: float = 0.01,
        threshold: float = 0.5,
    ):
        """
        Initialize Bayesian changepoint detector.

        Args:
            hazard_rate: Probability of changepoint at each time step
            mean_prior_mu: Prior mean for Gaussian conjugate prior
            mean_prior_var: Prior variance for Gaussian conjugate prior
            obs_var: Observation variance
            threshold: Probability threshold for declaring changepoint
        """
        self.hazard_rate = hazard_rate
        self.mean_prior_mu = mean_prior_mu
        self.mean_prior_var = mean_prior_var
        self.obs_var = obs_var
        self.threshold = threshold

        # State variables (reset on fit)
        self.run_length_probs: Optional[np.ndarray] = None
        self.posterior_means: Optional[np.ndarray] = None
        self.posterior_vars: Optional[np.ndarray] = None
        self.changepoints: List[ChangePoint] = []

    def detect(self, df: pd.DataFrame, feature: str = "returns") -> ChangepointResult:
        """
        Detect changepoints in data.

        Args:
            df: DataFrame with OHLCV data
            feature: Feature to analyze ("returns", "volatility", "price")

        Returns:
            ChangepointResult with detected changepoints
        """
        # Prepare data
        if feature == "returns":
            data = df["close"].pct_change().dropna().values
        elif feature == "volatility":
            data = df["close"].pct_change().rolling(10).std().dropna().values
        elif feature == "price":
            data = df["close"].values
        else:
            data = df["close"].pct_change().dropna().values

        if len(data) < 10:
            return self._empty_result()

        # Run online changepoint detection
        changepoints, run_length_history = self._run_detection(data)

        # Map changepoints back to timestamps
        return_idx_offset = 1 if feature == "returns" else 0
        vol_idx_offset = 10 if feature == "volatility" else return_idx_offset

        mapped_changepoints = []
        for cp in changepoints:
            adj_idx = cp.index + vol_idx_offset
            if adj_idx < len(df):
                mapped_changepoints.append(
                    ChangePoint(
                        timestamp=df.index[adj_idx],
                        index=adj_idx,
                        probability=cp.probability,
                        run_length=cp.run_length,
                        magnitude=cp.magnitude,
                    )
                )

        # Current state
        current_run_length = (
            int(np.argmax(run_length_history[-1])) if len(run_length_history) > 0 else 0
        )
        regime_change_prob = (
            float(run_length_history[-1][0]) if len(run_length_history) > 0 else 0.0
        )

        # Time since last changepoint
        if len(mapped_changepoints) > 0:
            time_since_last = len(df) - mapped_changepoints[-1].index
        else:
            time_since_last = len(df)

        return ChangepointResult(
            changepoints=mapped_changepoints,
            current_run_length=current_run_length,
            run_length_probabilities=(
                run_length_history[-1]
                if len(run_length_history) > 0
                else np.array([1.0])
            ),
            regime_change_probability=regime_change_prob,
            time_since_last_change=time_since_last,
        )

    def _run_detection(
        self,
        data: np.ndarray,
    ) -> Tuple[List[ChangePoint], List[np.ndarray]]:
        """Run online Bayesian changepoint detection."""
        T = len(data)

        # Initialize run length distribution
        # P(r_t = r | x_{1:t})
        run_length_probs = [np.array([1.0])]  # Start with run length 0

        # Posterior parameters for each run length
        # Using Gaussian conjugate prior: N(mu | mu_0, sigma_0^2)
        posterior_means = [np.array([self.mean_prior_mu])]
        posterior_vars = [np.array([self.mean_prior_var])]

        changepoints = []
        run_length_history = []

        for t in range(T):
            x_t = data[t]

            # Current number of run lengths
            n_rl = len(run_length_probs[-1])

            # Predictive probability: P(x_t | r_{t-1}, x_{1:t-1})
            # For Gaussian: predictive is N(mu_r, sigma_r^2 + obs_var)
            pred_means = posterior_means[-1]
            pred_vars = posterior_vars[-1] + self.obs_var

            # Gaussian likelihood
            pred_probs = self._gaussian_pdf(x_t, pred_means, pred_vars)

            # Growth probability: P(r_t = r_{t-1} + 1 | x_{1:t})
            growth_probs = run_length_probs[-1] * pred_probs * (1 - self.hazard_rate)

            # Changepoint probability: P(r_t = 0 | x_{1:t})
            cp_prob = np.sum(run_length_probs[-1] * pred_probs * self.hazard_rate)

            # New run length distribution
            new_rl_probs = np.zeros(n_rl + 1)
            new_rl_probs[0] = cp_prob
            new_rl_probs[1:] = growth_probs

            # Normalize
            new_rl_probs = new_rl_probs / (np.sum(new_rl_probs) + 1e-10)

            # Update posterior parameters
            # Bayesian update for Gaussian conjugate prior
            new_means = np.zeros(n_rl + 1)
            new_vars = np.zeros(n_rl + 1)

            # For run length 0 (new segment): use prior
            new_means[0] = self.mean_prior_mu
            new_vars[0] = self.mean_prior_var

            # For other run lengths: update posterior
            for r in range(n_rl):
                old_var = posterior_vars[-1][r]
                # Posterior precision = prior precision + observation precision
                new_precision = 1.0 / old_var + 1.0 / self.obs_var
                new_vars[r + 1] = 1.0 / new_precision
                # Posterior mean = weighted combination
                new_means[r + 1] = new_vars[r + 1] * (
                    posterior_means[-1][r] / old_var + x_t / self.obs_var
                )

            run_length_probs.append(new_rl_probs)
            posterior_means.append(new_means)
            posterior_vars.append(new_vars)
            run_length_history.append(new_rl_probs)

            # Check for changepoint
            if new_rl_probs[0] > self.threshold:
                # Calculate magnitude (change in mean)
                if len(data) > t + 10:
                    after_mean = np.mean(data[t : min(t + 10, len(data))])
                else:
                    after_mean = x_t

                if t >= 10:
                    before_mean = np.mean(data[max(0, t - 10) : t])
                else:
                    before_mean = x_t

                magnitude = abs(after_mean - before_mean)

                changepoints.append(
                    ChangePoint(
                        timestamp=None,  # Will be mapped later
                        index=t,
                        probability=float(new_rl_probs[0]),
                        run_length=0,
                        magnitude=magnitude,
                    )
                )

        return changepoints, run_length_history

    def _gaussian_pdf(
        self,
        x: float,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> np.ndarray:
        """Compute Gaussian PDF values."""
        return np.exp(-0.5 * (x - means) ** 2 / variances) / np.sqrt(
            2 * np.pi * variances
        )

    def _empty_result(self) -> ChangepointResult:
        """Return empty result."""
        return ChangepointResult(
            changepoints=[],
            current_run_length=0,
            run_length_probabilities=np.array([1.0]),
            regime_change_probability=0.0,
            time_since_last_change=0,
        )

    def get_changepoint_probability_series(
        self,
        df: pd.DataFrame,
        feature: str = "returns",
    ) -> pd.Series:
        """
        Get changepoint probability for each timestamp.

        Args:
            df: DataFrame with OHLCV data
            feature: Feature to analyze

        Returns:
            Series of changepoint probabilities
        """
        # Prepare data
        if feature == "returns":
            data = df["close"].pct_change().dropna().values
            start_idx = 1
        elif feature == "volatility":
            data = df["close"].pct_change().rolling(10).std().dropna().values
            start_idx = 10
        else:
            data = df["close"].pct_change().dropna().values
            start_idx = 1

        if len(data) < 10:
            return pd.Series(0.0, index=df.index, name="changepoint_prob")

        _, run_length_history = self._run_detection(data)

        # Extract changepoint probability (P(r_t = 0))
        cp_probs = [rlh[0] for rlh in run_length_history]

        # Create full series
        full_probs = [0.0] * start_idx + cp_probs
        full_probs = full_probs[: len(df)]  # Truncate if needed

        # Pad if necessary
        while len(full_probs) < len(df):
            full_probs.append(0.0)

        return pd.Series(full_probs, index=df.index, name="changepoint_prob")
