# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Strategy Selection Environment — capital allocation across strategy pool.

The agent observes per-strategy performance metrics (rolling Sharpe, win
rate, max drawdown, regime fit, IC score) and outputs allocation weights.
Reward is the rolling 21-day portfolio Sharpe ratio, encouraging the agent
to concentrate capital on strategies with the best risk-adjusted outlook.

Designed to sit above individual strategy signals and make the meta-
decision: *how much capital does each strategy deserve right now?*
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


# Per-strategy observation features.
_STRATEGY_FEATURES = 5  # sharpe, win_rate, max_drawdown, regime_fit, ic_score


class StrategySelectEnv(gym.Env):
    """
    Strategy capital allocation environment.

    Observation space
        Box with shape ``(n_strategies * 5,)`` — per-strategy features:
        ``[rolling_sharpe, win_rate, max_drawdown, regime_fit, ic_score]``
        repeated for each strategy.

    Action space
        Box(low=0, high=1, shape=(n_strategies,)) — raw allocation weights.
        Normalized to sum to 1 before applying.

    Reward
        Rolling 21-day annualized Sharpe ratio of the meta-portfolio.

    Parameters
    ----------
    strategy_returns_df : pd.DataFrame
        Columns are strategy names, rows are daily returns (DatetimeIndex
        or sequential integer index).
    n_strategies : int
        Number of strategies (must match number of columns in
        ``strategy_returns_df``).
    window : int
        Lookback window for computing Sharpe ratio and other rolling
        statistics (default 21 trading days).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        strategy_returns_df: pd.DataFrame,
        n_strategies: int,
        window: int = 21,
    ) -> None:
        super().__init__()

        if strategy_returns_df.shape[1] != n_strategies:
            raise ValueError(
                f"strategy_returns_df has {strategy_returns_df.shape[1]} columns "
                f"but n_strategies={n_strategies}"
            )

        self.returns_df = strategy_returns_df.values.astype(np.float64)  # (T, n_strats)
        self.strategy_names = list(strategy_returns_df.columns)
        self.n_strategies = n_strategies
        self.window = window
        self._n_dates = len(self.returns_df)

        obs_dim = self.n_strategies * _STRATEGY_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_strategies,), dtype=np.float32
        )

        # Episode state
        self._step_idx: int = 0
        self._prev_weights: np.ndarray = np.ones(n_strategies, dtype=np.float64) / n_strategies
        self.portfolio_returns: list[float] = []

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Start after enough history for a full rolling window.
        self._step_idx = self.window
        self._prev_weights = np.ones(self.n_strategies, dtype=np.float64) / self.n_strategies
        self.portfolio_returns = []

        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Normalize weights.
        raw = np.asarray(action, dtype=np.float64)
        weight_sum = raw.sum()
        weights = raw / weight_sum if weight_sum > 1e-12 else np.ones(self.n_strategies) / self.n_strategies

        # Portfolio return for this step.
        strat_returns = self.returns_df[self._step_idx]  # (n_strategies,)
        port_return = float(np.dot(weights, strat_returns))
        self.portfolio_returns.append(port_return)

        # Reward: rolling annualized Sharpe.
        reward = self._rolling_sharpe()

        self._prev_weights = weights.copy()
        self._step_idx += 1

        terminated = self._step_idx >= self._n_dates - 1
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info: dict[str, Any] = {
            "portfolio_return": port_return,
            "weights": weights.tolist(),
            "rolling_sharpe": reward,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """
        Build per-strategy feature vector from rolling window ending at
        ``_step_idx``.
        """
        start = max(0, self._step_idx - self.window)
        window_rets = self.returns_df[start : self._step_idx]  # (window, n_strats)

        features: list[float] = []
        for s in range(self.n_strategies):
            col = window_rets[:, s]
            if len(col) < 2:
                features.extend([0.0, 0.5, 0.0, 0.5, 0.0])
                continue

            # Rolling Sharpe (annualized).
            mean_r = float(np.mean(col))
            std_r = float(np.std(col))
            sharpe = (mean_r / (std_r + 1e-12)) * np.sqrt(252)

            # Win rate.
            win_rate = float(np.mean(col > 0))

            # Max drawdown over window.
            cum = np.cumprod(1.0 + col)
            running_max = np.maximum.accumulate(cum)
            drawdowns = (running_max - cum) / (running_max + 1e-12)
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

            # Regime fit: autocorrelation as a proxy (positive = trending regime).
            if len(col) > 1:
                regime_fit = float(np.corrcoef(col[:-1], col[1:])[0, 1])
                if np.isnan(regime_fit):
                    regime_fit = 0.0
            else:
                regime_fit = 0.0

            # IC score: rank correlation of return vs position in window (momentum proxy).
            ranks = np.arange(len(col), dtype=np.float64)
            if np.std(col) > 1e-12:
                ic_score = float(np.corrcoef(ranks, col)[0, 1])
                if np.isnan(ic_score):
                    ic_score = 0.0
            else:
                ic_score = 0.0

            features.extend([
                np.clip(sharpe / 3.0, -2.0, 2.0),
                np.clip(win_rate, 0.0, 1.0),
                np.clip(max_dd, 0.0, 1.0),
                np.clip(regime_fit, -1.0, 1.0),
                np.clip(ic_score, -1.0, 1.0),
            ])

        return np.array(features, dtype=np.float32)

    def _rolling_sharpe(self) -> float:
        """Annualized Sharpe of the meta-portfolio over the most recent window."""
        if len(self.portfolio_returns) < 2:
            return 0.0
        recent = self.portfolio_returns[-self.window :]
        mean_r = float(np.mean(recent))
        std_r = float(np.std(recent))
        return float((mean_r / (std_r + 1e-12)) * np.sqrt(252))
