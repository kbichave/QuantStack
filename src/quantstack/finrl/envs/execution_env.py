# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Order Execution Environment — learns optimal execution of large orders.

The agent must slice a large parent order across discrete time-steps to
minimize implementation shortfall versus a naive TWAP benchmark.  At each
step, the agent chooses what fraction of the remaining quantity to execute.
Market impact is modeled as a square-root function of participation rate,
so large child orders move the price more.

This is a *simpler* companion to the full-featured ``ExecutionEnv`` in
``quantstack.finrl.environments``.  It is designed to be fast to train
(low-dimensional obs/action) and easy to integrate as a signal source.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger


class ExecutionEnv(gym.Env):
    """
    Optimal order execution environment.

    Observation (5)
        [remaining_qty_pct, time_remaining_pct, spread, volatility,
         volume_ratio]

    Action (Discrete 5)
        Execute 0 / 25 / 50 / 75 / 100 % of remaining quantity.

    Reward
        Negative implementation shortfall versus TWAP.  Lower shortfall
        (closer execution to TWAP cost) yields higher reward.

    Parameters
    ----------
    total_qty : float
        Total shares to execute during the episode.
    time_horizon_steps : int
        Number of discrete time-steps available for execution.
    base_impact : float
        Market-impact coefficient (higher = more costly fills).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    # Maps discrete action index -> fraction of remaining qty to execute.
    EXECUTION_FRACTIONS: dict[int, float] = {
        0: 0.00,
        1: 0.25,
        2: 0.50,
        3: 0.75,
        4: 1.00,
    }

    def __init__(
        self,
        total_qty: float = 1000.0,
        time_horizon_steps: int = 20,
        base_impact: float = 0.001,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.total_qty = total_qty
        self.time_horizon_steps = time_horizon_steps
        self.base_impact = base_impact

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self._rng = np.random.default_rng(seed)

        # Episode state (set on reset)
        self.remaining_qty: float = 0.0
        self.remaining_steps: int = 0
        self.arrival_price: float = 0.0
        self.current_price: float = 0.0
        self.executed_qty: float = 0.0
        self.executed_cost: float = 0.0  # total dollar cost of fills
        self.twap_cost: float = 0.0  # cumulative TWAP benchmark cost
        self.execution_history: list[dict[str, Any]] = []
        self._volatility: float = 0.02
        self._spread: float = 0.0005
        self._volume_ratio: float = 1.0

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
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.remaining_qty = self.total_qty
        self.remaining_steps = self.time_horizon_steps
        self.arrival_price = 100.0 + self._rng.normal(0, 5)
        self.current_price = self.arrival_price
        self.executed_qty = 0.0
        self.executed_cost = 0.0
        self.twap_cost = 0.0
        self.execution_history = []

        # Randomize market microstructure parameters per episode.
        self._volatility = float(np.clip(self._rng.lognormal(-4, 0.5), 0.005, 0.10))
        self._spread = float(np.clip(self._rng.lognormal(-8, 0.5), 0.0001, 0.005))
        self._volume_ratio = float(np.clip(self._rng.lognormal(0, 0.3), 0.3, 3.0))

        return self._get_observation(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        exec_frac = self.EXECUTION_FRACTIONS[int(action)]
        child_qty = self.remaining_qty * exec_frac

        # --- Market impact (square-root model) ---
        participation = child_qty / (self.total_qty + 1e-12)
        impact = self.base_impact * np.sqrt(participation) * self._volatility * 100
        fill_price = self.current_price * (1.0 + impact + self._spread)

        # --- TWAP benchmark: uniform slice ---
        twap_slice = self.total_qty / max(self.time_horizon_steps, 1)
        twap_fill = self.current_price * (
            1.0 + self.base_impact * np.sqrt(twap_slice / (self.total_qty + 1e-12)) * self._volatility * 100
            + self._spread
        )
        self.twap_cost += twap_slice * twap_fill

        # --- Update execution state ---
        if child_qty > 0:
            self.executed_qty += child_qty
            self.executed_cost += child_qty * fill_price

        self.remaining_qty -= child_qty
        self.remaining_steps -= 1

        self.execution_history.append(
            {
                "step": self.time_horizon_steps - self.remaining_steps,
                "action": int(action),
                "child_qty": child_qty,
                "fill_price": fill_price,
                "impact": impact,
            }
        )

        # --- Advance market ---
        price_change = self._rng.normal(0, self._volatility) * self.current_price
        self.current_price = max(self.current_price + price_change, 0.01)

        # --- Termination ---
        terminated = self.remaining_qty <= 1e-8 or self.remaining_steps <= 0
        truncated = False

        # Force-fill any remainder at end of horizon.
        if self.remaining_steps <= 0 and self.remaining_qty > 1e-8:
            penalty_price = self.current_price * (1.0 + self._spread * 3)
            self.executed_cost += self.remaining_qty * penalty_price
            self.executed_qty += self.remaining_qty
            self.remaining_qty = 0.0

        # --- Reward: negative implementation shortfall vs TWAP ---
        if self.executed_qty > 0 and self.twap_cost > 0:
            agent_avg = self.executed_cost / self.executed_qty
            twap_avg = self.twap_cost / min(self.executed_qty, self.total_qty)
            # Positive reward when agent beats TWAP, negative when worse.
            shortfall = (agent_avg - twap_avg) / self.arrival_price
            reward = -shortfall * 1000  # scale for learning
        else:
            reward = 0.0

        obs = self._get_observation() if not terminated else np.zeros(5, dtype=np.float32)

        info: dict[str, Any] = {
            "shortfall_vs_twap": shortfall if self.executed_qty > 0 else 0.0,
            "child_qty": child_qty,
            "fill_price": fill_price,
            "executed_pct": self.executed_qty / self.total_qty,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        remaining_pct = self.remaining_qty / max(self.total_qty, 1e-12)
        time_pct = self.remaining_steps / max(self.time_horizon_steps, 1)
        return np.array(
            [
                np.clip(remaining_pct, 0.0, 1.0),
                np.clip(time_pct, 0.0, 1.0),
                np.clip(self._spread, 0.0, 0.01),
                np.clip(self._volatility, 0.0, 0.20),
                np.clip(self._volume_ratio, 0.0, 5.0),
            ],
            dtype=np.float32,
        )
