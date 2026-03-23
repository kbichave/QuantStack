"""
Custom Gymnasium environments for domain-specific RL use cases.

These are Gymnasium-compatible envs that FinRL's DRLAgent can train on.
They cover use cases that FinRL's built-in environments don't support:
  - ExecutionEnv: order execution optimization (TWAP/VWAP-style)
  - SizingEnv: dynamic position sizing
  - AlphaSelectionEnv: alpha signal weighting / selection

For standard stock trading / portfolio allocation, use FinRL's built-in
StockTradingEnv directly via the trainer module.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class ExecutionEnv(gym.Env):
    """
    Order execution optimization environment.

    Learns to slice orders over time to minimize implementation shortfall,
    balancing urgency against market impact.

    Observation (8): qty_frac, time_frac, price_dev, spread, volatility,
                     volume_ratio, vwap_dev, shortfall
    Action (Discrete 5): WAIT, SMALL(10%), MEDIUM(25%), LARGE(50%), MARKET(100%)
    Reward: -impact + completion_bonus - time_penalty + progress
    """

    metadata = {"render_modes": []}

    EXECUTION_FRACTIONS = {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.50, 4: 1.00}

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        total_quantity: float = 1000,
        time_horizon: int = 20,
        market_impact_coef: float = 0.1,
        spread_bps: float = 5.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.data = data
        self.total_quantity = total_quantity
        self.time_horizon = time_horizon
        self.market_impact_coef = market_impact_coef
        self.spread_bps = spread_bps

        self.observation_space = spaces.Box(low=-1, high=5, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # State variables (set on reset)
        self.remaining_qty = 0.0
        self.remaining_time = 0
        self.arrival_price = 0.0
        self.current_price = 0.0
        self.executed_qty = 0.0
        self.executed_value = 0.0
        self.vwap = 0.0
        self.data_idx = 0

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.remaining_qty = self.total_quantity
        self.remaining_time = self.time_horizon
        self.executed_qty = 0.0
        self.executed_value = 0.0

        if self.data is not None and len(self.data) > self.time_horizon + 20:
            self.data_idx = self._rng.integers(20, len(self.data) - self.time_horizon - 1)
            self.arrival_price = float(self.data.iloc[self.data_idx]["close"])
        else:
            self.data_idx = 0
            self.arrival_price = 100.0

        self.current_price = self.arrival_price
        self.vwap = self.arrival_price

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        exec_frac = self.EXECUTION_FRACTIONS[action]
        exec_qty = self.remaining_qty * exec_frac

        # Market impact
        vol_frac = exec_qty / (self._get_volume() * self.total_quantity + 1e-8)
        temp_impact = self.market_impact_coef * np.sqrt(vol_frac) * self._get_volatility()
        if action == 4:  # market order
            temp_impact *= 2.0
        spread_cost = (self.spread_bps / 20_000) if action == 4 else 0.0
        fill_price = self.current_price * (1 + temp_impact + spread_cost)

        # Update execution state
        if exec_qty > 0:
            self.executed_qty += exec_qty
            self.executed_value += exec_qty * fill_price

        self.remaining_qty -= exec_qty
        self.remaining_time -= 1

        # Advance market
        self.data_idx += 1
        self._update_market()

        # Shortfall
        shortfall = (
            (self.executed_value / self.executed_qty - self.arrival_price)
            / self.arrival_price
            if self.executed_qty > 0
            else 0.0
        )

        # Termination
        done = self.remaining_qty <= 0 or self.remaining_time <= 0
        truncated = False

        # Reward
        reward = -temp_impact * 100  # impact cost
        if self.remaining_qty <= 0:
            reward += 1.0  # completion bonus
        elif self.remaining_time <= 0:
            unfilled = self.remaining_qty / self.total_quantity
            reward -= unfilled * 5.0  # time penalty
        if exec_qty > 0:
            reward += (exec_qty / self.total_quantity) * 0.1  # progress
        else:
            reward -= 0.01  # wait penalty

        info = {
            "shortfall": shortfall,
            "executed_qty": exec_qty,
            "fill_price": fill_price,
            "market_impact": temp_impact,
        }

        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self) -> np.ndarray:
        qty_frac = self.remaining_qty / max(self.total_quantity, 1e-8)
        time_frac = self.remaining_time / max(self.time_horizon, 1)
        price_dev = (self.current_price - self.arrival_price) / max(self.arrival_price, 1e-8)
        spread = self.spread_bps / 10_000
        vwap_dev = (
            (self.current_price - self.vwap) / max(self.vwap, 1e-8)
            if self.vwap > 0
            else 0.0
        )
        shortfall = (
            (self.executed_value / self.executed_qty - self.arrival_price)
            / self.arrival_price
            if self.executed_qty > 0
            else 0.0
        )
        return np.array(
            [
                np.clip(qty_frac, 0, 1),
                np.clip(time_frac, 0, 1),
                np.clip(price_dev, -0.1, 0.1),
                np.clip(spread, 0, 0.005),
                np.clip(self._get_volatility(), 0, 0.2),
                np.clip(self._get_volume(), 0, 5),
                np.clip(vwap_dev, -0.05, 0.05),
                np.clip(shortfall, -0.05, 0.05),
            ],
            dtype=np.float32,
        )

    def _get_volatility(self) -> float:
        if self.data is not None and "close" in self.data.columns and self.data_idx >= 20:
            rets = self.data["close"].pct_change().iloc[self.data_idx - 20 : self.data_idx]
            vol = rets.std()
            return float(vol) if not np.isnan(vol) else 0.02
        return 0.02

    def _get_volume(self) -> float:
        if self.data is not None and "volume" in self.data.columns and self.data_idx >= 20:
            recent = self.data["volume"].iloc[self.data_idx]
            avg = self.data["volume"].iloc[self.data_idx - 20 : self.data_idx].mean()
            return float(recent / avg) if avg > 0 else 1.0
        return 1.0

    def _update_market(self) -> None:
        if self.data is not None and self.data_idx < len(self.data):
            self.current_price = float(self.data.iloc[self.data_idx]["close"])
            if self.data_idx > 0 and all(
                c in self.data.columns for c in ["high", "low", "close", "volume"]
            ):
                recent = self.data.iloc[max(0, self.data_idx - 10) : self.data_idx + 1]
                tp = (recent["high"] + recent["low"] + recent["close"]) / 3
                vol_sum = recent["volume"].sum()
                if vol_sum > 0:
                    self.vwap = float((tp * recent["volume"]).sum() / vol_sum)
        else:
            self.current_price *= 1 + self._rng.normal(0, 0.02)


class SizingEnv(gym.Env):
    """
    Dynamic position sizing environment.

    Learns to scale position size based on signal confidence, drawdown,
    regime, and risk budget.

    Observation (10): confidence, direction, vol, dd, risk_budget, sharpe,
                      position_pct, time_since_trade, regime, win_rate
    Action (Box [0,1]): position scale factor
    Reward: risk_adjusted_return + drawdown_penalty + consistency + sizing_bonus
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        initial_equity: float = 100_000,
        max_position_pct: float = 0.2,
        max_drawdown_limit: float = 0.15,
        max_steps: int = 250,
        seed: int | None = None,
    ):
        super().__init__()
        self.data = data
        self.initial_equity = initial_equity
        self.max_position_pct = max_position_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.max_steps = max_steps

        self.observation_space = spaces.Box(low=-5, high=5, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # Episode state
        self.equity = initial_equity
        self.position = 0.0
        self.equity_curve: list[float] = []
        self.returns: list[float] = []
        self.positions: list[float] = []
        self.step_count = 0
        self.data_idx = 0

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.equity = self.initial_equity
        self.position = 0.0
        self.equity_curve = [self.initial_equity]
        self.returns = []
        self.positions = []
        self.step_count = 0

        if self.data is not None and len(self.data) > self.max_steps + 50:
            self.data_idx = self._rng.integers(50, len(self.data) - self.max_steps - 1)
        else:
            self.data_idx = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        scale = float(np.clip(action[0], 0, 1))

        # Synthetic signal
        signal_dir = self._rng.choice([1, -1, 0], p=[0.4, 0.4, 0.2])
        confidence = float(self._rng.beta(2, 2))

        max_pos = self.equity * self.max_position_pct
        target = signal_dir * scale * max_pos

        # Market return
        ret = self._get_return()
        pnl = self.position * ret
        self.equity += pnl
        self.position = target

        self.equity_curve.append(self.equity)
        period_return = (self.equity - self.equity_curve[-2]) / self.equity_curve[-2]
        self.returns.append(period_return)
        self.positions.append(target)

        # Drawdown
        peak = max(self.equity_curve)
        dd = (peak - self.equity) / peak

        # Termination
        self.step_count += 1
        self.data_idx += 1
        done = dd >= self.max_drawdown_limit or self.equity <= self.initial_equity * 0.5
        truncated = self.step_count >= self.max_steps

        # Reward
        reward = 0.0
        if len(self.returns) >= 5:
            recent = self.returns[-5:]
            reward += float(np.mean(recent) / (np.std(recent) + 1e-8))  # risk-adjusted
        else:
            reward += pnl / (self.initial_equity * 0.01)
        reward -= dd * 10  # drawdown penalty
        if dd >= self.max_drawdown_limit:
            reward -= 5.0

        info = {"equity": self.equity, "drawdown": dd, "pnl": pnl, "scale": scale}
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self) -> np.ndarray:
        confidence = float(self._rng.beta(2, 2))
        direction = self._rng.choice([1.0, -1.0, 0.0], p=[0.4, 0.4, 0.2])

        vol = float(np.std(self.returns[-20:]) * np.sqrt(252)) if len(self.returns) >= 20 else 0.15
        peak = max(self.equity_curve) if self.equity_curve else self.initial_equity
        dd = (peak - self.equity) / peak
        risk_budget = abs(self.position) / (self.equity * self.max_position_pct + 1e-8)

        sharpe = 0.0
        if len(self.returns) >= 20:
            r = self.returns[-20:]
            sharpe = float(np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252))

        pos_pct = self.position / (self.equity * self.max_position_pct + 1e-8)

        time_since = 0
        for i in range(len(self.positions) - 1, -1, -1):
            if i > 0 and self.positions[i] != self.positions[i - 1]:
                break
            time_since += 1

        regime = 0
        if vol > 0.25:
            regime = 1
        elif vol < 0.10:
            regime = -1

        win_rate = 0.5
        if len(self.returns) >= 10:
            win_rate = float(np.mean([r > 0 for r in self.returns[-20:]]))

        return np.array(
            [
                np.clip(confidence, 0, 1),
                direction,
                np.clip(vol / 0.3, 0, 5),
                np.clip(dd / self.max_drawdown_limit, 0, 2),
                np.clip(risk_budget, 0, 1),
                np.clip(sharpe / 3, -2, 2),
                np.clip(pos_pct, -1, 1),
                np.clip(time_since / 10, 0, 1),
                regime,
                np.clip(win_rate, 0, 1),
            ],
            dtype=np.float32,
        )

    def _get_return(self) -> float:
        if self.data is not None and "close" in self.data.columns:
            if 0 < self.data_idx < len(self.data):
                cur = self.data.iloc[self.data_idx]["close"]
                prev = self.data.iloc[self.data_idx - 1]["close"]
                return float((cur - prev) / prev)
        return float(self._rng.normal(0.0005, 0.015))


class AlphaSelectionEnv(gym.Env):
    """
    Alpha signal selection environment.

    Learns which alpha signal to follow given the current regime and
    each alpha's recent performance.

    Observation (variable): regime_onehot(4) + per_alpha(4*n) + market(4)
    Action (Discrete n_alphas+1): select alpha or no-trade
    Reward: selected_return + regime_bonus - regret_penalty - switching_cost
    """

    metadata = {"render_modes": []}

    REGIME_ALIGNMENTS = {
        0: {"momentum": 0.8, "mean_reversion": 0.3, "macro": 0.7},
        1: {"momentum": 0.5, "mean_reversion": 0.6, "macro": 0.8},
        2: {"momentum": 0.3, "mean_reversion": 0.8, "macro": 0.5},
        3: {"momentum": 0.6, "mean_reversion": 0.4, "macro": 0.7},
    }

    def __init__(
        self,
        alpha_names: list[str] | None = None,
        alpha_returns: dict[str, list[float]] | None = None,
        lookback: int = 20,
        max_steps: int = 100,
        seed: int | None = None,
    ):
        super().__init__()
        self.alpha_names = alpha_names or [
            "momentum",
            "mean_reversion",
            "macro",
            "microstructure",
            "cross_asset",
            "spread",
            "regime",
        ]
        self.n_alphas = len(self.alpha_names)
        self.alpha_returns_data = alpha_returns
        self.lookback = lookback
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        state_dim = 4 + 4 * self.n_alphas + 4
        self.observation_space = spaces.Box(
            low=-2, high=2, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_alphas + 1)  # +1 for no-trade

        # Episode state
        self.alpha_history: dict[str, list[float]] = {}
        self.selected_history: list[int] = []
        self.regime_history: list[int] = []
        self.step_count = 0
        self.data_idx = 0

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.alpha_history = {n: [] for n in self.alpha_names}
        self.selected_history = []
        self.regime_history = []
        self.step_count = 0
        self.data_idx = 0

        # Warm up history
        for _ in range(self.lookback):
            rets = self._generate_returns()
            for name, r in rets.items():
                self.alpha_history[name].append(r)
            self.regime_history.append(self._get_regime())

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        alpha_rets = self._generate_returns()
        for name, r in alpha_rets.items():
            self.alpha_history[name].append(r)

        if action == self.n_alphas:
            selected_return = 0.0
            selected_name = "NO_TRADE"
        else:
            selected_name = self.alpha_names[action]
            selected_return = alpha_rets[selected_name]

        self.selected_history.append(action)
        self.regime_history.append(self._get_regime())
        self.step_count += 1
        self.data_idx += 1

        # Reward
        best_return = max(alpha_rets.values())
        regret = best_return - selected_return
        regime = self._get_regime()
        alignment = self._get_alignment(selected_name, regime) if action < self.n_alphas else 0.3

        reward = selected_return * 100 - regret * 50 + alignment * 0.5
        if len(self.selected_history) >= 2 and self.selected_history[-1] != self.selected_history[-2]:
            reward -= 0.1

        done = False
        truncated = self.step_count >= self.max_steps

        info = {
            "selected_alpha": selected_name,
            "selected_return": selected_return,
            "best_return": best_return,
        }
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self) -> np.ndarray:
        features: list[float] = []

        # Regime one-hot
        regime = self._get_regime()
        onehot = [0.0, 0.0, 0.0, 0.0]
        onehot[regime] = 1.0
        features.extend(onehot)

        # Per-alpha
        for name in self.alpha_names:
            rets = self.alpha_history.get(name, [])
            if len(rets) >= self.lookback:
                recent = rets[-self.lookback :]
                sharpe = float(np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252))
                recent_ret = float(np.sum(recent))
                hit = float(np.mean([r > 0 for r in recent]))
            else:
                sharpe, recent_ret, hit = 0.0, 0.0, 0.5
            alignment = self._get_alignment(name, regime)
            features.extend([
                np.clip(sharpe / 3, -2, 2),
                np.clip(recent_ret * 10, -1, 1),
                np.clip(hit, 0, 1),
                np.clip(alignment, 0, 1),
            ])

        # Market features (neutral defaults)
        features.extend([0.3, 0.0, 0.0, 0.3])

        return np.array(features, dtype=np.float32)

    def _generate_returns(self) -> dict[str, float]:
        if self.alpha_returns_data:
            rets = {}
            for name in self.alpha_names:
                data = self.alpha_returns_data.get(name, [])
                if self.data_idx < len(data):
                    rets[name] = float(data[self.data_idx])
                else:
                    rets[name] = float(self._rng.normal(0.001, 0.01))
            return rets

        regime = self._get_regime()
        rets = {}
        for name in self.alpha_names:
            base = float(self._rng.normal(0.001, 0.01))
            alignment = self._get_alignment(name, regime)
            base += (alignment - 0.4) * 0.003
            rets[name] = base
        return rets

    def _get_regime(self) -> int:
        if self.regime_history:
            prev = self.regime_history[-1]
            if self._rng.random() < 0.05:
                return int(self._rng.integers(4))
            return prev
        return int(self._rng.integers(4))

    def _get_alignment(self, alpha_name: str, regime: int) -> float:
        return self.REGIME_ALIGNMENTS.get(regime, {}).get(alpha_name, 0.4)
