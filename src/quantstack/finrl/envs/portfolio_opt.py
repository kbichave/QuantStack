# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Portfolio Optimization Gymnasium Environment.

Learns target portfolio weights across multiple symbols to maximize
risk-adjusted returns.  The agent observes a rolling window of OHLCV
plus technical features and outputs a weight vector that is softmax-
normalized to sum to 1.

Reward combines daily portfolio return, a volatility penalty to
discourage erratic equity curves, and a turnover penalty to keep
transaction costs in check.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


# Number of technical features appended per symbol alongside OHLCV (5).
_TECHNICAL_FEATURE_COUNT = 20
_OHLCV_COUNT = 5
_FEATURES_PER_SYMBOL = _OHLCV_COUNT + _TECHNICAL_FEATURE_COUNT


class PortfolioOptEnv(gym.Env):
    """
    Multi-asset portfolio weight optimization environment.

    Observation space
        Box with shape ``(window_size, n_symbols * n_features)``.
        Each row is one time-step; columns are per-symbol OHLCV (5) plus
        20 technical indicator values, concatenated across all symbols.

    Action space
        Box(low=0, high=1, shape=(n_symbols,)).  Raw outputs are softmax-
        normalized so they sum to 1 before being applied as weights.

    Reward
        ``daily_portfolio_return
          - vol_penalty * portfolio_volatility
          - turnover_penalty * sum(|weight_change|)``

    Parameters
    ----------
    df : pd.DataFrame
        Long-format frame with columns ``[date, symbol, open, high, low,
        close, volume, feat_0 .. feat_19]``.  Must be sorted by
        ``(date, symbol)``.
    symbols : list[str]
        Ordered list of symbols present in *df*.
    window_size : int
        Number of historical bars visible per observation.
    vol_penalty : float
        Weight applied to realized portfolio volatility in the reward.
    turnover_penalty : float
        Weight applied to absolute weight change in the reward.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        window_size: int = 20,
        vol_penalty: float = 0.5,
        turnover_penalty: float = 0.001,
    ) -> None:
        super().__init__()

        self.symbols = list(symbols)
        self.n_symbols = len(self.symbols)
        self.window_size = window_size
        self.vol_penalty = vol_penalty
        self.turnover_penalty = turnover_penalty

        # --- Pre-process data into a 3-D array (dates x symbols x features) ---
        self._build_tensor(df)

        n_features = _FEATURES_PER_SYMBOL * self.n_symbols
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_symbols,),
            dtype=np.float32,
        )

        # Episode state
        self._step_idx: int = 0
        self._prev_weights: np.ndarray = np.ones(self.n_symbols, dtype=np.float64) / self.n_symbols
        self.portfolio_value: float = 1.0
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

        self._step_idx = self.window_size  # first valid index after a full window
        self._prev_weights = np.ones(self.n_symbols, dtype=np.float64) / self.n_symbols
        self.portfolio_value = 1.0
        self.portfolio_returns = []

        obs = self._get_observation()
        info: dict[str, Any] = {"portfolio_value": self.portfolio_value}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Normalize weights to sum to 1 via softmax to guarantee valid allocation.
        weights = _softmax(np.asarray(action, dtype=np.float64))

        # Per-symbol daily returns at the current step.
        daily_returns = self._symbol_returns[self._step_idx]  # shape (n_symbols,)

        reward = self._compute_reward(weights, self._prev_weights, daily_returns)

        # Update portfolio value.
        port_return = float(np.dot(weights, daily_returns))
        self.portfolio_value *= 1.0 + port_return
        self.portfolio_returns.append(port_return)

        self._prev_weights = weights.copy()
        self._step_idx += 1

        terminated = self._step_idx >= self._n_dates - 1
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info: dict[str, Any] = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": port_return,
            "weights": weights.tolist(),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Return the rolling window observation ending at ``_step_idx``."""
        start = self._step_idx - self.window_size
        # _data shape: (n_dates, n_symbols * n_features)
        window = self._data[start : self._step_idx]
        return window.astype(np.float32)

    def _compute_reward(
        self,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        daily_returns: np.ndarray,
    ) -> float:
        """
        Reward = daily_portfolio_return
                 - vol_penalty * portfolio_vol
                 - turnover_penalty * |weight_change|
        """
        port_return = float(np.dot(weights, daily_returns))

        # Realized rolling volatility of portfolio returns.
        if len(self.portfolio_returns) >= 2:
            port_vol = float(np.std(self.portfolio_returns[-20:]))
        else:
            port_vol = 0.0

        turnover = float(np.sum(np.abs(weights - prev_weights)))

        reward = (
            port_return
            - self.vol_penalty * port_vol
            - self.turnover_penalty * turnover
        )
        return reward

    def _build_tensor(self, df: pd.DataFrame) -> None:
        """
        Convert the long-format DataFrame into dense numpy arrays.

        After this call:
            self._data           — (n_dates, n_symbols * n_features)
            self._symbol_returns — (n_dates, n_symbols)  daily close-to-close
            self._n_dates        — total trading days
        """
        # Identify feature columns beyond OHLCV.
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = [c for c in df.columns if c not in (*ohlcv_cols, "date", "symbol")]
        if len(feature_cols) < _TECHNICAL_FEATURE_COUNT:
            logger.warning(
                "PortfolioOptEnv: expected %d technical features, got %d — padding with zeros",
                _TECHNICAL_FEATURE_COUNT,
                len(feature_cols),
            )

        dates = sorted(df["date"].unique())
        self._n_dates = len(dates)
        date_to_idx = {d: i for i, d in enumerate(dates)}

        n_features = _FEATURES_PER_SYMBOL * self.n_symbols
        self._data = np.zeros((self._n_dates, n_features), dtype=np.float64)
        self._symbol_returns = np.zeros((self._n_dates, self.n_symbols), dtype=np.float64)

        for sym_idx, sym in enumerate(self.symbols):
            sym_df = df.loc[df["symbol"] == sym].sort_values("date")
            col_offset = sym_idx * _FEATURES_PER_SYMBOL

            prev_close: float | None = None
            for _, row in sym_df.iterrows():
                t = date_to_idx.get(row["date"])
                if t is None:
                    continue

                # OHLCV
                self._data[t, col_offset + 0] = row["open"]
                self._data[t, col_offset + 1] = row["high"]
                self._data[t, col_offset + 2] = row["low"]
                self._data[t, col_offset + 3] = row["close"]
                self._data[t, col_offset + 4] = row["volume"]

                # Technical features
                for f_idx, fc in enumerate(feature_cols[:_TECHNICAL_FEATURE_COUNT]):
                    self._data[t, col_offset + _OHLCV_COUNT + f_idx] = row.get(fc, 0.0)

                # Daily return
                close = float(row["close"])
                if prev_close is not None and prev_close != 0.0:
                    self._symbol_returns[t, sym_idx] = (close - prev_close) / prev_close
                prev_close = close

        logger.debug(
            "PortfolioOptEnv: built tensor (%d dates, %d symbols, %d features/symbol)",
            self._n_dates,
            self.n_symbols,
            _FEATURES_PER_SYMBOL,
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax — guarantees weights sum to 1 and are non-negative."""
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)
