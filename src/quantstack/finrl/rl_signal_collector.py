# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
RL Signal Collector — integrates RL model predictions as a signal source.

Loads the latest promoted (or shadow) FinRL model from the model registry,
runs inference on the current market state, and returns a signal dict
compatible with the signal engine's collector interface.

Signal weight: 0.15 (same tier as the ML collector).

Returns ``None`` when no trained model is available for the requested
symbol — this is expected for most symbols and is not an error.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.db import db_conn
from quantstack.finrl.config import get_finrl_config
from quantstack.finrl.model_registry import ModelRegistry


# Default signal weight — same tier as ML collector.
RL_SIGNAL_WEIGHT: float = 0.15


async def collect_rl_signals(
    symbol: str,
    bars_df: pd.DataFrame,
    regime: str,
) -> dict[str, Any] | None:
    """
    Collect an RL-based signal for *symbol*.

    Parameters
    ----------
    symbol : str
        Ticker symbol to generate signal for.
    bars_df : pd.DataFrame
        Recent OHLCV bars (sorted ascending by date).  Must contain at
        least ``close`` column.
    regime : str
        Current market regime label (e.g. ``"trending_up"``,
        ``"ranging"``).

    Returns
    -------
    dict | None
        ``{"signal_value": float, "confidence": float, "model_id": str,
        "strategy": str}`` on success, or ``None`` if no trained model
        is available.
    """
    try:
        return await asyncio.to_thread(
            _collect_rl_signal_sync, symbol, bars_df, regime
        )
    except Exception as exc:
        logger.warning("[rl_signal] %s: collection failed: %s — returning None", symbol, exc)
        return None


def _collect_rl_signal_sync(
    symbol: str,
    bars_df: pd.DataFrame,
    regime: str,
) -> dict[str, Any] | None:
    """Synchronous RL signal collection — called via ``asyncio.to_thread``."""

    model_meta = _load_best_model(symbol)
    if model_meta is None:
        logger.debug("[rl_signal] %s: no trained RL model found", symbol)
        return None

    model_id: str = model_meta["model_id"]
    checkpoint_path: str = model_meta["checkpoint_path"]
    algorithm: str = model_meta.get("algorithm", "ppo")
    env_type: str = model_meta.get("env_type", "stock_trading")

    if not Path(checkpoint_path).exists():
        logger.warning(
            "[rl_signal] %s: checkpoint missing at %s — skipping",
            symbol,
            checkpoint_path,
        )
        return None

    # Build observation from bars_df.
    obs = _build_observation(bars_df, env_type)
    if obs is None:
        logger.debug("[rl_signal] %s: insufficient data for observation", symbol)
        return None

    # Load model and run inference.
    action, confidence = _run_inference(checkpoint_path, algorithm, obs)
    if action is None:
        return None

    # Translate raw action to a directional signal in [-1, 1].
    signal_value = _action_to_signal(action, env_type)

    # Discount confidence when regime doesn't align with signal direction.
    confidence = _adjust_confidence_for_regime(signal_value, confidence, regime)

    return {
        "signal_value": float(signal_value),
        "confidence": float(confidence),
        "model_id": model_id,
        "strategy": f"rl_{env_type}",
    }


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_best_model(symbol: str) -> dict[str, Any] | None:
    """
    Query the model registry for the best available model covering *symbol*.

    Prefers ``live`` status over ``shadow``.  Returns None if nothing found.
    """
    try:
        with db_conn() as conn:
            registry = ModelRegistry(conn)

            # Try live models first, then shadow.
            for status in ("live", "shadow"):
                models = registry.list_models(status=status)
                for m in models:
                    model_symbols = m.get("symbols", "") or ""
                    if symbol in model_symbols or model_symbols == "":
                        return m

    except Exception as exc:
        logger.debug("[rl_signal] registry query failed: %s", exc)

    return None


def _build_observation(
    bars_df: pd.DataFrame,
    env_type: str,
) -> np.ndarray | None:
    """
    Construct a flat observation vector from recent bars.

    Uses the last 20 bars to compute momentum, volatility, and volume
    features — a lightweight feature set suitable for inference without
    needing the full training feature pipeline.
    """
    if bars_df is None or len(bars_df) < 20:
        return None

    close = bars_df["close"].values[-20:].astype(np.float64)
    volume = bars_df["volume"].values[-20:].astype(np.float64) if "volume" in bars_df.columns else np.ones(20)

    returns = np.diff(close) / (close[:-1] + 1e-12)
    volatility = float(np.std(returns))
    momentum = float(np.mean(returns[-5:]))
    volume_ratio = float(volume[-1] / (np.mean(volume) + 1e-12))
    price_sma_ratio = float(close[-1] / (np.mean(close) + 1e-12))
    rsi = _compute_rsi(close)

    obs = np.array(
        [
            np.clip(momentum * 100, -5, 5),
            np.clip(volatility * 100, 0, 10),
            np.clip(volume_ratio, 0, 5),
            np.clip(price_sma_ratio - 1.0, -0.2, 0.2),
            np.clip(rsi / 100.0, 0, 1),
        ],
        dtype=np.float32,
    )
    return obs


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Simple RSI calculation from price array."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _run_inference(
    checkpoint_path: str,
    algorithm: str,
    obs: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """
    Load a stable-baselines3 model and predict an action.

    Returns (action, confidence) or (None, 0.0) on failure.
    """
    try:
        # Lazy import to avoid loading torch/sb3 when no model exists.
        from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

        algo_map = {
            "ppo": PPO,
            "a2c": A2C,
            "sac": SAC,
            "td3": TD3,
            "ddpg": DDPG,
            "dqn": DQN,
        }

        cls = algo_map.get(algorithm.lower())
        if cls is None:
            logger.warning("[rl_signal] unknown algorithm %s", algorithm)
            return None, 0.0

        model = cls.load(checkpoint_path)
        action, _states = model.predict(obs, deterministic=True)

        # Confidence: use action probability if available, else fixed 0.6.
        confidence = 0.6
        try:
            dist = model.policy.get_distribution(
                model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
            )
            log_prob = dist.log_prob(
                __import__("torch").tensor(action).to(model.device)
            )
            confidence = float(np.clip(np.exp(log_prob.detach().cpu().numpy()), 0.1, 1.0))
        except Exception:
            pass  # Many algos (DDPG, TD3) don't support log_prob cleanly.

        return np.asarray(action), confidence

    except Exception as exc:
        logger.warning("[rl_signal] inference failed: %s", exc)
        return None, 0.0


def _action_to_signal(action: np.ndarray, env_type: str) -> float:
    """
    Convert raw RL action to a directional signal in [-1, 1].

    Interpretation depends on the environment type:
    - stock_trading / portfolio_opt: action represents position sizing
    - execution: not directional — return 0 (neutral)
    - strategy_select: weighted average of allocations mapped to direction
    """
    flat = action.flatten()

    if env_type in ("stock_trading", "sizing"):
        # Continuous action: clip to [-1, 1] directly.
        return float(np.clip(np.mean(flat), -1.0, 1.0))

    if env_type == "portfolio_opt":
        # Weights — net direction from deviation from equal weight.
        equal = 1.0 / max(len(flat), 1)
        deviation = float(np.mean(flat - equal))
        return float(np.clip(deviation * 10, -1.0, 1.0))

    if env_type == "execution":
        # Execution env is not directional.
        return 0.0

    # Fallback: scale mean action.
    return float(np.clip(np.mean(flat) * 2 - 1, -1.0, 1.0))


def _adjust_confidence_for_regime(
    signal_value: float,
    confidence: float,
    regime: str,
) -> float:
    """
    Discount confidence when the signal direction conflicts with the
    current market regime.
    """
    bullish_regimes = {"trending_up", "breakout"}
    bearish_regimes = {"trending_down"}

    if signal_value > 0.2 and regime in bearish_regimes:
        confidence *= 0.5
    elif signal_value < -0.2 and regime in bullish_regimes:
        confidence *= 0.5

    return float(np.clip(confidence, 0.0, 1.0))
