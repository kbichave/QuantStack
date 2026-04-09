"""Neural network for optimal derivatives hedging (Deep Hedging).

Trains a feed-forward network that maps option state
(time-to-expiry, moneyness, realised vol, delta) to an optimal hedge ratio
that minimises a mean-variance utility loss over simulated price paths.

Graceful degradation:
  1. PyTorch available — full neural network training
  2. No PyTorch — Black-Scholes delta (analytical fallback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt
from typing import Any

import numpy as np
from loguru import logger
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------

_HAS_TORCH = False

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    logger.info("torch not installed — DeepHedger will use Black-Scholes delta fallback")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DeepHedgingConfig:
    """Configuration for the deep hedging network.

    Attributes:
        n_steps: Number of discrete hedging time steps per path.
        hidden_layers: Sizes of hidden layers in the hedging network.
        learning_rate: Optimiser step size.
        risk_aversion: Lambda in the mean-variance utility.
            Higher values penalise variance more, producing tighter hedges.
        n_simulations: Number of Monte Carlo price paths for training.
    """

    n_steps: int = 20
    hidden_layers: list[int] = field(default_factory=lambda: [64, 32])
    learning_rate: float = 1e-3
    risk_aversion: float = 1.0
    n_simulations: int = 10_000


# ---------------------------------------------------------------------------
# Black-Scholes helpers (used as fallback and for delta feature)
# ---------------------------------------------------------------------------


def _bs_delta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    is_call: bool = True,
) -> float:
    """Black-Scholes delta for a European option."""
    if T <= 0 or sigma <= 0:
        # At or past expiry — intrinsic delta
        return 1.0 if (S > K and is_call) else 0.0

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    delta = float(norm.cdf(d1))
    return delta if is_call else delta - 1.0


# ---------------------------------------------------------------------------
# Deep Hedger
# ---------------------------------------------------------------------------


class DeepHedger:
    """Learns an optimal hedging strategy via neural-network policy.

    The network takes (time_to_expiry, moneyness, realised_vol, delta) as
    input and outputs a hedge ratio.  Training minimises negative
    mean-variance utility over simulated hedging P&L paths.
    """

    def __init__(self, config: DeepHedgingConfig | None = None) -> None:
        self._cfg = config or DeepHedgingConfig()
        self._model: Any = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        option_data: dict[str, float],
        price_paths: np.ndarray,
    ) -> DeepHedger:
        """Train the hedging network.

        Parameters:
            option_data: Must contain keys "strike", "expiry_years",
                "implied_vol", "is_call" (bool), and optionally "risk_free_rate".
            price_paths: Array of shape (n_simulations, n_steps+1) with
                simulated underlying price paths.  Column 0 is the initial
                price; subsequent columns are prices at each hedging step.

        Returns:
            self for chaining.
        """
        if not _HAS_TORCH:
            logger.info("No torch — DeepHedger will use BS delta fallback (no training)")
            self._fitted = True
            return self

        K = option_data["strike"]
        T = option_data["expiry_years"]
        sigma = option_data["implied_vol"]
        is_call = option_data.get("is_call", True)
        r = option_data.get("risk_free_rate", 0.0)
        n_steps = self._cfg.n_steps

        paths = price_paths  # (n_sims, n_steps+1)
        n_sims = paths.shape[0]
        dt = T / n_steps

        # Build feature tensors per time step
        # Features: (time_to_expiry, moneyness, realised_vol, bs_delta)
        features_list: list[torch.Tensor] = []  # type: ignore[name-defined]
        for t_idx in range(n_steps):
            ttm = T - t_idx * dt
            S_t = paths[:, t_idx]
            moneyness = S_t / K

            # Rolling realised vol (use path history up to t_idx)
            if t_idx > 0:
                log_rets = np.log(paths[:, 1 : t_idx + 1] / paths[:, :t_idx])
                rvol = np.std(log_rets, axis=1) * np.sqrt(252)
            else:
                rvol = np.full(n_sims, sigma)

            bs_d = np.array([_bs_delta(s, K, ttm, sigma, r, is_call) for s in S_t])

            feat = np.stack([
                np.full(n_sims, ttm),
                moneyness,
                rvol,
                bs_d,
            ], axis=1).astype(np.float32)

            features_list.append(torch.tensor(feat))  # type: ignore[name-defined]

        # Build network
        model = _HedgingNetwork(
            input_dim=4,
            hidden_layers=self._cfg.hidden_layers,
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=self._cfg.learning_rate)  # type: ignore[name-defined]

        paths_t = torch.tensor(paths.astype(np.float32))  # type: ignore[name-defined]

        # Option payoff
        if is_call:
            payoff = torch.clamp(paths_t[:, -1] - K, min=0.0)  # type: ignore[name-defined]
        else:
            payoff = torch.clamp(K - paths_t[:, -1], min=0.0)  # type: ignore[name-defined]

        model.train()
        for epoch in range(200):
            optimiser.zero_grad()

            # Simulate hedging P&L
            hedge_pnl = torch.zeros(n_sims)  # type: ignore[name-defined]
            for t_idx in range(n_steps):
                h = model(features_list[t_idx]).squeeze()
                dS = paths_t[:, t_idx + 1] - paths_t[:, t_idx]
                hedge_pnl = hedge_pnl + h * dS

            # Total P&L = hedge gains - option payoff (short option position)
            total_pnl = hedge_pnl - payoff

            # Negative mean-variance utility
            mean_pnl = total_pnl.mean()
            var_pnl = total_pnl.var()
            loss = -(mean_pnl - self._cfg.risk_aversion * var_pnl)

            loss.backward()
            optimiser.step()

            if (epoch + 1) % 50 == 0:
                logger.debug(
                    "DeepHedger epoch {}: loss={:.4f}, mean_pnl={:.4f}, std_pnl={:.4f}",
                    epoch + 1,
                    loss.item(),
                    mean_pnl.item(),
                    var_pnl.item() ** 0.5,
                )

        model.eval()
        self._model = model
        self._fitted = True
        logger.info("DeepHedger trained on {} paths x {} steps", n_sims, n_steps)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def hedge_ratio(self, state: dict[str, float]) -> float:
        """Return the optimal hedge ratio for the given option state.

        Parameters:
            state: Must contain "spot", "strike", "time_to_expiry",
                "implied_vol". Optionally "realised_vol", "is_call",
                "risk_free_rate".

        Returns:
            Hedge ratio (typically between 0 and 1 for calls).
        """
        S = state["spot"]
        K = state["strike"]
        ttm = state["time_to_expiry"]
        sigma = state["implied_vol"]
        r = state.get("risk_free_rate", 0.0)
        is_call = state.get("is_call", True)
        rvol = state.get("realised_vol", sigma)

        # Fallback: Black-Scholes delta
        if not _HAS_TORCH or self._model is None:
            return _bs_delta(S, K, ttm, sigma, r, is_call)

        bs_d = _bs_delta(S, K, ttm, sigma, r, is_call)
        feat = torch.tensor(  # type: ignore[name-defined]
            [[ttm, S / K, rvol, bs_d]],
            dtype=torch.float32,  # type: ignore[name-defined]
        )
        with torch.no_grad():  # type: ignore[name-defined]
            ratio = self._model(feat).item()

        return float(np.clip(ratio, -1.0, 2.0))

    # ------------------------------------------------------------------
    # P&L simulation
    # ------------------------------------------------------------------

    def compute_hedging_pnl(
        self,
        hedge_ratios: np.ndarray,
        price_paths: np.ndarray,
    ) -> dict[str, float]:
        """Simulate hedging P&L given pre-computed hedge ratios.

        Parameters:
            hedge_ratios: Array of shape (n_sims, n_steps) with hedge ratios
                at each rebalance point.
            price_paths: Array of shape (n_sims, n_steps+1) with price paths.

        Returns:
            {mean_pnl, std_pnl, max_loss, sharpe}
        """
        n_sims, n_steps = hedge_ratios.shape
        pnl = np.zeros(n_sims)

        for t in range(n_steps):
            dS = price_paths[:, t + 1] - price_paths[:, t]
            pnl += hedge_ratios[:, t] * dS

        mean_pnl = float(np.mean(pnl))
        std_pnl = float(np.std(pnl))
        max_loss = float(np.min(pnl))
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

        logger.info(
            "Hedging P&L: mean={:.4f}, std={:.4f}, max_loss={:.4f}, sharpe={:.4f}",
            mean_pnl,
            std_pnl,
            max_loss,
            sharpe,
        )
        return {
            "mean_pnl": mean_pnl,
            "std_pnl": std_pnl,
            "max_loss": max_loss,
            "sharpe": sharpe,
        }


# ---------------------------------------------------------------------------
# Neural network (only defined when torch is available)
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class _HedgingNetwork(nn.Module):  # type: ignore[name-defined]
        """Feed-forward network mapping option state to hedge ratio."""

        def __init__(self, input_dim: int = 4, hidden_layers: list[int] | None = None):
            super().__init__()
            layers: list[nn.Module] = []  # type: ignore[name-defined]
            hidden = hidden_layers or [64, 32]
            prev = input_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))  # type: ignore[name-defined]
                layers.append(nn.ReLU())  # type: ignore[name-defined]
                prev = h
            layers.append(nn.Linear(prev, 1))  # type: ignore[name-defined]
            layers.append(nn.Sigmoid())  # type: ignore[name-defined]
            self.net = nn.Sequential(*layers)  # type: ignore[name-defined]

        def forward(self, x: Any) -> Any:
            return self.net(x)
