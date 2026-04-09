# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Vol Arb Strategy — sell/buy straddles based on IV rank vs realized vol.

Trades the spread between implied and realized volatility. When IV is rich
relative to RV (high z-score), sells vol via iron condors or straddles.
When IV is cheap, buys vol via long straddles. Exits on mean-reversion,
profit target, or time stop.

Complements the discovery-layer computations in
``quantstack.alpha_discovery.vol_arb`` by wrapping them in a strategy object
with entry/exit logic and position-level state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class VolArbConfig:
    """Tuneable parameters for the vol-arb strategy."""

    iv_rank_min: float = 50.0
    """Minimum IV rank (0-100) required to consider a sell-vol signal."""

    rv_iv_divergence_threshold: float = 0.10
    """Minimum absolute spread (|IV - RV|) to trigger a signal."""

    z_threshold: float = 1.0
    """Z-score threshold on the IV-RV spread for signal generation."""

    profit_target_pct: float = 0.50
    """Take-profit as fraction of initial premium collected."""

    max_holding_days: int = 30
    """Hard time stop — close position after this many calendar days."""

    hedge_frequency_minutes: int = 30
    """How often delta-hedge adjustments should be evaluated."""


class VolArbStrategy:
    """Sell or buy straddles based on IV rank vs realised vol."""

    def __init__(self, config: VolArbConfig | None = None) -> None:
        self.config = config or VolArbConfig()

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        iv_rank: float,
        realized_vol: float,
        implied_vol: float,
        historical_spreads: np.ndarray | None = None,
    ) -> dict | None:
        """Evaluate whether to sell or buy vol for *symbol*.

        Parameters
        ----------
        symbol:
            Underlying ticker.
        iv_rank:
            Current IV percentile rank (0-100).
        realized_vol:
            Annualised realised vol (e.g. 21-day close-to-close).
        implied_vol:
            Current 30-day ATM implied vol.
        historical_spreads:
            Array of past IV-RV spread values for z-score calc.
            If ``None``, z-score gating is skipped and only the
            divergence threshold is used.

        Returns
        -------
        dict with keys ``direction``, ``structure``, ``confidence``,
        ``vol_spread``, ``z_score``, ``symbol`` — or ``None`` if no
        signal.
        """
        spread = self.compute_vol_spread(realized_vol, implied_vol)
        abs_spread = abs(spread)

        if abs_spread < self.config.rv_iv_divergence_threshold:
            logger.debug(
                "{} vol spread {:.4f} below threshold {:.4f}",
                symbol,
                abs_spread,
                self.config.rv_iv_divergence_threshold,
            )
            return None

        z_score: float | None = None
        if historical_spreads is not None and len(historical_spreads) >= 60:
            z_score = self.compute_z_score(spread, historical_spreads)
            if abs(z_score) < self.config.z_threshold:
                logger.debug(
                    "{} z-score {:.2f} below threshold {:.2f}",
                    symbol,
                    z_score,
                    self.config.z_threshold,
                )
                return None

        # Determine direction
        if spread > 0 and iv_rank >= self.config.iv_rank_min:
            direction = "sell_vol"
            structure = "iron_condor"
            confidence = min(0.95, 0.5 + abs_spread * 2)
        elif spread < 0:
            direction = "buy_vol"
            structure = "long_straddle"
            confidence = min(0.95, 0.5 + abs_spread * 2)
        else:
            return None

        signal = {
            "symbol": symbol,
            "direction": direction,
            "structure": structure,
            "confidence": round(confidence, 4),
            "vol_spread": round(spread, 6),
            "z_score": round(z_score, 4) if z_score is not None else None,
            "iv_rank": round(iv_rank, 2),
            "implied_vol": round(implied_vol, 6),
            "realized_vol": round(realized_vol, 6),
        }
        logger.info("{} vol-arb signal: {}", symbol, signal)
        return signal

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def should_exit(
        self,
        entry_metadata: dict,
        current_pnl_pct: float,
        days_held: int,
        current_z: float,
    ) -> tuple[bool, str]:
        """Decide whether to close the position.

        Returns
        -------
        (should_close, reason)
        """
        # Profit target
        if current_pnl_pct >= self.config.profit_target_pct:
            return True, f"profit_target ({current_pnl_pct:.2%} >= {self.config.profit_target_pct:.2%})"

        # Time stop
        if days_held >= self.config.max_holding_days:
            return True, f"time_stop ({days_held}d >= {self.config.max_holding_days}d)"

        # Mean-reversion: z-score has collapsed back toward zero
        if abs(current_z) < 0.5:
            return True, f"mean_reversion (|z|={abs(current_z):.2f} < 0.5)"

        return False, ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_vol_spread(realized_vol: float, implied_vol: float) -> float:
        """IV minus RV spread.  Positive means IV is rich."""
        return implied_vol - realized_vol

    @staticmethod
    def compute_z_score(spread: float, historical_spreads: np.ndarray) -> float:
        """Z-score of *spread* relative to *historical_spreads*."""
        arr = np.asarray(historical_spreads, dtype=np.float64)
        mu = float(np.nanmean(arr))
        sigma = float(np.nanstd(arr, ddof=1))
        if sigma <= 0:
            return 0.0
        return (spread - mu) / sigma
