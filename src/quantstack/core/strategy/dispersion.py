# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dispersion Trading — sell index vol, buy component vol.

Trades the spread between implied correlation (derived from index and
single-stock IVs) and realised correlation.  When implied correlation is
significantly higher than realised, the strategy sells index vol and buys
component vol, profiting as the correlation premium decays.

This is a *relative-value* strategy: vega exposure is largely hedged out
because long and short legs offset.  The primary risk is a sudden spike in
realised correlation (market crash / macro shock).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DispersionConfig:
    """Tuneable parameters for the dispersion strategy."""

    correlation_threshold: float = 0.10
    """Minimum implied-minus-realised correlation spread to trigger a signal."""

    min_components: int = 5
    """Minimum basket size to compute meaningful correlation."""

    index_symbol: str = "SPY"
    """Index used as the short-vol leg."""

    max_holding_days: int = 21
    """Hard time stop."""

    profit_target_pct: float = 0.40
    """Take-profit as fraction of initial credit."""

    max_correlation_spike: float = 0.85
    """Emergency exit if realised correlation exceeds this level."""


# ------------------------------------------------------------------
# Free functions — usable outside the strategy class
# ------------------------------------------------------------------


def compute_implied_correlation(
    index_iv: float,
    component_ivs: list[float],
    weights: list[float],
) -> float | None:
    """Derive implied correlation from index IV and weighted-component IVs.

    Uses the standard approximation:

        index_var ≈ Σ wi² σi² + ρ_impl Σ_{i≠j} wi wj σi σj

    Solving for ρ_impl gives:

        ρ_impl = (σ_idx² - Σ wi² σi²) / (Σ_{i≠j} wi wj σi σj)

    Returns ``None`` when the denominator is near-zero or inputs are invalid.
    """
    if len(component_ivs) != len(weights):
        logger.warning("component_ivs length {} != weights length {}", len(component_ivs), len(weights))
        return None

    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(component_ivs, dtype=np.float64)

    if len(w) < 2 or np.any(s <= 0) or np.any(w <= 0):
        return None

    # Normalise weights
    w = w / w.sum()

    weighted_var_sum = float(np.sum((w * s) ** 2))
    cross_sum = float(np.sum(np.outer(w * s, w * s)) - weighted_var_sum)

    if cross_sum <= 1e-12:
        return None

    implied_corr = (index_iv**2 - weighted_var_sum) / cross_sum
    # Clamp to [-1, 1] — values outside this range indicate model breakdown
    implied_corr = float(np.clip(implied_corr, -1.0, 1.0))
    return implied_corr


def compute_realized_correlation(
    returns_matrix: pd.DataFrame,
    window: int = 21,
) -> float:
    """Average pairwise realised correlation over trailing *window* days.

    Parameters
    ----------
    returns_matrix:
        DataFrame of daily returns, one column per component.
    window:
        Lookback in trading days.

    Returns
    -------
    Average off-diagonal element of the trailing correlation matrix.
    """
    tail = returns_matrix.iloc[-window:]
    if tail.shape[1] < 2 or len(tail) < window // 2:
        return 0.0

    corr = tail.corr()
    mask = ~np.eye(corr.shape[0], dtype=bool)
    avg_corr = float(corr.values[mask].mean())
    return avg_corr


# ------------------------------------------------------------------
# Strategy class
# ------------------------------------------------------------------


class DispersionStrategy:
    """Sell index vol / buy component vol when implied corr > realised."""

    def __init__(self, config: DispersionConfig | None = None) -> None:
        self.config = config or DispersionConfig()

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        index_iv: float,
        component_ivs: list[float],
        weights: list[float],
        returns_matrix: pd.DataFrame,
    ) -> dict | None:
        """Generate dispersion signal if correlation spread exceeds threshold.

        Returns
        -------
        dict with ``direction``, ``implied_corr``, ``realized_corr``,
        ``corr_spread``, ``confidence`` — or ``None``.
        """
        if len(component_ivs) < self.config.min_components:
            logger.debug(
                "Only {} components, need {}",
                len(component_ivs),
                self.config.min_components,
            )
            return None

        implied_corr = compute_implied_correlation(index_iv, component_ivs, weights)
        if implied_corr is None:
            logger.warning("Could not compute implied correlation")
            return None

        realized_corr = compute_realized_correlation(returns_matrix)
        corr_spread = implied_corr - realized_corr

        if corr_spread < self.config.correlation_threshold:
            logger.debug(
                "Corr spread {:.4f} below threshold {:.4f}",
                corr_spread,
                self.config.correlation_threshold,
            )
            return None

        confidence = min(0.95, 0.5 + corr_spread)

        signal = {
            "direction": "sell_index_buy_components",
            "index_symbol": self.config.index_symbol,
            "implied_corr": round(implied_corr, 4),
            "realized_corr": round(realized_corr, 4),
            "corr_spread": round(corr_spread, 4),
            "confidence": round(confidence, 4),
            "n_components": len(component_ivs),
        }
        logger.info("Dispersion signal: {}", signal)
        return signal

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def should_exit(
        self,
        entry_metadata: dict,
        current_implied_corr: float,
        realized_corr: float,
        pnl_pct: float,
        days_held: int,
    ) -> tuple[bool, str]:
        """Decide whether to unwind the dispersion trade."""
        # Emergency: correlation spike
        if realized_corr >= self.config.max_correlation_spike:
            return True, f"correlation_spike (realized={realized_corr:.2f} >= {self.config.max_correlation_spike:.2f})"

        # Profit target
        if pnl_pct >= self.config.profit_target_pct:
            return True, f"profit_target ({pnl_pct:.2%} >= {self.config.profit_target_pct:.2%})"

        # Time stop
        if days_held >= self.config.max_holding_days:
            return True, f"time_stop ({days_held}d >= {self.config.max_holding_days}d)"

        # Mean-reversion: correlation spread has collapsed
        corr_spread = current_implied_corr - realized_corr
        entry_spread = entry_metadata.get("corr_spread", self.config.correlation_threshold)
        if corr_spread < entry_spread * 0.25:
            return True, f"mean_reversion (spread {corr_spread:.4f} < 25% of entry {entry_spread:.4f})"

        return False, ""

    # ------------------------------------------------------------------
    # Leg construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_dispersion_legs(
        index_symbol: str,
        components: list[str],
        weights: list[float],
        direction: str = "sell_index_buy_components",
    ) -> list[dict]:
        """Build the list of option legs for a dispersion trade.

        Parameters
        ----------
        index_symbol:
            Ticker for the index (short vol side).
        components:
            List of component tickers (long vol side).
        weights:
            Weight of each component in the index.
        direction:
            Trade direction; currently only ``sell_index_buy_components``
            is supported.

        Returns
        -------
        List of leg dicts with ``symbol``, ``side``, ``structure``, ``weight``.
        """
        if direction != "sell_index_buy_components":
            logger.warning("Unsupported dispersion direction: {}", direction)
            return []

        legs: list[dict] = [
            {
                "symbol": index_symbol,
                "side": "sell",
                "structure": "straddle",
                "weight": 1.0,
            }
        ]

        w_arr = np.asarray(weights, dtype=np.float64)
        w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr

        for ticker, w in zip(components, w_arr):
            legs.append(
                {
                    "symbol": ticker,
                    "side": "buy",
                    "structure": "straddle",
                    "weight": round(float(w), 6),
                }
            )

        return legs
