# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Gamma Scalping — buy ATM straddles and delta-hedge periodically.

The strategy profits when realised vol exceeds implied vol.  A long ATM
straddle has positive gamma — every time the underlying moves, the
position's delta shifts and the hedge locks in a small profit.  If
cumulative gamma P&L exceeds theta bleed, the trade is profitable.

Key risk: theta decay.  If realised vol underwhelms, the position bleeds
premium faster than hedging recovers.  The ``bleed_ratio`` metric tracks
this balance and triggers an exit when theta dominates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger


@dataclass
class GammaScalpConfig:
    """Tuneable parameters for gamma-scalping strategy."""

    hedge_interval_minutes: int = 30
    """How often to evaluate delta-hedge adjustments."""

    hedge_threshold_pct: float = 0.005
    """Minimum absolute delta (as fraction of notional) before hedging."""

    theta_bleed_exit_ratio: float = 1.5
    """Exit if cumulative theta cost / cumulative gamma P&L exceeds this."""

    min_rv_iv_spread: float = 0.05
    """Minimum RV - IV required to enter (annualised vol points)."""

    max_holding_days: int = 14
    """Hard time stop."""

    target_gamma_dollars: float = 100.0
    """Target dollar gamma per 1% underlying move (position sizing guide)."""


class GammaScalpingStrategy:
    """Buy ATM straddles, delta-hedge periodically, profit from realised > IV."""

    def __init__(self, config: GammaScalpConfig | None = None) -> None:
        self.config = config or GammaScalpConfig()

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        realized_vol_21d: float,
        atm_iv: float,
        iv_rank: float,
    ) -> dict | None:
        """Generate entry signal when RV is expected to exceed IV.

        Parameters
        ----------
        symbol:
            Underlying ticker.
        realized_vol_21d:
            21-day annualised realised vol.
        atm_iv:
            Current at-the-money implied vol.
        iv_rank:
            IV percentile rank (0-100).  Low IV rank means options are
            cheap — favourable for buying vol.

        Returns
        -------
        dict with ``direction``, ``structure``, ``confidence``,
        ``rv_iv_spread`` — or ``None``.
        """
        rv_iv_spread = realized_vol_21d - atm_iv

        if rv_iv_spread < self.config.min_rv_iv_spread:
            logger.debug(
                "{} RV-IV spread {:.4f} below min {:.4f}",
                symbol,
                rv_iv_spread,
                self.config.min_rv_iv_spread,
            )
            return None

        # Prefer entries when IV is relatively low (options are cheap)
        iv_rank_bonus = max(0.0, (50.0 - iv_rank) / 100.0)
        confidence = min(0.95, 0.50 + rv_iv_spread + iv_rank_bonus * 0.15)

        signal = {
            "symbol": symbol,
            "direction": "buy_vol",
            "structure": "long_straddle",
            "confidence": round(confidence, 4),
            "rv_iv_spread": round(rv_iv_spread, 6),
            "realized_vol_21d": round(realized_vol_21d, 6),
            "atm_iv": round(atm_iv, 6),
            "iv_rank": round(iv_rank, 2),
            "target_gamma_dollars": self.config.target_gamma_dollars,
        }
        logger.info("{} gamma-scalp signal: {}", symbol, signal)
        return signal

    # ------------------------------------------------------------------
    # Delta-hedge evaluation
    # ------------------------------------------------------------------

    def compute_hedge_action(
        self,
        position_delta: float,
        underlying_price: float,
        hedge_threshold_pct: float | None = None,
    ) -> dict | None:
        """Determine whether a delta hedge is needed and its parameters.

        Parameters
        ----------
        position_delta:
            Net delta of the options position (positive = long underlying
            exposure, negative = short).
        underlying_price:
            Current price of the underlying.
        hedge_threshold_pct:
            Override for ``config.hedge_threshold_pct``.

        Returns
        -------
        dict with ``action`` ("buy" | "sell"), ``shares``,
        ``notional`` — or ``None`` if delta is within threshold.
        """
        threshold = hedge_threshold_pct or self.config.hedge_threshold_pct
        delta_dollars = position_delta * underlying_price

        if abs(position_delta) < threshold:
            return None

        # Hedge to flat: sell shares when delta is positive, buy when negative
        shares = -round(position_delta)  # integer shares
        if shares == 0:
            return None

        action = "buy" if shares > 0 else "sell"
        hedge = {
            "action": action,
            "shares": abs(shares),
            "notional": round(abs(shares) * underlying_price, 2),
            "position_delta_before": round(position_delta, 4),
        }
        logger.debug("Gamma-scalp hedge: {}", hedge)
        return hedge

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def should_exit(
        self,
        entry_metadata: dict,
        bleed_ratio: float,
        days_held: int,
        rv_minus_iv: float,
    ) -> tuple[bool, str]:
        """Decide whether to close the gamma-scalp position.

        Parameters
        ----------
        entry_metadata:
            Original signal dict stored at entry.
        bleed_ratio:
            ``cumulative_theta / cumulative_gamma_pnl``.  Values > 1.0
            mean theta is eating into profits.
        days_held:
            Calendar days since entry.
        rv_minus_iv:
            Current realised vol minus implied vol.

        Returns
        -------
        (should_close, reason)
        """
        # Theta bleed dominating gamma gains
        if bleed_ratio >= self.config.theta_bleed_exit_ratio:
            return True, (
                f"theta_bleed (ratio={bleed_ratio:.2f} >= "
                f"{self.config.theta_bleed_exit_ratio:.2f})"
            )

        # Time stop
        if days_held >= self.config.max_holding_days:
            return True, f"time_stop ({days_held}d >= {self.config.max_holding_days}d)"

        # Vol environment flipped: RV now below IV
        if rv_minus_iv < -self.config.min_rv_iv_spread:
            return True, (
                f"vol_flip (RV-IV={rv_minus_iv:.4f}, "
                f"entry required >={self.config.min_rv_iv_spread:.4f})"
            )

        return False, ""


class GammaScalpPnLTracker:
    """Track cumulative gamma P&L vs theta cost for a gamma-scalp position.

    Each hedge event contributes gamma P&L; each day contributes theta cost.
    The ``bleed_ratio`` summarises the balance.
    """

    def __init__(self) -> None:
        self._cumulative_gamma_pnl: float = 0.0
        self._cumulative_theta_cost: float = 0.0
        self._hedge_count: int = 0

    @property
    def cumulative_gamma_pnl(self) -> float:
        return self._cumulative_gamma_pnl

    @property
    def cumulative_theta_cost(self) -> float:
        return self._cumulative_theta_cost

    @property
    def hedge_count(self) -> int:
        return self._hedge_count

    def record_hedge(self, gamma_pnl: float) -> None:
        """Record P&L from a single delta-hedge round-trip."""
        self._cumulative_gamma_pnl += gamma_pnl
        self._hedge_count += 1

    def record_theta(self, daily_theta: float) -> None:
        """Record one day's theta decay (pass as positive value)."""
        self._cumulative_theta_cost += abs(daily_theta)

    def bleed_ratio(self) -> float:
        """Ratio of cumulative theta cost to cumulative gamma P&L.

        Returns ``0.0`` when no gamma P&L has been realised yet (avoid
        division by zero).  A value above 1.0 means theta is winning.
        """
        if self._cumulative_gamma_pnl <= 0:
            # No gamma gains yet — theta is entirely unrecovered
            return float("inf") if self._cumulative_theta_cost > 0 else 0.0
        return self._cumulative_theta_cost / self._cumulative_gamma_pnl

    def net_pnl(self) -> float:
        """Net P&L = gamma gains minus theta cost."""
        return self._cumulative_gamma_pnl - self._cumulative_theta_cost

    def summary(self) -> dict:
        """Snapshot of tracking state."""
        return {
            "cumulative_gamma_pnl": round(self._cumulative_gamma_pnl, 2),
            "cumulative_theta_cost": round(self._cumulative_theta_cost, 2),
            "net_pnl": round(self.net_pnl(), 2),
            "bleed_ratio": round(self.bleed_ratio(), 4),
            "hedge_count": self._hedge_count,
        }
