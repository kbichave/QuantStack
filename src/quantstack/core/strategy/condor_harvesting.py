# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iron Condor Harvesting — sell OTM spreads in ranging regimes.

Collects premium by selling out-of-the-money put and call spreads when the
market is range-bound and implied volatility is elevated.  The strategy is
theta-positive and vega-negative: it profits from time decay and vol
contraction.

Guard rails:
- Only enters when the detected regime is *ranging* (configurable).
- Exits or rolls when a short strike's delta exceeds a management trigger.
- Emergency exit if regime shifts to *trending*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger


@dataclass
class CondorConfig:
    """Tuneable parameters for iron-condor harvesting."""

    iv_rank_min: float = 50.0
    """Minimum IV rank (0-100) to enter — ensures premium is adequate."""

    regime_required: str = "ranging"
    """Market regime gate (from regime detector)."""

    short_delta_target: float = 0.16
    """Target delta for the short strikes (~1 SD OTM)."""

    wing_width: float = 5.0
    """Distance in dollars between short and long strikes on each side."""

    profit_target_pct: float = 0.50
    """Close when unrealised P&L reaches this fraction of max profit."""

    management_trigger_delta: float = 0.30
    """Roll or close a side when its short-strike delta exceeds this."""

    max_holding_days: int = 45
    """Hard time stop in calendar days."""

    min_dte: int = 21
    """Minimum days to expiration at entry."""

    max_dte: int = 45
    """Maximum days to expiration at entry."""


class CondorHarvestingStrategy:
    """Sell iron condors in ranging, high-IV environments."""

    def __init__(self, config: CondorConfig | None = None) -> None:
        self.config = config or CondorConfig()

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        iv_rank: float,
        regime: str,
        spot_price: float,
    ) -> dict | None:
        """Generate entry signal gated by IV rank and market regime.

        Returns
        -------
        dict with ``symbol``, ``direction``, ``structure``, ``confidence``,
        ``strikes`` — or ``None``.
        """
        if regime != self.config.regime_required:
            logger.debug(
                "{} regime '{}' != required '{}'",
                symbol,
                regime,
                self.config.regime_required,
            )
            return None

        if iv_rank < self.config.iv_rank_min:
            logger.debug(
                "{} IV rank {:.1f} below minimum {:.1f}",
                symbol,
                iv_rank,
                self.config.iv_rank_min,
            )
            return None

        strikes = self.select_condor_strikes(
            spot_price,
            self.config.short_delta_target,
            self.config.wing_width,
        )

        confidence = min(0.90, 0.45 + (iv_rank - self.config.iv_rank_min) / 200.0)

        signal = {
            "symbol": symbol,
            "direction": "sell_condor",
            "structure": "iron_condor",
            "confidence": round(confidence, 4),
            "iv_rank": round(iv_rank, 2),
            "regime": regime,
            "spot_price": round(spot_price, 2),
            "strikes": strikes,
        }
        logger.info("{} condor signal: {}", symbol, signal)
        return signal

    # ------------------------------------------------------------------
    # Strike selection
    # ------------------------------------------------------------------

    @staticmethod
    def select_condor_strikes(
        spot: float,
        short_delta: float,
        wing_width: float,
    ) -> dict:
        """Compute condor strike levels from spot, delta target, and wing width.

        Uses a simplified delta-to-distance approximation. In production the
        actual option chain should be queried to find strikes closest to
        the target delta.

        Returns
        -------
        dict with ``short_put``, ``long_put``, ``short_call``, ``long_call``.
        """
        # Approximate OTM distance: delta of ~0.16 ≈ 1 SD for 30-day options.
        # 1 SD ≈ spot * IV * sqrt(T).  Without IV here, use delta as a proxy
        # ratio (works for ~30-day ATM vol of ~20-30%).
        otm_distance = round(spot * short_delta, 2)

        short_put = round(spot - otm_distance, 2)
        long_put = round(short_put - wing_width, 2)
        short_call = round(spot + otm_distance, 2)
        long_call = round(short_call + wing_width, 2)

        return {
            "short_put": short_put,
            "long_put": long_put,
            "short_call": short_call,
            "long_call": long_call,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_condor_metrics(
        short_put: float,
        long_put: float,
        short_call: float,
        long_call: float,
        credit_received: float,
    ) -> dict:
        """Compute key risk/reward metrics for an iron condor.

        Parameters
        ----------
        short_put, long_put, short_call, long_call:
            Strike prices of the four legs.
        credit_received:
            Net premium collected per share (before multiplier).

        Returns
        -------
        dict with ``max_profit``, ``max_loss``, ``breakeven_lower``,
        ``breakeven_upper``, ``risk_reward``.
        """
        put_wing = short_put - long_put
        call_wing = short_call - long_call  # will be negative; abs it
        max_wing = max(put_wing, abs(call_wing))
        max_loss = max_wing - credit_received
        max_profit = credit_received

        risk_reward = round(max_profit / max_loss, 4) if max_loss > 0 else float("inf")

        return {
            "max_profit": round(max_profit, 4),
            "max_loss": round(max_loss, 4),
            "breakeven_lower": round(short_put - credit_received, 2),
            "breakeven_upper": round(short_call + credit_received, 2),
            "risk_reward": risk_reward,
        }

    # ------------------------------------------------------------------
    # Exit / management
    # ------------------------------------------------------------------

    def should_exit(
        self,
        entry_metadata: dict,
        unrealized_pnl_pct: float,
        dte: int,
        regime: str,
        short_strike_delta: float,
    ) -> tuple[bool, str]:
        """Decide whether to close the condor.

        Parameters
        ----------
        entry_metadata:
            Signal dict stored at entry.
        unrealized_pnl_pct:
            Current unrealised P&L as fraction of max profit.
        dte:
            Days to expiration remaining.
        regime:
            Current detected market regime.
        short_strike_delta:
            Maximum absolute delta across the two short strikes.

        Returns
        -------
        (should_close, reason)
        """
        # Profit target
        if unrealized_pnl_pct >= self.config.profit_target_pct:
            return True, (
                f"profit_target ({unrealized_pnl_pct:.2%} >= "
                f"{self.config.profit_target_pct:.2%})"
            )

        # Regime shift — no longer ranging
        if regime != self.config.regime_required:
            return True, f"regime_shift (now '{regime}', required '{self.config.regime_required}')"

        # Short strike breached management trigger
        if short_strike_delta >= self.config.management_trigger_delta:
            return True, (
                f"delta_breach (short delta {short_strike_delta:.2f} >= "
                f"{self.config.management_trigger_delta:.2f})"
            )

        # Time stop — based on days held from entry metadata
        days_held = entry_metadata.get("max_holding_days", self.config.max_holding_days)
        entry_dte = entry_metadata.get("entry_dte", self.config.max_dte)
        elapsed = entry_dte - dte
        if elapsed >= self.config.max_holding_days:
            return True, f"time_stop ({elapsed}d >= {self.config.max_holding_days}d)"

        return False, ""

    def compute_management_action(
        self,
        position_metadata: dict,
        current_deltas: dict,
    ) -> dict | None:
        """Recommend a roll or adjustment when a side is tested.

        Parameters
        ----------
        position_metadata:
            Stored position info including original strikes and credit.
        current_deltas:
            Dict with ``short_put_delta`` and ``short_call_delta``
            (absolute values).

        Returns
        -------
        dict with ``action``, ``side``, ``recommendation`` — or ``None``
        if no management needed.
        """
        put_delta = abs(current_deltas.get("short_put_delta", 0.0))
        call_delta = abs(current_deltas.get("short_call_delta", 0.0))

        trigger = self.config.management_trigger_delta

        if put_delta >= trigger and call_delta >= trigger:
            return {
                "action": "close",
                "side": "both",
                "recommendation": (
                    "Both sides tested — close entire position to limit loss."
                ),
            }

        if put_delta >= trigger:
            return {
                "action": "roll",
                "side": "put",
                "recommendation": (
                    f"Put side tested (delta={put_delta:.2f}). "
                    "Roll put spread down and out to collect additional credit."
                ),
            }

        if call_delta >= trigger:
            return {
                "action": "roll",
                "side": "call",
                "recommendation": (
                    f"Call side tested (delta={call_delta:.2f}). "
                    "Roll call spread up and out to collect additional credit."
                ),
            }

        return None
