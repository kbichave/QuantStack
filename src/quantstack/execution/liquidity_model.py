# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Pre-trade liquidity model.

Estimates executable depth and spread cost for a symbol at a given time of
day, then gates orders that would exceed a safe participation fraction of
that depth.  Also provides a portfolio-level stressed-exit slippage estimate
used by the supervisor graph for risk monitoring.

Data hierarchy (highest to lowest fidelity):
  1. EWMA parameters from ``tca_parameters`` (section 06, real fill data)
  2. Bar-range proxy: ``(high - low) / midpoint * BAR_SPREAD_SCALE``
  3. Conservative defaults bucketed by ADV tier (large/mid/small cap)

Time buckets reuse the TCA EWMA definitions from
:func:`quantstack.execution.tca_ewma.resolve_time_bucket`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from quantstack.db import PgConnection
from quantstack.execution.tca_ewma import resolve_time_bucket


# =============================================================================
# Data models
# =============================================================================


class LiquidityVerdict(Enum):
    PASS = "pass"
    SCALE_DOWN = "scale_down"
    REJECT = "reject"


@dataclass
class LiquidityCheckResult:
    verdict: LiquidityVerdict
    reason: str
    recommended_quantity: int | None = None
    estimated_spread_bps: float = 0.0
    estimated_depth_shares: int = 0


@dataclass
class StressedExitResult:
    total_slippage_bps: float
    portfolio_value: float
    slippage_dollar_estimate: float
    alert: bool
    per_symbol_breakdown: dict = field(default_factory=dict)


# =============================================================================
# Time-of-day distribution weights
# =============================================================================

# Synthetic U-shaped intraday volume curve.  Weights represent the fraction
# of daily volume typically executed during each bucket.  Sums to ~1.0.
_BUCKET_VOLUME_WEIGHTS: dict[str, float] = {
    "morning": 0.30,   # 09:30-11:00 — heaviest volume
    "midday": 0.20,    # 11:00-14:00 — lunch lull
    "afternoon": 0.25, # 14:00-15:30 — picks up
    "close": 0.25,     # 15:30-16:00 — MOC imbalances
}


# =============================================================================
# LiquidityModel
# =============================================================================


class LiquidityModel:
    """Pre-trade liquidity estimator and gating layer.

    Args:
        conn: Optional DB connection for TCA EWMA lookups.  When *None*,
            the model falls back to bar-range proxy or conservative defaults.
        daily_volumes: Injected ADV map ``{symbol: shares_per_day}``.  Used
            in tests and when the caller already has the data in memory.
        bar_data: Injected recent bar data ``{symbol: {high, low, close}}``.
            Used for bar-range spread proxy when TCA EWMA data is unavailable.
    """

    # Time-of-day spread multipliers — wider spreads at open/close, tighter midday.
    TOD_MULTIPLIERS: dict[str, float] = {
        "open": 1.5,
        "morning": 1.2,
        "midday": 1.0,
        "afternoon": 1.1,
        "close": 1.3,
    }

    DEPTH_THRESHOLD: float = 0.10       # 10% of estimated bucket depth
    STRESS_THRESHOLD_BPS: float = 100.0  # portfolio-level slippage alert
    DEFAULT_SPREAD_BPS: float = 10.0     # large-cap fallback
    BAR_SPREAD_SCALE: float = 0.2        # bar-range → spread conversion factor

    # ADV-tier spread defaults (bps) when no EWMA or bar data available
    _TIER_DEFAULTS_BPS: dict[str, float] = {
        "large": 10.0,   # ADV >= 5M
        "mid": 25.0,     # ADV >= 500K
        "small": 50.0,   # ADV < 500K
    }

    _DEFAULT_ADV: int = 1_000_000  # fallback when ADV unknown

    def __init__(
        self,
        conn: PgConnection | None = None,
        daily_volumes: dict[str, int] | None = None,
        bar_data: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._conn = conn
        self._daily_volumes = daily_volumes or {}
        self._bar_data = bar_data or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adv_tier(self, adv: int) -> str:
        if adv >= 5_000_000:
            return "large"
        if adv >= 500_000:
            return "mid"
        return "small"

    def _get_adv(self, symbol: str) -> int:
        return self._daily_volumes.get(symbol, self._DEFAULT_ADV)

    def _lookup_ewma_spread(self, symbol: str, time_bucket: str) -> float | None:
        """Try to fetch EWMA spread from tca_parameters.  Returns None on miss."""
        if self._conn is None:
            return None
        try:
            from quantstack.execution.tca_ewma import get_ewma_forecast

            forecast = get_ewma_forecast(self._conn, symbol, time_bucket)
            if forecast is not None:
                return float(forecast["ewma_spread_bps"])
        except Exception as exc:
            logger.debug(f"[LIQUIDITY] EWMA lookup failed for {symbol}/{time_bucket}: {exc}")
        return None

    def _bar_range_spread(self, symbol: str) -> float | None:
        """Compute spread proxy from recent bar high/low/close."""
        bar = self._bar_data.get(symbol)
        if bar is None:
            return None
        high = bar.get("high", 0.0)
        low = bar.get("low", 0.0)
        close = bar.get("close", 0.0)
        if close <= 0 or high <= low:
            return None
        midpoint = (high + low) / 2.0
        range_bps = (high - low) / midpoint * 10_000
        return range_bps * self.BAR_SPREAD_SCALE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_spread(self, symbol: str, time_bucket: str | None = None) -> float:
        """Estimate bid-ask spread in basis points.

        Resolution order:
          1. EWMA from tca_parameters (if conn provided and data exists)
          2. Bar-range proxy (if bar_data injected)
          3. Conservative default based on ADV tier

        The result is scaled by time-of-day multiplier when *time_bucket* is
        provided.
        """
        bucket = time_bucket or "midday"
        tod_mult = self.TOD_MULTIPLIERS.get(bucket, 1.0)

        # 1. EWMA lookup
        ewma = self._lookup_ewma_spread(symbol, bucket)
        if ewma is not None:
            return ewma * tod_mult

        # 2. Bar-range proxy
        bar_spread = self._bar_range_spread(symbol)
        if bar_spread is not None:
            return bar_spread * tod_mult

        # 3. ADV-tier default
        adv = self._get_adv(symbol)
        tier = self._adv_tier(adv)
        return self._TIER_DEFAULTS_BPS[tier] * tod_mult

    def estimate_depth(self, symbol: str, time_bucket: str | None = None) -> int:
        """Estimate executable shares for a given time bucket.

        Uses daily volume divided by the bucket's weight in the synthetic
        U-curve.  Returns at least 1 share.
        """
        bucket = time_bucket or "midday"
        adv = self._get_adv(symbol)
        weight = _BUCKET_VOLUME_WEIGHTS.get(bucket, 0.25)
        return max(1, int(adv * weight))

    def pre_trade_check(
        self,
        symbol: str,
        order_quantity: int,
        current_time: datetime | None = None,
    ) -> LiquidityCheckResult:
        """Evaluate whether *order_quantity* is executable without excessive impact.

        Decision logic:
          - ratio = order_quantity / estimated_depth
          - ratio <= DEPTH_THRESHOLD (10%) --> PASS
          - ratio > DEPTH_THRESHOLD but <= 2 * DEPTH_THRESHOLD --> SCALE_DOWN
          - ratio > 2 * DEPTH_THRESHOLD --> REJECT

        Returns:
            :class:`LiquidityCheckResult` with verdict, reason, and optional
            recommended quantity.
        """
        bucket = resolve_time_bucket(current_time) if current_time else "midday"
        spread = self.estimate_spread(symbol, bucket)
        depth = self.estimate_depth(symbol, bucket)
        ratio = order_quantity / depth if depth > 0 else float("inf")

        if ratio <= self.DEPTH_THRESHOLD:
            return LiquidityCheckResult(
                verdict=LiquidityVerdict.PASS,
                reason=f"{symbol} order {order_quantity} is {ratio:.1%} of bucket depth {depth}",
                recommended_quantity=order_quantity,
                estimated_spread_bps=spread,
                estimated_depth_shares=depth,
            )

        if ratio <= 2 * self.DEPTH_THRESHOLD:
            recommended = int(depth * self.DEPTH_THRESHOLD)
            return LiquidityCheckResult(
                verdict=LiquidityVerdict.SCALE_DOWN,
                reason=(
                    f"{symbol} order {order_quantity} is {ratio:.1%} of bucket depth {depth} "
                    f"(>{self.DEPTH_THRESHOLD:.0%}); scaling to {recommended}"
                ),
                recommended_quantity=recommended,
                estimated_spread_bps=spread,
                estimated_depth_shares=depth,
            )

        return LiquidityCheckResult(
            verdict=LiquidityVerdict.REJECT,
            reason=(
                f"{symbol} order {order_quantity} is {ratio:.1%} of bucket depth {depth} "
                f"(>{2 * self.DEPTH_THRESHOLD:.0%}); too illiquid"
            ),
            recommended_quantity=None,
            estimated_spread_bps=spread,
            estimated_depth_shares=depth,
        )

    def stressed_exit_slippage(
        self,
        positions: list[dict[str, Any]],
        current_time: datetime | None = None,
    ) -> StressedExitResult:
        """Estimate portfolio-level slippage if all positions exited now.

        Each position dict must contain ``symbol``, ``quantity``, and ``price``.
        Slippage per symbol is the spread estimate (in bps) scaled by a
        participation multiplier: ``sqrt(quantity / depth)`` (square-root
        market impact model).

        Args:
            positions: List of position dicts with keys: symbol, quantity, price.
            current_time: Optional timestamp for time-bucket resolution.

        Returns:
            :class:`StressedExitResult` with portfolio-level and per-symbol data.
        """
        import math

        bucket = resolve_time_bucket(current_time) if current_time else "midday"
        portfolio_value = 0.0
        total_slippage_dollars = 0.0
        breakdown: dict[str, float] = {}

        for pos in positions:
            symbol = pos["symbol"]
            qty = abs(pos["quantity"])
            price = pos["price"]
            notional = qty * price
            portfolio_value += notional

            spread = self.estimate_spread(symbol, bucket)
            depth = self.estimate_depth(symbol, bucket)

            # Square-root impact: participation-adjusted slippage
            participation = qty / depth if depth > 0 else 1.0
            impact_mult = math.sqrt(max(participation, 1e-9))
            symbol_slippage_bps = spread * impact_mult

            slippage_dollars = notional * symbol_slippage_bps / 10_000
            total_slippage_dollars += slippage_dollars
            breakdown[symbol] = round(symbol_slippage_bps, 2)

        total_slippage_bps = (
            (total_slippage_dollars / portfolio_value * 10_000)
            if portfolio_value > 0
            else 0.0
        )

        return StressedExitResult(
            total_slippage_bps=round(total_slippage_bps, 2),
            portfolio_value=round(portfolio_value, 2),
            slippage_dollar_estimate=round(total_slippage_dollars, 2),
            alert=total_slippage_bps > self.STRESS_THRESHOLD_BPS,
            per_symbol_breakdown=breakdown,
        )
