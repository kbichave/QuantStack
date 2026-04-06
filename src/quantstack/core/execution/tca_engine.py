"""
Transaction Cost Analysis (TCA) Engine.

Implements the Perold (1988) implementation shortfall framework:

  Implementation Shortfall = Paper Return − Live Return
                           = Timing cost + Market impact + Spread + Fees

Two modes:
  1. Pre-trade TCA  — forecast cost distribution before order submission,
                      select execution algorithm (TWAP / VWAP / immediate),
                      and determine whether expected alpha > expected cost.
  2. Post-trade TCA — measure actual fill vs. arrival price, VWAP, TWAP,
                      and compute per-trade and aggregate implementation
                      shortfall for ongoing execution quality monitoring.

Key benchmarks:
  - Arrival price  : price when the signal fired (theoretically correct for
                     signal-based systems — captures timing cost fully).
  - VWAP           : volume-weighted average price over the execution window.
  - TWAP           : time-weighted average price over the execution window.
  - Previous close : simple EOD benchmark (understates timing cost).

References:
  Perold, A. (1988). "The Implementation Shortfall: Paper vs. Reality."
  Journal of Portfolio Management.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.core.execution.almgren_chriss import almgren_chriss_expected_cost_bps
from quantstack.db import db_conn

# ---------------------------------------------------------------------------
# Default coefficients (Almgren et al. 2005 literature values)
# ---------------------------------------------------------------------------

DEFAULT_ETA: float = 0.142       # Temporary impact coefficient
DEFAULT_GAMMA: float = 0.314     # Permanent impact coefficient
DEFAULT_BETA: float = 0.60       # Impact exponent (3/5 power law, not square-root)

ADV_LARGE_CAP_THRESHOLD: float = 10_000_000  # $10M ADV separates large/small cap

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExecAlgo(str, Enum):
    """Execution algorithm recommendation."""

    IMMEDIATE = "IMMEDIATE"  # Market order now — low market impact, high timing cost
    TWAP = "TWAP"  # Time-weighted — uniform slicing across horizon
    VWAP = "VWAP"  # Volume-weighted — matches intraday volume curve
    POV = "POV"  # Percentage of volume — tracks live volume
    LIMIT = "LIMIT"  # Passive limit order — cheapest, uncertain fill


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


# ---------------------------------------------------------------------------
# Pre-trade TCA
# ---------------------------------------------------------------------------


@dataclass
class PreTradeForecast:
    """Pre-trade cost forecast and execution algorithm recommendation."""

    symbol: str
    side: OrderSide
    shares: float
    arrival_price: float  # Price at signal-fire time

    # Cost components (all in basis points)
    spread_cost_bps: float  # Half-spread (one-way)
    market_impact_bps: float  # Estimated market impact (square-root law)
    timing_cost_bps: float  # Volatility-driven execution uncertainty
    commission_bps: float  # Broker fee
    total_expected_bps: float  # Sum of all one-way cost components

    # Participation / liquidity
    participation_rate: float  # Order / estimated daily volume
    adv_fraction: float  # participation_rate alias (same value)
    is_liquid: bool  # True when participation < 1%

    # Algo recommendation
    recommended_algo: ExecAlgo
    algo_rationale: str

    # Break-even
    min_alpha_bps: float  # Alpha needed to overcome round-trip costs


def pre_trade_forecast(
    symbol: str,
    side: OrderSide,
    shares: float,
    arrival_price: float,
    adv: float,  # Average daily volume (shares)
    daily_volatility_pct: float,  # Daily return volatility in percent (e.g. 1.5)
    spread_bps: float = 5.0,  # Current bid-ask spread estimate in bps
    commission_per_share: float = 0.005,  # Broker commission $/share
) -> PreTradeForecast:
    """
    Forecast execution costs before submitting an order.

    Uses the Almgren-Chriss square-root market impact model:
        impact ∝ σ × √(order_size / ADV)

    Args:
        symbol: Ticker symbol.
        side: BUY or SELL.
        shares: Number of shares to trade.
        arrival_price: Last price when signal fired (arrival benchmark).
        adv: Average daily volume for the symbol.
        daily_volatility_pct: Daily return volatility (annualised / √252).
        spread_bps: Estimated bid-ask spread in basis points.
        commission_per_share: Broker commission per share in dollars.

    Returns:
        PreTradeForecast with cost breakdown and algo recommendation.
    """
    if adv <= 0 or arrival_price <= 0:
        logger.warning(
            f"[TCA] Invalid inputs for {symbol}: adv={adv}, price={arrival_price}"
        )
        adv = max(adv, 1)
        arrival_price = max(arrival_price, 0.01)

    participation_rate = shares / adv

    # --- Spread cost (half-spread, one-way) ---
    spread_cost = spread_bps / 2.0

    # --- Market impact (3/5 power law, Almgren et al. 2005) ---
    # Impact (bps) = η × participation_rate^β × σ_bps
    sigma_bps = daily_volatility_pct * 100.0  # vol in bps
    eta, _gamma, beta = _get_coefficients_for_adv(adv, arrival_price)
    market_impact_bps = eta * (participation_rate ** beta) * sigma_bps

    # --- Timing uncertainty cost ---
    # If we spread execution over time we face market moves; scale with participation
    timing_horizon = min(participation_rate * 0.5, 1.0)  # fraction of trading day
    timing_cost_bps = sigma_bps * timing_horizon * 0.3  # 30% of vol × horizon

    # --- Commission ---
    notional = shares * arrival_price
    commission_bps = (
        (commission_per_share * shares / notional) * 10000 if notional > 0 else 0
    )

    total_one_way = spread_cost + market_impact_bps + timing_cost_bps + commission_bps
    # Round-trip break-even alpha
    min_alpha_bps = total_one_way * 2.0

    is_liquid = participation_rate < 0.01  # Below 1% of ADV

    # --- Algorithm recommendation ---
    if participation_rate < 0.002:
        algo = ExecAlgo.IMMEDIATE
        rationale = (
            f"Order is {participation_rate * 100:.2f}% of ADV — negligible market impact. "
            "Immediate execution captures signal timing with minimal cost."
        )
    elif participation_rate < 0.01:
        algo = ExecAlgo.TWAP
        rationale = (
            f"Order is {participation_rate * 100:.2f}% of ADV. "
            "TWAP spreads impact uniformly; suitable for low-urgency signals."
        )
    elif participation_rate < 0.05:
        algo = ExecAlgo.VWAP
        rationale = (
            f"Order is {participation_rate * 100:.2f}% of ADV. "
            "VWAP tracks intraday volume profile to minimise market impact."
        )
    else:
        algo = ExecAlgo.POV
        rationale = (
            f"Order is {participation_rate * 100:.2f}% of ADV — large relative size. "
            "POV (10% participation cap) spreads execution across the day."
        )

    return PreTradeForecast(
        symbol=symbol,
        side=side,
        shares=shares,
        arrival_price=arrival_price,
        spread_cost_bps=spread_cost,
        market_impact_bps=market_impact_bps,
        timing_cost_bps=timing_cost_bps,
        commission_bps=commission_bps,
        total_expected_bps=total_one_way,
        participation_rate=participation_rate,
        adv_fraction=participation_rate,
        is_liquid=is_liquid,
        recommended_algo=algo,
        algo_rationale=rationale,
        min_alpha_bps=min_alpha_bps,
    )


# ---------------------------------------------------------------------------
# Post-trade TCA
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """Single trade record for post-trade TCA."""

    trade_id: str
    symbol: str
    side: OrderSide
    shares: float
    arrival_price: float  # Price at signal-fire (arrival benchmark)
    fill_price: float  # Actual average execution price
    vwap_price: float | None = None  # Interval VWAP if available
    twap_price: float | None = None  # Interval TWAP if available
    prev_close: float | None = None  # Previous day close
    timestamp: pd.Timestamp | None = None
    daily_volume: float | None = None  # ADV for AC benchmark
    daily_volatility: float | None = None  # Daily vol for AC benchmark
    bid_at_arrival: float | None = None  # Best bid when signal fired
    ask_at_arrival: float | None = None  # Best ask when signal fired
    algo_used: str | None = None  # Execution algorithm actually used
    symbol_adv_bucket: str | None = None  # ADV category


@dataclass
class TradeTCAResult:
    """Post-trade TCA for a single trade."""

    trade_id: str
    symbol: str
    side: OrderSide

    # Implementation shortfall vs. each benchmark (bps, positive = cost)
    shortfall_vs_arrival_bps: float
    shortfall_vs_vwap_bps: float | None
    shortfall_vs_twap_bps: float | None
    shortfall_vs_prev_close_bps: float | None

    # Dollar cost
    shortfall_dollar: float  # (fill_price - arrival_price) × shares × direction

    # Summary
    is_favorable: bool  # True if fill was better than arrival price

    # Almgren-Chriss benchmark
    ac_expected_cost_bps: float | None = None  # AC model expected cost for this order

    # Enhanced TCA fields
    spread_cost_bps: float | None = None  # Half-spread component
    time_of_day_effect_bps: float | None = None  # Deviation from window avg shortfall


def _shortfall_bps(
    fill: float,
    benchmark: float,
    side: OrderSide,
) -> float:
    """
    Compute implementation shortfall vs. a benchmark in basis points.

    Positive = execution was worse than benchmark (a cost).
    Negative = execution was better than benchmark (a saving / price improvement).
    """
    if benchmark <= 0:
        return 0.0
    if side == OrderSide.BUY:
        # For buys, paying more than benchmark = positive shortfall (cost)
        return (fill - benchmark) / benchmark * 10000
    else:
        # For sells, receiving less than benchmark = positive shortfall (cost)
        return (benchmark - fill) / benchmark * 10000


def post_trade_tca(
    record: TradeRecord,
    window_avg_shortfall_bps: float | None = None,
) -> TradeTCAResult:
    """
    Compute post-trade TCA for a single executed trade.

    Args:
        record: TradeRecord with fill and benchmark prices.
        window_avg_shortfall_bps: Average shortfall in the 30-min window (for time-of-day effect).

    Returns:
        TradeTCAResult with shortfall vs. each benchmark.
    """
    # Use prev_close as fallback when arrival_price is missing or zero
    effective_arrival = record.arrival_price
    if effective_arrival <= 0:
        if record.prev_close and record.prev_close > 0:
            effective_arrival = record.prev_close
        elif record.vwap_price and record.vwap_price > 0:
            effective_arrival = record.vwap_price
        else:
            effective_arrival = record.fill_price  # last resort

    vs_arrival = _shortfall_bps(record.fill_price, effective_arrival, record.side)
    vs_vwap = (
        _shortfall_bps(record.fill_price, record.vwap_price, record.side)
        if record.vwap_price
        else None
    )
    vs_twap = (
        _shortfall_bps(record.fill_price, record.twap_price, record.side)
        if record.twap_price
        else None
    )
    vs_close = (
        _shortfall_bps(record.fill_price, record.prev_close, record.side)
        if record.prev_close
        else None
    )

    direction = 1 if record.side == OrderSide.BUY else -1
    shortfall_dollar = (
        (record.fill_price - record.arrival_price) * record.shares * direction
    )

    # Almgren-Chriss expected cost benchmark (when volume/vol data available)
    ac_cost = None
    if record.daily_volume and record.daily_volatility and record.shares > 0:
        try:
            ac_cost = almgren_chriss_expected_cost_bps(
                order_shares=record.shares,
                daily_volume=record.daily_volume,
                daily_volatility=record.daily_volatility,
            )
        except Exception as exc:
            logger.debug("[TCA] Almgren-Chriss cost estimation failed: %s", exc)

    # Enhanced TCA: spread cost from bid-ask at arrival
    spread_cost = None
    if record.bid_at_arrival and record.ask_at_arrival and effective_arrival > 0:
        spread_cost = (record.ask_at_arrival - record.bid_at_arrival) / effective_arrival * 10000 / 2

    # Enhanced TCA: time-of-day effect
    tod_effect = None
    if window_avg_shortfall_bps is not None:
        tod_effect = vs_arrival - window_avg_shortfall_bps

    return TradeTCAResult(
        trade_id=record.trade_id,
        symbol=record.symbol,
        side=record.side,
        shortfall_vs_arrival_bps=vs_arrival,
        shortfall_vs_vwap_bps=vs_vwap,
        shortfall_vs_twap_bps=vs_twap,
        shortfall_vs_prev_close_bps=vs_close,
        ac_expected_cost_bps=ac_cost,
        shortfall_dollar=shortfall_dollar,
        is_favorable=vs_arrival <= 0,
        spread_cost_bps=spread_cost,
        time_of_day_effect_bps=tod_effect,
    )


# ---------------------------------------------------------------------------
# Classification helpers for TCA aggregation
# ---------------------------------------------------------------------------


def classify_adv_bucket(adv: float) -> str:
    """Classify a symbol's ADV into a bucket for TCA aggregation."""
    if adv < 500_000:
        return "low_adv"
    elif adv < 5_000_000:
        return "mid_adv"
    else:
        return "high_adv"


def classify_time_bucket(timestamp: pd.Timestamp) -> str:
    """Classify an execution timestamp into a 30-min bucket (ET)."""
    if timestamp.tzinfo is None:
        et = timestamp
    else:
        from zoneinfo import ZoneInfo
        et = timestamp.astimezone(ZoneInfo("America/New_York"))
    hour = et.hour
    minute = 0 if et.minute < 30 else 30
    start = f"{hour:02d}:{minute:02d}"
    end_minute = minute + 30
    end_hour = hour + (end_minute // 60)
    end_minute = end_minute % 60
    return f"{start}-{end_hour:02d}:{end_minute:02d}"


# ---------------------------------------------------------------------------
# TCA Engine — aggregate tracker
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coefficient loading from tca_coefficients table
# ---------------------------------------------------------------------------

# Module-level cache: {symbol_group: (eta, gamma, beta)}
_loaded_coefficients: dict[str, tuple[float, float, float]] = {}
_coefficients_loaded: bool = False


def _load_coefficients_from_db() -> dict[str, tuple[float, float, float]]:
    """Load calibrated coefficients from tca_coefficients table.

    Returns mapping of symbol_group -> (eta, gamma, beta).
    Falls back to empty dict on any failure.
    """
    try:
        with db_conn() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT ON (symbol_group)
                    symbol_group, eta, gamma, beta
                FROM tca_coefficients
                ORDER BY symbol_group, updated_at DESC
                """
            ).fetchall()
        result = {}
        for row in rows:
            group, eta, gamma, beta = row
            if eta is not None and gamma is not None and beta is not None:
                result[group] = (float(eta), float(gamma), float(beta))
        return result
    except Exception as exc:
        logger.debug("[TCA] Could not load coefficients from DB: %s", exc)
        return {}


def _ensure_coefficients_loaded() -> None:
    """Load coefficients once (lazy init)."""
    global _loaded_coefficients, _coefficients_loaded
    if not _coefficients_loaded:
        _loaded_coefficients = _load_coefficients_from_db()
        _coefficients_loaded = True


def _get_coefficients_for_adv(
    adv: float, price: float
) -> tuple[float, float, float]:
    """Return (eta, gamma, beta) for the given ADV level.

    Lookup order:
    1. symbol_group-specific ('large_cap' or 'small_cap')
    2. 'market_wide' fallback
    3. Module-level defaults
    """
    _ensure_coefficients_loaded()

    adv_dollars = adv * price if adv > 0 and price > 0 else 0
    group = "large_cap" if adv_dollars > ADV_LARGE_CAP_THRESHOLD else "small_cap"

    if group in _loaded_coefficients:
        return _loaded_coefficients[group]
    if "market_wide" in _loaded_coefficients:
        return _loaded_coefficients["market_wide"]
    return (DEFAULT_ETA, DEFAULT_GAMMA, DEFAULT_BETA)


class TCAEngine:
    """
    Persistent TCA tracker for a live trading session.

    Records arrival prices at signal-fire time and fill prices after execution,
    then computes aggregate implementation shortfall to measure ongoing
    execution quality.

    Usage::

        tca = TCAEngine()

        # When signal fires
        tca.record_arrival(trade_id="t1", symbol="AAPL", side=OrderSide.BUY,
                           shares=100, arrival_price=182.50)

        # Before submitting
        forecast = tca.pre_trade("t1", adv=85_000_000, daily_vol_pct=1.4)
        if forecast.total_expected_bps > expected_alpha_bps:
            skip order (alpha < cost)

        # After fill confirmed
        tca.record_fill(trade_id="t1", fill_price=182.63, vwap_price=182.55)

        # Aggregate report
        report = tca.aggregate_report()
    """

    def __init__(self) -> None:
        self._arrivals: dict[str, TradeRecord] = {}
        self._results: list[TradeTCAResult] = []
        self._forecasts: dict[str, PreTradeForecast] = {}
        # Trigger lazy coefficient loading
        _ensure_coefficients_loaded()

    def record_arrival(
        self,
        trade_id: str,
        symbol: str,
        side: OrderSide,
        shares: float,
        arrival_price: float,
        timestamp: pd.Timestamp | None = None,
    ) -> None:
        """
        Record the arrival price (price when the signal fired).

        Call this immediately when a trade decision is made, before any
        execution begins.
        """
        self._arrivals[trade_id] = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            shares=shares,
            arrival_price=arrival_price,
            fill_price=0.0,  # populated after fill
            timestamp=timestamp or pd.Timestamp.now(tz="America/New_York"),
        )
        logger.debug(
            f"[TCA] Arrival recorded: {symbol} {side.value} {shares}sh @ {arrival_price}"
        )

    def pre_trade(
        self,
        trade_id: str,
        adv: float,
        daily_vol_pct: float,
        spread_bps: float = 5.0,
        commission_per_share: float = 0.005,
    ) -> PreTradeForecast | None:
        """
        Run pre-trade cost forecast for a pending trade.

        Args:
            trade_id: Must match a previously recorded arrival.
            adv: Average daily volume.
            daily_vol_pct: Daily return volatility in percent.

        Returns:
            PreTradeForecast, or None if trade_id not found.
        """
        record = self._arrivals.get(trade_id)
        if not record:
            logger.warning(f"[TCA] pre_trade called for unknown trade_id={trade_id}")
            return None

        forecast = pre_trade_forecast(
            symbol=record.symbol,
            side=record.side,
            shares=record.shares,
            arrival_price=record.arrival_price,
            adv=adv,
            daily_volatility_pct=daily_vol_pct,
            spread_bps=spread_bps,
            commission_per_share=commission_per_share,
        )
        self._forecasts[trade_id] = forecast

        logger.info(
            f"[TCA] Pre-trade {record.symbol}: "
            f"expected={forecast.total_expected_bps:.1f}bps "
            f"(impact={forecast.market_impact_bps:.1f}, spread={forecast.spread_cost_bps:.1f}), "
            f"algo={forecast.recommended_algo.value}, "
            f"break-even alpha={forecast.min_alpha_bps:.1f}bps"
        )
        return forecast

    def record_fill(
        self,
        trade_id: str,
        fill_price: float,
        vwap_price: float | None = None,
        twap_price: float | None = None,
        prev_close: float | None = None,
    ) -> TradeTCAResult | None:
        """
        Record the actual fill price and compute post-trade TCA.

        Args:
            trade_id: Must match a previously recorded arrival.
            fill_price: Average execution price.
            vwap_price: Interval VWAP (optional).
            twap_price: Interval TWAP (optional).
            prev_close: Previous day close (optional).

        Returns:
            TradeTCAResult, or None if arrival not found.
        """
        record = self._arrivals.get(trade_id)
        if not record:
            logger.warning(f"[TCA] record_fill for unknown trade_id={trade_id}")
            return None

        record.fill_price = fill_price
        record.vwap_price = vwap_price
        record.twap_price = twap_price
        record.prev_close = prev_close

        result = post_trade_tca(record)
        self._results.append(result)

        favorable = "FAVORABLE" if result.is_favorable else "COST"
        logger.info(
            f"[TCA] Post-trade {record.symbol}: "
            f"IS vs arrival={result.shortfall_vs_arrival_bps:+.1f}bps ({favorable}), "
            f"dollar cost=${result.shortfall_dollar:+.2f}"
        )

        forecast = self._forecasts.get(trade_id)
        if forecast:
            over_under = result.shortfall_vs_arrival_bps - forecast.total_expected_bps
            logger.info(
                f"[TCA] Forecast accuracy: expected={forecast.total_expected_bps:.1f}bps, "
                f"actual={result.shortfall_vs_arrival_bps:.1f}bps, "
                f"delta={over_under:+.1f}bps"
            )

        return result

    def aggregate_report(self) -> dict:
        """
        Compute aggregate TCA statistics across all recorded trades.

        Returns:
            Dict with summary statistics for execution quality monitoring.
        """
        if not self._results:
            return {"n_trades": 0, "message": "No completed trades recorded"}

        arrivals = [r.shortfall_vs_arrival_bps for r in self._results]
        vwap_vals = [
            r.shortfall_vs_vwap_bps
            for r in self._results
            if r.shortfall_vs_vwap_bps is not None
        ]
        dollar_costs = [r.shortfall_dollar for r in self._results]
        favorable_count = sum(1 for r in self._results if r.is_favorable)

        report: dict = {
            "n_trades": len(self._results),
            "avg_shortfall_vs_arrival_bps": float(np.mean(arrivals)),
            "median_shortfall_vs_arrival_bps": float(np.median(arrivals)),
            "std_shortfall_vs_arrival_bps": (
                float(np.std(arrivals)) if len(arrivals) > 1 else 0.0
            ),
            "p95_shortfall_bps": float(np.percentile(arrivals, 95)),
            "total_dollar_cost": float(np.sum(dollar_costs)),
            "pct_favorable": favorable_count / len(self._results) * 100,
        }

        if vwap_vals:
            report["avg_shortfall_vs_vwap_bps"] = float(np.mean(vwap_vals))

        # Execution quality verdict
        avg_is = report["avg_shortfall_vs_arrival_bps"]
        if avg_is < 0:
            report["execution_quality"] = "EXCELLENT"
        elif avg_is < 5:
            report["execution_quality"] = "GOOD"
        elif avg_is < 15:
            report["execution_quality"] = "ACCEPTABLE"
        else:
            report["execution_quality"] = "POOR — investigate execution"

        return report

    def alpha_vs_cost_check(
        self,
        trade_id: str,
        expected_alpha_bps: float,
    ) -> tuple[bool, str]:
        """
        Check whether expected alpha clears the pre-trade cost forecast.

        Returns:
            (should_trade: bool, reason: str)
        """
        forecast = self._forecasts.get(trade_id)
        if not forecast:
            return True, "No forecast available — proceeding without TCA check"

        round_trip_cost = forecast.total_expected_bps * 2.0
        if expected_alpha_bps <= round_trip_cost:
            return (
                False,
                f"Alpha ({expected_alpha_bps:.1f}bps) ≤ round-trip cost ({round_trip_cost:.1f}bps). "
                "Trade not cost-effective.",
            )
        return (
            True,
            f"Alpha ({expected_alpha_bps:.1f}bps) > round-trip cost ({round_trip_cost:.1f}bps). "
            "Trade is cost-effective.",
        )
