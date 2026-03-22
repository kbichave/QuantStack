# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Algo selection engine — urgency-aware wrapper around TCA pre-trade forecast.

The TCA engine (tca_engine.py) provides Almgren-Chriss based algo selection.
This module adds override rules for special situations:
  - Stop-loss orders → always MARKET (urgency > slippage)
  - High VIX (>30) → always LIMIT (spreads are wide)
  - Earnings within 24h → LIMIT only (gaps possible)
  - Low liquidity (ADV < 500K) → LIMIT with wide buffer

Usage:
    from quantstack.core.execution.algo_selector import select_algo

    recommendation = select_algo(
        symbol="AAPL", side="buy", shares=100,
        current_price=185.50, adv=15_000_000,
        daily_vol_pct=1.4, spread_bps=2.0,
        urgency="normal", vix=18.5, earnings_within_24h=False,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from quantstack.core.execution.tca_engine import (
    ExecAlgo,
    OrderSide,
    PreTradeForecast,
    pre_trade_forecast,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Buffer factor controls how far inside the spread we place the limit price.
# Higher values = more aggressive (closer to far side), lower = more passive.
_BUFFER_FACTORS: dict[str, float] = {
    "stop_loss": 0.0,  # Not used — stop_loss always goes MARKET
    "high": 0.3,
    "normal": 0.5,
    "low": 0.8,
}

_EXECUTION_WINDOWS: dict[str, str] = {
    "stop_loss": "immediately",
    "high": "next 15 minutes",
    "normal": "10:00-12:00 ET",
    "low": "10:00-15:00 ET (patient)",
}

_VIX_HIGH_THRESHOLD = 30.0
_LOW_LIQUIDITY_ADV = 500_000


# ---------------------------------------------------------------------------
# AlgoRecommendation
# ---------------------------------------------------------------------------


@dataclass
class AlgoRecommendation:
    """Final algo selection with override context and limit price."""

    recommended_algo: str  # "MARKET", "LIMIT", "TWAP", "VWAP"
    limit_price: float | None  # Only set for LIMIT orders
    urgency: str  # "stop_loss", "high", "normal", "low"
    expected_slippage_bps: float
    expected_total_cost_bps: float
    override_reason: str | None  # Why override was applied; None if standard TCA
    notes: str  # Human-readable explanation of the selection for TCA audit logs
    execution_window: str  # Recommended time window, e.g. "10:00-12:00 ET"
    tca_forecast: PreTradeForecast | None  # Raw TCA output


# ---------------------------------------------------------------------------
# select_algo
# ---------------------------------------------------------------------------


def select_algo(
    symbol: str,
    side: str,
    shares: float,
    current_price: float,
    adv: float,
    daily_vol_pct: float,
    spread_bps: float = 5.0,
    urgency: str = "normal",
    vix: float = 0.0,
    earnings_within_24h: bool = False,
    bid: float | None = None,
    ask: float | None = None,
) -> AlgoRecommendation:
    """
    Select execution algorithm with urgency-aware overrides.

    1. Runs pre_trade_forecast() from tca_engine for the base recommendation.
    2. Applies override rules for special situations.
    3. Computes limit_price for LIMIT orders.
    4. Sets execution_window based on urgency.

    Args:
        symbol: Ticker symbol.
        side: "buy" or "sell" (case-insensitive).
        shares: Number of shares to trade.
        current_price: Current last-trade price.
        adv: Average daily volume in shares.
        daily_vol_pct: Daily return volatility in percent (e.g. 1.5).
        spread_bps: Current bid-ask spread estimate in basis points.
        urgency: One of "stop_loss", "high", "normal", "low".
        vix: Current VIX level (0 if unknown — no VIX override).
        earnings_within_24h: True if earnings report is within 24 hours.
        bid: Current best bid price (used for LIMIT price on sells).
        ask: Current best ask price (used for LIMIT price on buys).

    Returns:
        AlgoRecommendation with selected algo, limit price, and context.
    """
    urgency = urgency.lower().strip()
    if urgency not in _BUFFER_FACTORS:
        logger.warning(
            f"[AlgoSelector] Unknown urgency '{urgency}', defaulting to 'normal'"
        )
        urgency = "normal"

    order_side = OrderSide.BUY if side.lower().startswith("b") else OrderSide.SELL

    # ── 1. Base TCA forecast ────────────────────────────────────────────────
    tca_forecast: PreTradeForecast | None = None
    try:
        tca_forecast = pre_trade_forecast(
            symbol=symbol,
            side=order_side,
            shares=shares,
            arrival_price=current_price,
            adv=adv,
            daily_volatility_pct=daily_vol_pct,
            spread_bps=spread_bps,
        )
    except Exception as exc:
        logger.warning(f"[AlgoSelector] TCA forecast failed for {symbol}: {exc}")

    # Start with TCA's recommendation (converted to our string format)
    if tca_forecast:
        base_algo = _exec_algo_to_str(tca_forecast.recommended_algo)
        expected_slippage = (
            tca_forecast.market_impact_bps + tca_forecast.spread_cost_bps
        )
        expected_total_cost = tca_forecast.total_expected_bps
    else:
        base_algo = "LIMIT"  # Conservative default when TCA unavailable
        expected_slippage = spread_bps / 2.0
        expected_total_cost = spread_bps

    # ── 2. Override rules (priority order: stop_loss > vix > earnings > liquidity) ──
    algo = base_algo
    override_reason: str | None = None
    notes: str

    if urgency == "stop_loss":
        algo = "MARKET"
        override_reason = "stop_loss_urgency"
        notes = "Stop-loss: execution speed takes priority over slippage cost."
        logger.debug(f"[AlgoSelector] {symbol}: MARKET override — stop-loss urgency")

    elif vix > _VIX_HIGH_THRESHOLD:
        algo = "LIMIT"
        override_reason = "high_vix_wide_spreads"
        notes = f"VIX={vix:.1f} above {_VIX_HIGH_THRESHOLD}: spreads are wide, LIMIT reduces execution cost."
        logger.debug(
            f"[AlgoSelector] {symbol}: LIMIT override — VIX={vix:.1f} > {_VIX_HIGH_THRESHOLD}"
        )

    elif earnings_within_24h:
        algo = "LIMIT"
        override_reason = "earnings_gap_risk"
        notes = "Earnings within 24h: gap risk elevated, LIMIT protects against adverse opens."
        logger.debug(f"[AlgoSelector] {symbol}: LIMIT override — earnings within 24h")

    elif adv < _LOW_LIQUIDITY_ADV:
        algo = "LIMIT"
        override_reason = "low_liquidity"
        notes = f"ADV={adv:,.0f} below {_LOW_LIQUIDITY_ADV:,}: low liquidity, LIMIT avoids outsized market impact."
        logger.debug(
            f"[AlgoSelector] {symbol}: LIMIT override — ADV={adv:,.0f} < {_LOW_LIQUIDITY_ADV:,}"
        )

    else:
        pct_adv = shares / adv * 100 if adv > 0 else 0.0
        notes = (
            f"Standard TCA: {base_algo} recommended. "
            f"Order is {pct_adv:.2f}% of ADV, "
            f"spread={spread_bps:.1f}bps, urgency={urgency}."
        )

    # ── 3. Compute limit price for LIMIT orders ────────────────────────────
    limit_price: float | None = None
    if algo == "LIMIT":
        limit_price = _compute_limit_price(
            side=order_side,
            current_price=current_price,
            spread_bps=spread_bps,
            urgency=urgency,
            bid=bid,
            ask=ask,
        )

    # ── 4. Execution window ─────────────────────────────────────────────────
    execution_window = _EXECUTION_WINDOWS.get(urgency, "10:00-12:00 ET")

    recommendation = AlgoRecommendation(
        recommended_algo=algo,
        limit_price=limit_price,
        urgency=urgency,
        expected_slippage_bps=round(expected_slippage, 2),
        expected_total_cost_bps=round(expected_total_cost, 2),
        override_reason=override_reason,
        notes=notes,
        execution_window=execution_window,
        tca_forecast=tca_forecast,
    )

    logger.info(
        f"[AlgoSelector] {symbol} {side.upper()} {shares}sh: "
        f"algo={algo} limit={limit_price} urgency={urgency} "
        f"override={override_reason or 'none'} "
        f"cost={expected_total_cost:.1f}bps"
    )

    return recommendation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _exec_algo_to_str(algo: ExecAlgo) -> str:
    """Map TCA ExecAlgo enum to our string format."""
    mapping = {
        ExecAlgo.IMMEDIATE: "MARKET",
        ExecAlgo.LIMIT: "LIMIT",
        ExecAlgo.TWAP: "TWAP",
        ExecAlgo.VWAP: "VWAP",
        ExecAlgo.POV: "VWAP",  # POV maps to VWAP for simplicity
    }
    return mapping.get(algo, "LIMIT")


def _compute_limit_price(
    side: OrderSide,
    current_price: float,
    spread_bps: float,
    urgency: str,
    bid: float | None = None,
    ask: float | None = None,
) -> float:
    """Compute limit price based on side, spread, and urgency.

    For buys:  limit = ask - (spread_in_dollars * buffer_factor)
    For sells: limit = bid + (spread_in_dollars * buffer_factor)

    If bid/ask are not provided, we estimate them from current_price and spread_bps.
    """
    buffer_factor = _BUFFER_FACTORS.get(urgency, 0.5)
    spread_dollars = current_price * spread_bps / 10_000

    if side == OrderSide.BUY:
        reference = ask if ask is not None else current_price + spread_dollars / 2
        limit = reference - spread_dollars * buffer_factor
    else:
        reference = bid if bid is not None else current_price - spread_dollars / 2
        limit = reference + spread_dollars * buffer_factor

    return round(limit, 4)
