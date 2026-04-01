# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Phase 5 — Intraday execution enhancement tools.

Tools:
  - get_intraday_status   — intraday loop status, positions, P&L, bar count
  - get_tca_report        — aggregate TCA stats from TCAStore
  - get_algo_recommendation — urgency-aware algo selection via AlgoSelector
"""

from typing import Any

from loguru import logger

import quantstack.intraday.loop as _loop_mod
from quantstack.core.execution.algo_selector import select_algo
from quantstack.core.execution.tca_storage import TCAStore
from quantstack.intraday.loop import LiveIntradayLoop
from quantstack.mcp._state import _serialize
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.tools._registry import domain


# =============================================================================
# TOOL 1: get_intraday_status
# =============================================================================


@domain(Domain.SIGNALS)
@tool_def()
async def get_intraday_status() -> dict[str, Any]:
    """
    Return the current intraday loop status.

    Reports whether the loop is running, open positions, realized P&L,
    trades executed today, and bars processed.  Use in /review sessions
    to monitor intraday activity.

    Returns:
        Dict with keys: running, positions_held, realized_pnl, trades_today,
        bars_processed, flattened, symbols.
    """
    try:
        # The intraday loop stores its singleton state in the module-level
        # _active_loop variable when running.  If no loop is active, report
        # a dormant status.
        active_loop: LiveIntradayLoop | None = _get_active_loop()

        if active_loop is None:
            return {
                "success": True,
                "running": False,
                "note": "No active intraday loop. Start one with scripts/run_intraday.py.",
                "positions_held": 0,
                "realized_pnl": 0.0,
                "trades_today": 0,
                "bars_processed": 0,
                "flattened": False,
                "symbols": [],
            }

        # Extract status from the loop's internal components
        bar_count = active_loop._bar_count
        symbols = list(active_loop._symbols)

        # Try to read position manager state if the loop has been wired up
        positions_held = 0
        realized_pnl = 0.0
        trades_today = 0
        flattened = False

        try:
            # The position manager is wired during run() — may not exist
            # if the loop was created but not started.
            pm = _find_position_manager(active_loop)
            if pm is not None:
                positions_held = len(pm._position_meta)
                realized_pnl = pm.intraday_pnl
                trades_today = pm.trades_today
                flattened = pm.is_flattened
        except Exception as exc:
            logger.debug(f"[intraday] position manager state read failed: {exc}")

        return {
            "success": True,
            "running": True,
            "positions_held": positions_held,
            "realized_pnl": round(realized_pnl, 2),
            "trades_today": trades_today,
            "bars_processed": bar_count,
            "flattened": flattened,
            "symbols": symbols,
        }

    except Exception as exc:
        logger.error(f"[quantstack_mcp] get_intraday_status failed: {exc}")
        return {"success": False, "error": str(exc)}


# =============================================================================
# TOOL 2: get_tca_report
# =============================================================================


@domain(Domain.SIGNALS, Domain.PORTFOLIO)
@tool_def()
async def get_tca_report(
    lookback_days: int = 30,
    symbol: str | None = None,
) -> dict[str, Any]:
    """
    Return aggregate TCA (Transaction Cost Analysis) statistics.

    Queries the persistent TCA store for execution quality metrics over
    a lookback window.  Use in /reflect sessions to track slippage trends,
    identify worst fills, and assess algo recommendation accuracy.

    Args:
        lookback_days: Number of days to look back (default 30).
        symbol: Optional ticker symbol filter.  None returns all symbols.

    Returns:
        Dict with avg_slippage_bps, worst_fills, algo_breakdown,
        execution_quality verdict, and trade count.
    """
    try:
        with TCAStore() as store:
            stats = store.get_aggregate_stats(
                lookback_days=lookback_days,
                symbol=symbol,
            )
        return {"success": True, **stats}
    except Exception as exc:
        logger.error(f"[quantstack_mcp] get_tca_report failed: {exc}")
        return {"success": False, "error": str(exc)}


# =============================================================================
# TOOL 3: get_algo_recommendation
# =============================================================================


@domain(Domain.SIGNALS, Domain.EXECUTION)
@tool_def()
async def get_algo_recommendation(
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
) -> dict[str, Any]:
    """
    Get an urgency-aware execution algorithm recommendation.

    Wraps the TCA pre-trade forecast with override rules for special
    situations (stop-loss, high VIX, earnings, low liquidity).  Returns
    the recommended algo, limit price (if applicable), and cost estimate.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        side: "buy" or "sell".
        shares: Number of shares to trade.
        current_price: Current last-trade price.
        adv: Average daily volume in shares.
        daily_vol_pct: Daily return volatility in percent (e.g. 1.5).
        spread_bps: Current bid-ask spread in basis points (default 5.0).
        urgency: One of "stop_loss", "high", "normal", "low".
        vix: Current VIX level (0 if unknown).
        earnings_within_24h: True if earnings report is within 24 hours.
        bid: Current best bid price (optional, improves LIMIT pricing).
        ask: Current best ask price (optional, improves LIMIT pricing).

    Returns:
        Dict with recommended_algo, limit_price, urgency, expected costs,
        override_reason, execution_window, and TCA forecast details.
    """
    try:
        recommendation = select_algo(
            symbol=symbol.upper().strip(),
            side=side,
            shares=shares,
            current_price=current_price,
            adv=adv,
            daily_vol_pct=daily_vol_pct,
            spread_bps=spread_bps,
            urgency=urgency,
            vix=vix,
            earnings_within_24h=earnings_within_24h,
            bid=bid,
            ask=ask,
        )

        # Serialize the TCA forecast if present
        tca_summary: dict[str, Any] | None = None
        if recommendation.tca_forecast is not None:
            f = recommendation.tca_forecast
            tca_summary = {
                "spread_cost_bps": round(f.spread_cost_bps, 2),
                "market_impact_bps": round(f.market_impact_bps, 2),
                "timing_cost_bps": round(f.timing_cost_bps, 2),
                "commission_bps": round(f.commission_bps, 2),
                "total_expected_bps": round(f.total_expected_bps, 2),
                "participation_rate": round(f.participation_rate, 6),
                "is_liquid": f.is_liquid,
                "tca_recommended_algo": f.recommended_algo.value,
                "algo_rationale": f.algo_rationale,
                "min_alpha_bps": round(f.min_alpha_bps, 2),
            }

        return {
            "success": True,
            "symbol": symbol.upper().strip(),
            "recommended_algo": recommendation.recommended_algo,
            "limit_price": recommendation.limit_price,
            "urgency": recommendation.urgency,
            "expected_slippage_bps": recommendation.expected_slippage_bps,
            "expected_total_cost_bps": recommendation.expected_total_cost_bps,
            "override_reason": recommendation.override_reason,
            "execution_window": recommendation.execution_window,
            "tca_forecast": tca_summary,
        }

    except Exception as exc:
        logger.error(f"[quantstack_mcp] get_algo_recommendation({symbol}) failed: {exc}")
        return {"success": False, "symbol": symbol, "error": str(exc)}


# =============================================================================
# Helpers
# =============================================================================


def _get_active_loop() -> Any:
    """Retrieve the active LiveIntradayLoop singleton, if any.

    The run_intraday.py script sets ``quantstack.intraday.loop._active_loop``
    when a loop is started. Returns None if no loop is running.
    """
    return getattr(_loop_mod, "_active_loop", None)


def _find_position_manager(loop: Any) -> Any:
    """Extract the IntradayPositionManager from a running loop.

    The position manager is created inside run() and not stored as an
    instance attribute, so we check for it defensively.
    """
    # The position manager may have been stored as an attribute during run()
    if hasattr(loop, "_position_manager"):
        return loop._position_manager
    return None


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
