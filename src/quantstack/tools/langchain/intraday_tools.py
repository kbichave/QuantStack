"""Intraday execution tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def get_intraday_status() -> str:
    """Return the current intraday loop status.

    Reports whether the loop is running, open positions, realized P&L,
    trades executed today, and bars processed. Use in /review sessions
    to monitor intraday activity.

    Returns JSON with running, positions_held, realized_pnl, trades_today,
    bars_processed, flattened, symbols.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_tca_report(
    lookback_days: int = 30,
    symbol: str | None = None,
) -> str:
    """Return aggregate TCA (Transaction Cost Analysis) statistics.

    Queries the persistent TCA store for execution quality metrics over
    a lookback window. Use in /reflect sessions to track slippage trends,
    identify worst fills, and assess algo recommendation accuracy.

    Args:
        lookback_days: Number of days to look back (default 30).
        symbol: Optional ticker symbol filter. None returns all symbols.

    Returns JSON with avg_slippage_bps, worst_fills, algo_breakdown,
    execution_quality verdict, and trade count.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
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
) -> str:
    """Get an urgency-aware execution algorithm recommendation.

    Wraps the TCA pre-trade forecast with override rules for special
    situations (stop-loss, high VIX, earnings, low liquidity). Returns
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

    Returns JSON with recommended_algo, limit_price, urgency, expected costs,
    override_reason, execution_window, and TCA forecast details.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
