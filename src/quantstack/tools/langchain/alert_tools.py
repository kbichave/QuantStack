"""Equity alert lifecycle tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def create_equity_alert(
    symbol: str,
    action: str,
    time_horizon: str,
    thesis: str,
    strategy_id: str = "",
    strategy_name: str = "",
    confidence: float = 0.0,
    debate_verdict: str = "",
    debate_summary: str = "",
    current_price: float = 0.0,
    suggested_entry: float = 0.0,
    stop_price: float = 0.0,
    target_price: float = 0.0,
    trailing_stop_pct: float = 0.0,
    regime: str = "unknown",
    sector: str = "",
    catalyst: str = "",
    key_risks: str = "",
    piotroski_f_score: int = 0,
    fcf_yield_pct: float = 0.0,
    pe_ratio: float = 0.0,
    analyst_consensus: str = "",
    urgency: str = "today",
) -> str:
    """Create an equity/investment entry alert from the research loop.

    Deduplicates: if a recent alert exists for the same symbol + time_horizon
    with status pending/watching (within 7 days), returns the existing alert
    instead of creating a duplicate.

    Args:
        symbol: Ticker symbol.
        action: "buy" or "sell".
        time_horizon: "investment", "swing", or "position".
        thesis: Full investment thesis in natural language.
        strategy_id: Strategy that generated this alert.
        strategy_name: Human-readable strategy name.
        confidence: 0-1 conviction score.
        debate_verdict: ENTER/SKIP from trade-debater.
        debate_summary: Bull/bear/risk summary.
        current_price: Price when alert was created.
        suggested_entry: Limit price or current price.
        stop_price: Initial stop loss level.
        target_price: Take-profit target.
        trailing_stop_pct: Trailing stop as % (e.g. 15.0).
        regime: Market regime at alert creation.
        sector: Sector/industry.
        catalyst: What triggered this alert.
        key_risks: What could go wrong (natural language).
        piotroski_f_score: Fundamental quality score (investment alerts).
        fcf_yield_pct: Free cash flow yield %.
        pe_ratio: Price-to-earnings ratio.
        analyst_consensus: buy/hold/sell.
        urgency: "immediate", "today", or "this_week".

    Returns JSON with alert_id and creation status.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_equity_alerts(
    symbol: str = "",
    status: str = "",
    time_horizon: str = "",
    alert_id: int = 0,
    include_updates: bool = False,
    include_exit_signals: bool = False,
    limit: int = 20,
) -> str:
    """Retrieve equity alerts with optional update history and exit signals.

    Args:
        symbol: Filter by ticker. Empty = all.
        status: Filter by status (pending/watching/acted/expired/skipped). Empty = all.
        time_horizon: Filter by horizon (investment/swing/position). Empty = all.
        alert_id: Fetch a single alert by ID. Overrides other filters.
        include_updates: Include alert_updates timeline.
        include_exit_signals: Include alert_exit_signals.
        limit: Max alerts to return.

    Returns JSON with alerts list and count.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def update_alert_status(
    alert_id: int,
    status: str,
    status_reason: str = "",
) -> str:
    """Change the status of an equity alert.

    Status lifecycle: pending -> watching -> acted -> expired | skipped

    Args:
        alert_id: Alert to update.
        status: New status (pending/watching/acted/expired/skipped).
        status_reason: Why the status changed (natural language).

    Returns JSON with confirmation and new status.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def create_exit_signal(
    alert_id: int,
    signal_type: str,
    severity: str,
    headline: str,
    exit_price: float = 0.0,
    pnl_pct: float = 0.0,
    commentary: str = "",
    what_changed: str = "",
    lesson: str = "",
    recommended_action: str = "hold",
    recommended_reason: str = "",
) -> str:
    """Create an exit signal for an active equity alert.

    Called by the trading loop when price/regime conditions trigger an exit.
    If severity is "auto_close", the parent alert is automatically expired.

    Args:
        alert_id: Parent alert ID.
        signal_type: stop_loss_hit, target_reached, thesis_invalidated, trailing_stop_hit,
                     time_stop, regime_flip, fundamental_deterioration, earnings_miss,
                     insider_selling, manual_close.
        severity: info, warning, critical, auto_close.
        headline: One-line summary (e.g. "AAPL stop hit at $172 (-8.2%)").
        exit_price: Price at signal time.
        pnl_pct: Unrealized P&L % at signal time.
        commentary: Detailed reasoning.
        what_changed: Specific data point that triggered exit.
        lesson: What to learn for next time.
        recommended_action: hold, trim, close, tighten_stop, add.
        recommended_reason: Why this action.

    Returns JSON with exit signal ID confirmation.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def add_alert_update(
    alert_id: int,
    update_type: str,
    commentary: str,
    data_snapshot: str = "",
    thesis_status: str = "intact",
) -> str:
    """Add a running commentary update to an equity alert.

    Research loop writes: thesis_check, fundamental_update, earnings_report.
    Trading loop writes: price_update, regime_change.
    If thesis_status is "broken", automatically creates a critical exit signal.

    Args:
        alert_id: Parent alert ID.
        update_type: thesis_check, price_update, fundamental_update, regime_change,
                     news_event, earnings_report, position_review, user_note.
        commentary: Natural language description of what happened and thesis impact.
        data_snapshot: JSON string of relevant metrics at time of update.
        thesis_status: intact, strengthening, weakening, broken.

    Returns JSON with update ID confirmation.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
