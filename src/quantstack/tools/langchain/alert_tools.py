"""Equity alert lifecycle tools for LangGraph agents."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
async def create_equity_alert(
    symbol: Annotated[str, Field(description="Ticker symbol for the equity alert, e.g. 'AAPL', 'TSLA'")],
    action: Annotated[str, Field(description="Trade direction: 'buy' or 'sell'")],
    time_horizon: Annotated[str, Field(description="Holding period category: 'investment', 'swing', or 'position'")],
    thesis: Annotated[str, Field(description="Full investment thesis explaining the rationale in natural language")],
    strategy_id: Annotated[str, Field(description="ID of the strategy that generated this alert")] = "",
    strategy_name: Annotated[str, Field(description="Human-readable strategy name for display")] = "",
    confidence: Annotated[float, Field(description="Conviction score from 0.0 to 1.0")] = 0.0,
    debate_verdict: Annotated[str, Field(description="Trade-debater verdict: 'ENTER' or 'SKIP'")] = "",
    debate_summary: Annotated[str, Field(description="Bull/bear/risk summary from the trade debate")] = "",
    current_price: Annotated[float, Field(description="Market price at the time of alert creation")] = 0.0,
    suggested_entry: Annotated[float, Field(description="Suggested limit entry price or current market price")] = 0.0,
    stop_price: Annotated[float, Field(description="Initial stop-loss price level")] = 0.0,
    target_price: Annotated[float, Field(description="Take-profit target price level")] = 0.0,
    trailing_stop_pct: Annotated[float, Field(description="Trailing stop percentage, e.g. 15.0 for 15%")] = 0.0,
    regime: Annotated[str, Field(description="Market regime at alert creation: trending_up, trending_down, ranging, unknown")] = "unknown",
    sector: Annotated[str, Field(description="Equity sector or industry classification")] = "",
    catalyst: Annotated[str, Field(description="Event or catalyst that triggered this alert")] = "",
    key_risks: Annotated[str, Field(description="Key risk factors described in natural language")] = "",
    piotroski_f_score: Annotated[int, Field(description="Piotroski F-score for fundamental quality (0-9)")] = 0,
    fcf_yield_pct: Annotated[float, Field(description="Free cash flow yield as a percentage")] = 0.0,
    pe_ratio: Annotated[float, Field(description="Price-to-earnings ratio for valuation context")] = 0.0,
    analyst_consensus: Annotated[str, Field(description="Wall Street analyst consensus: 'buy', 'hold', or 'sell'")] = "",
    urgency: Annotated[str, Field(description="Execution urgency: 'immediate', 'today', or 'this_week'")] = "today",
) -> str:
    """Create an equity or investment entry alert from the research loop with full thesis and risk parameters. Use when the research graph identifies a new trade opportunity and needs to persist it as a pending alert for the trading loop. Returns JSON with alert_id and creation status. Deduplicates against recent alerts for the same symbol and time horizon within 7 days. Synonyms: trade idea, entry signal, watchlist add, opportunity alert, buy signal, sell signal, position alert."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_equity_alerts(
    symbol: Annotated[str, Field(description="Filter by ticker symbol, e.g. 'AAPL'. Empty string returns all symbols")] = "",
    status: Annotated[str, Field(description="Filter by alert status: 'pending', 'watching', 'acted', 'expired', or 'skipped'. Empty returns all")] = "",
    time_horizon: Annotated[str, Field(description="Filter by holding horizon: 'investment', 'swing', or 'position'. Empty returns all")] = "",
    alert_id: Annotated[int, Field(description="Fetch a single alert by its numeric ID. Overrides all other filters when non-zero")] = 0,
    include_updates: Annotated[bool, Field(description="Whether to include the alert_updates commentary timeline")] = False,
    include_exit_signals: Annotated[bool, Field(description="Whether to include associated alert_exit_signals")] = False,
    limit: Annotated[int, Field(description="Maximum number of alerts to return in the response")] = 20,
) -> str:
    """Retrieve equity alerts with optional update history and exit signal details from the database. Use when reviewing pending trade ideas, checking alert status, or auditing the alert pipeline. Returns JSON with alerts list, count, and optional update/exit signal timelines. Provides filtering by symbol, status, and time horizon. Synonyms: list alerts, query watchlist, fetch trade ideas, alert history, pending signals."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def update_alert_status(
    alert_id: Annotated[int, Field(description="Numeric ID of the alert to update")],
    status: Annotated[str, Field(description="New alert status: 'pending', 'watching', 'acted', 'expired', or 'skipped'")],
    status_reason: Annotated[str, Field(description="Explanation of why the status changed, in natural language")] = "",
) -> str:
    """Change the lifecycle status of an equity alert following the pending-watching-acted-expired/skipped workflow. Use when transitioning an alert through its lifecycle stages after review or execution. Returns JSON with confirmation and the new status value. Synonyms: update alert, change status, mark acted, expire alert, skip alert, alert lifecycle transition."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def create_exit_signal(
    alert_id: Annotated[int, Field(description="Numeric ID of the parent alert to attach the exit signal to")],
    signal_type: Annotated[str, Field(description="Exit trigger type: 'stop_loss_hit', 'target_reached', 'thesis_invalidated', 'trailing_stop_hit', 'time_stop', 'regime_flip', 'fundamental_deterioration', 'earnings_miss', 'insider_selling', 'manual_close'")],
    severity: Annotated[str, Field(description="Signal severity level: 'info', 'warning', 'critical', or 'auto_close' (auto-expires parent alert)")],
    headline: Annotated[str, Field(description="One-line exit summary, e.g. 'AAPL stop hit at $172 (-8.2%)'")],
    exit_price: Annotated[float, Field(description="Market price at the time of the exit signal")] = 0.0,
    pnl_pct: Annotated[float, Field(description="Unrealized profit-and-loss percentage at signal time")] = 0.0,
    commentary: Annotated[str, Field(description="Detailed reasoning behind the exit signal")] = "",
    what_changed: Annotated[str, Field(description="Specific data point or event that triggered the exit")] = "",
    lesson: Annotated[str, Field(description="Lesson learned for future trade improvement")] = "",
    recommended_action: Annotated[str, Field(description="Suggested next action: 'hold', 'trim', 'close', 'tighten_stop', or 'add'")] = "hold",
    recommended_reason: Annotated[str, Field(description="Explanation of why the recommended action is appropriate")] = "",
) -> str:
    """Create an exit signal for an active equity alert when price, regime, or fundamental conditions trigger a close. Use when the trading loop detects a stop loss hit, target reached, thesis invalidation, or regime flip that warrants exiting a position. Returns JSON with exit signal ID confirmation. Auto-closes the parent alert when severity is 'auto_close'. Synonyms: sell signal, exit trigger, stop loss, take profit, close position, risk event, position exit."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def add_alert_update(
    alert_id: Annotated[int, Field(description="Numeric ID of the parent alert to attach the update to")],
    update_type: Annotated[str, Field(description="Update category: 'thesis_check', 'price_update', 'fundamental_update', 'regime_change', 'news_event', 'earnings_report', 'position_review', or 'user_note'")],
    commentary: Annotated[str, Field(description="Natural language description of what happened and impact on the thesis")],
    data_snapshot: Annotated[str, Field(description="JSON string of relevant metrics captured at the time of the update")] = "",
    thesis_status: Annotated[str, Field(description="Current thesis health: 'intact', 'strengthening', 'weakening', or 'broken' (auto-creates critical exit signal)")] = "intact",
) -> str:
    """Add a running commentary update to an equity alert to track thesis evolution and price developments over time. Use when the research or trading loop needs to log a thesis check, price movement, fundamental change, regime shift, or earnings event against an existing alert. Returns JSON with update ID confirmation. Automatically triggers a critical exit signal when thesis_status is 'broken'. Synonyms: alert note, thesis update, position commentary, alert journal, trade diary entry, status update."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
