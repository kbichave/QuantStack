# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Equity & investment alert MCP tools — full lifecycle CRUD.

Entry alerts are created by the research loop (deep fundamental analysis).
Exit signals and price/regime updates are written by the trading loop.
The /get-alerts skill reads everything and manages status.

Tools:
  - create_equity_alert   — research loop creates entry alerts
  - get_equity_alerts      — read alerts with optional updates + exit signals
  - update_alert_status    — change status (pending→watching→acted→expired/skipped)
  - create_exit_signal     — trading loop writes exit recommendations
  - add_alert_update       — both loops write running commentary
"""

import json
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from quantstack.mcp._state import live_db_or_error
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain


# ── Valid enum values ────────────────────────────────────────────────────

_VALID_ACTIONS = ("buy", "sell")
_VALID_HORIZONS = ("investment", "swing", "position")
_VALID_STATUSES = ("pending", "watching", "acted", "expired", "skipped")
_VALID_URGENCIES = ("immediate", "today", "this_week")
_VALID_SIGNAL_TYPES = (
    "stop_loss_hit", "target_reached", "thesis_invalidated",
    "trailing_stop_hit", "time_stop", "regime_flip",
    "fundamental_deterioration", "earnings_miss", "insider_selling",
    "manual_close",
)
_VALID_SEVERITIES = ("info", "warning", "critical", "auto_close")
_VALID_UPDATE_TYPES = (
    "thesis_check", "price_update", "fundamental_update", "regime_change",
    "news_event", "earnings_report", "position_review", "user_note",
)
_VALID_THESIS_STATUSES = ("intact", "strengthening", "weakening", "broken")


# ── Helpers ──────────────────────────────────────────────────────────────

def _serialize_rows(rows: list, columns: list[str]) -> list[dict]:
    """Convert DB rows to list of dicts with stringified dates."""
    result = []
    for row in rows:
        d = dict(zip(columns, row))
        for k, v in d.items():
            if isinstance(v, (datetime,)):
                d[k] = v.isoformat()
            elif hasattr(v, "isoformat"):
                d[k] = str(v)
        result.append(d)
    return result


# ── Tools ────────────────────────────────────────────────────────────────

@domain(Domain.EXECUTION)
@mcp.tool()
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
) -> dict[str, Any]:
    """
    Create an equity/investment entry alert from the research loop.

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

    Returns:
        Dict with alert_id and creation status.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        if action not in _VALID_ACTIONS:
            return {"success": False, "error": f"Invalid action '{action}'. Must be: {_VALID_ACTIONS}"}
        if time_horizon not in _VALID_HORIZONS:
            return {"success": False, "error": f"Invalid time_horizon '{time_horizon}'. Must be: {_VALID_HORIZONS}"}
        if urgency not in _VALID_URGENCIES:
            urgency = "today"

        # Deduplicate: check for recent active alert on same symbol + horizon
        cutoff = datetime.now() - timedelta(days=7)
        existing = ctx.db.execute(
            """
            SELECT id, status, created_at FROM equity_alerts
            WHERE symbol = ? AND time_horizon = ? AND status IN ('pending', 'watching')
              AND created_at >= ?
            ORDER BY created_at DESC LIMIT 1
            """,
            [symbol, time_horizon, cutoff],
        ).fetchone()
        if existing:
            return {
                "success": True,
                "alert_id": existing[0],
                "deduplicated": True,
                "existing_status": existing[1],
                "message": f"Active alert #{existing[0]} already exists for {symbol} ({time_horizon})",
            }

        # Compute risk/reward ratio
        rr = 0.0
        if suggested_entry > 0 and stop_price > 0 and target_price > 0:
            risk = abs(suggested_entry - stop_price)
            reward = abs(target_price - suggested_entry)
            rr = round(reward / risk, 2) if risk > 0 else 0.0

        ctx.db.execute(
            """
            INSERT INTO equity_alerts (
                symbol, action, time_horizon, instrument_type, strategy_id, strategy_name,
                confidence, debate_verdict, debate_summary,
                current_price, suggested_entry, stop_price, target_price,
                trailing_stop_pct, risk_reward_ratio,
                regime, sector, catalyst, thesis, key_risks,
                piotroski_f_score, fcf_yield_pct, pe_ratio, analyst_consensus,
                status, urgency
            ) VALUES (
                ?, ?, ?, 'equity', ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                'pending', ?
            )
            """,
            [
                symbol, action, time_horizon, strategy_id, strategy_name,
                confidence, debate_verdict, debate_summary,
                current_price, suggested_entry, stop_price, target_price,
                trailing_stop_pct, rr,
                regime, sector, catalyst, thesis, key_risks,
                piotroski_f_score or None, fcf_yield_pct or None, pe_ratio or None, analyst_consensus,
                urgency,
            ],
        )

        # Get the inserted ID
        row = ctx.db.execute(
            "SELECT id FROM equity_alerts WHERE symbol = ? ORDER BY created_at DESC LIMIT 1",
            [symbol],
        ).fetchone()
        if not row:
            raise RuntimeError(f"Insert into equity_alerts silently failed for {symbol} — no row returned after write")
        alert_id = row[0]

        # Initial thesis check update
        snapshot = json.dumps({
            "price": current_price, "regime": regime,
            "f_score": piotroski_f_score, "fcf_yield": fcf_yield_pct,
            "pe": pe_ratio,
        })
        ctx.db.execute(
            """
            INSERT INTO alert_updates (alert_id, update_type, commentary, data_snapshot, thesis_status)
            VALUES (?, 'thesis_check', ?, ?, 'intact')
            """,
            [alert_id, f"Alert created. {thesis[:500]}", snapshot],
        )

        logger.info(f"[alerts] Created alert #{alert_id}: {action} {symbol} ({time_horizon})")
        return {
            "success": True,
            "alert_id": alert_id,
            "symbol": symbol,
            "action": action,
            "time_horizon": time_horizon,
            "deduplicated": False,
        }
    except Exception as e:
        logger.error(f"[alerts] create_equity_alert failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.EXECUTION)
@mcp.tool()
async def get_equity_alerts(
    symbol: str = "",
    status: str = "",
    time_horizon: str = "",
    alert_id: int = 0,
    include_updates: bool = False,
    include_exit_signals: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Retrieve equity alerts with optional update history and exit signals.

    Args:
        symbol: Filter by ticker. Empty = all.
        status: Filter by status (pending/watching/acted/expired/skipped). Empty = all.
        time_horizon: Filter by horizon (investment/swing/position). Empty = all.
        alert_id: Fetch a single alert by ID. Overrides other filters.
        include_updates: Include alert_updates timeline.
        include_exit_signals: Include alert_exit_signals.
        limit: Max alerts to return.

    Returns:
        Dict with alerts list and count.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        # Build query
        query = "SELECT * FROM equity_alerts"
        params: list[Any] = []
        conditions = []

        if alert_id > 0:
            conditions.append("id = ?")
            params.append(alert_id)
        else:
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol.upper())
            if status:
                conditions.append("status = ?")
                params.append(status)
            if time_horizon:
                conditions.append("time_horizon = ?")
                params.append(time_horizon)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"
        if limit > 0:
            query += f" LIMIT {limit}"

        rows = ctx.db.execute(query, params).fetchall()
        columns = [
            "id", "symbol", "action", "time_horizon", "instrument_type",
            "strategy_id", "strategy_name",
            "confidence", "debate_verdict", "debate_summary",
            "current_price", "suggested_entry", "stop_price", "target_price",
            "trailing_stop_pct", "risk_reward_ratio",
            "regime", "sector", "catalyst", "thesis", "key_risks",
            "piotroski_f_score", "fcf_yield_pct", "pe_ratio", "analyst_consensus",
            "status", "status_reason", "urgency",
            "created_at", "acted_at", "expired_at",
        ]
        alerts = _serialize_rows(rows, columns)

        # Optionally join updates and exit signals
        if include_updates and alerts:
            alert_ids = [a["id"] for a in alerts]
            placeholders = ",".join(["?"] * len(alert_ids))
            update_rows = ctx.db.execute(
                f"SELECT * FROM alert_updates WHERE alert_id IN ({placeholders}) ORDER BY created_at DESC",
                alert_ids,
            ).fetchall()
            update_cols = [
                "id", "alert_id", "update_type", "commentary",
                "data_snapshot", "thesis_status", "created_at",
            ]
            updates = _serialize_rows(update_rows, update_cols)
            # Group by alert_id
            updates_by_alert: dict[int, list] = {}
            for u in updates:
                updates_by_alert.setdefault(u["alert_id"], []).append(u)
            for a in alerts:
                a["updates"] = updates_by_alert.get(a["id"], [])

        if include_exit_signals and alerts:
            alert_ids = [a["id"] for a in alerts]
            placeholders = ",".join(["?"] * len(alert_ids))
            sig_rows = ctx.db.execute(
                f"SELECT * FROM alert_exit_signals WHERE alert_id IN ({placeholders}) ORDER BY created_at DESC",
                alert_ids,
            ).fetchall()
            sig_cols = [
                "id", "alert_id", "signal_type", "severity", "exit_price", "pnl_pct",
                "headline", "commentary", "what_changed", "lesson",
                "recommended_action", "recommended_reason",
                "acknowledged", "action_taken", "created_at",
            ]
            signals = _serialize_rows(sig_rows, sig_cols)
            signals_by_alert: dict[int, list] = {}
            for s in signals:
                signals_by_alert.setdefault(s["alert_id"], []).append(s)
            for a in alerts:
                a["exit_signals"] = signals_by_alert.get(a["id"], [])

        return {"success": True, "alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.error(f"[alerts] get_equity_alerts failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.EXECUTION)
@mcp.tool()
async def update_alert_status(
    alert_id: int,
    status: str,
    status_reason: str = "",
) -> dict[str, Any]:
    """
    Change the status of an equity alert.

    Status lifecycle: pending → watching → acted → expired | skipped

    Args:
        alert_id: Alert to update.
        status: New status (pending/watching/acted/expired/skipped).
        status_reason: Why the status changed (natural language).

    Returns:
        Confirmation with new status.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        if status not in _VALID_STATUSES:
            return {"success": False, "error": f"Invalid status '{status}'. Must be: {_VALID_STATUSES}"}

        # Set timestamp columns based on status
        extra_set = ""
        if status == "acted":
            extra_set = ", acted_at = CURRENT_TIMESTAMP"
        elif status in ("expired", "skipped"):
            extra_set = ", expired_at = CURRENT_TIMESTAMP"

        ctx.db.execute(
            f"UPDATE equity_alerts SET status = ?, status_reason = ?{extra_set} WHERE id = ?",
            [status, status_reason, alert_id],
        )

        logger.info(f"[alerts] Alert #{alert_id} → {status}: {status_reason}")
        return {"success": True, "alert_id": alert_id, "new_status": status, "reason": status_reason}
    except Exception as e:
        logger.error(f"[alerts] update_alert_status failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.EXECUTION)
@mcp.tool()
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
) -> dict[str, Any]:
    """
    Create an exit signal for an active equity alert.

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
        commentary: Detailed reasoning — why this happened, what changed.
        what_changed: Specific data point that triggered exit.
        lesson: What to learn for next time.
        recommended_action: hold, trim, close, tighten_stop, add.
        recommended_reason: Why this action.

    Returns:
        Confirmation with exit signal ID.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        if signal_type not in _VALID_SIGNAL_TYPES:
            return {"success": False, "error": f"Invalid signal_type '{signal_type}'. Must be: {_VALID_SIGNAL_TYPES}"}
        if severity not in _VALID_SEVERITIES:
            return {"success": False, "error": f"Invalid severity '{severity}'. Must be: {_VALID_SEVERITIES}"}

        ctx.db.execute(
            """
            INSERT INTO alert_exit_signals (
                alert_id, signal_type, severity, exit_price, pnl_pct,
                headline, commentary, what_changed, lesson,
                recommended_action, recommended_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                alert_id, signal_type, severity, exit_price or None, pnl_pct or None,
                headline, commentary, what_changed, lesson,
                recommended_action, recommended_reason,
            ],
        )

        # Get inserted ID
        row = ctx.db.execute(
            "SELECT id FROM alert_exit_signals WHERE alert_id = ? ORDER BY created_at DESC LIMIT 1",
            [alert_id],
        ).fetchone()
        if not row:
            logger.warning(f"[alerts] Could not retrieve exit signal ID for alert #{alert_id} after insert")
        signal_id = row[0] if row else 0

        # Auto-expire parent alert on auto_close severity
        if severity == "auto_close":
            ctx.db.execute(
                "UPDATE equity_alerts SET status = 'expired', status_reason = ?, expired_at = CURRENT_TIMESTAMP WHERE id = ?",
                [f"Auto-closed: {headline}", alert_id],
            )
            logger.info(f"[alerts] Alert #{alert_id} auto-closed: {headline}")

        logger.info(f"[alerts] Exit signal #{signal_id} on alert #{alert_id}: {signal_type} ({severity})")
        return {"success": True, "exit_signal_id": signal_id, "alert_id": alert_id, "auto_closed": severity == "auto_close"}
    except Exception as e:
        logger.error(f"[alerts] create_exit_signal failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.EXECUTION)
@mcp.tool()
async def add_alert_update(
    alert_id: int,
    update_type: str,
    commentary: str,
    data_snapshot: str = "",
    thesis_status: str = "intact",
) -> dict[str, Any]:
    """
    Add a running commentary update to an equity alert.

    Research loop writes: thesis_check, fundamental_update, earnings_report.
    Trading loop writes: price_update, regime_change.
    User writes: user_note.

    If thesis_status is "broken", automatically creates a critical exit signal.

    Args:
        alert_id: Parent alert ID.
        update_type: thesis_check, price_update, fundamental_update, regime_change,
                     news_event, earnings_report, position_review, user_note.
        commentary: Natural language — what happened, what it means, thesis impact.
        data_snapshot: JSON string of relevant metrics at time of update.
        thesis_status: intact, strengthening, weakening, broken.

    Returns:
        Confirmation with update ID.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        if update_type not in _VALID_UPDATE_TYPES:
            return {"success": False, "error": f"Invalid update_type '{update_type}'. Must be: {_VALID_UPDATE_TYPES}"}
        if thesis_status not in _VALID_THESIS_STATUSES:
            thesis_status = "intact"

        ctx.db.execute(
            """
            INSERT INTO alert_updates (alert_id, update_type, commentary, data_snapshot, thesis_status)
            VALUES (?, ?, ?, ?, ?)
            """,
            [alert_id, update_type, commentary, data_snapshot, thesis_status],
        )

        row = ctx.db.execute(
            "SELECT id FROM alert_updates WHERE alert_id = ? ORDER BY created_at DESC LIMIT 1",
            [alert_id],
        ).fetchone()
        if not row:
            logger.warning(f"[alerts] Could not retrieve update ID for alert #{alert_id} after insert")
        update_id = row[0] if row else 0

        # Auto-create exit signal if thesis is broken
        if thesis_status == "broken":
            ctx.db.execute(
                """
                INSERT INTO alert_exit_signals (
                    alert_id, signal_type, severity, headline, commentary,
                    recommended_action, recommended_reason
                ) VALUES (?, 'thesis_invalidated', 'critical', ?, ?, 'close', ?)
                """,
                [
                    alert_id,
                    f"Thesis broken: {commentary[:100]}",
                    commentary,
                    "Thesis invalidated — close position",
                ],
            )
            logger.info(f"[alerts] Thesis broken on alert #{alert_id} — auto-created exit signal")

        logger.info(f"[alerts] Update #{update_id} on alert #{alert_id}: {update_type} ({thesis_status})")
        return {"success": True, "update_id": update_id, "alert_id": alert_id, "thesis_status": thesis_status}
    except Exception as e:
        logger.error(f"[alerts] add_alert_update failed: {e}")
        return {"success": False, "error": str(e)}
