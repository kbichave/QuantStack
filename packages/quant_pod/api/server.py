# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Local FastAPI server — single-user, no auth.

Exposes QuantPod functionality over HTTP for UI tooling and scripting.

Endpoints:
    GET  /health                          Health check + service status
    GET  /portfolio                       Current portfolio snapshot
    GET  /portfolio/positions             Current open positions
    POST /analyze/{symbol}                Run TradingDayFlow for a symbol
    GET  /trades                          Recent closed trades
    GET  /audit                           Recent audit log entries
    GET  /audit/{event_id}/trace          Full decision trace for an event
    GET  /audit/{event_id}/attribution    SHAP-style indicator attribution
    GET  /audit/session/{id}/summary      Session-level audit summary
    GET  /regime/{symbol}                 Current regime detection
    GET  /skills                          Agent skill performance
    GET  /calibration                     Agent confidence calibration report
    GET  /heartbeat                       Agent heartbeat monitoring (last seen)
    GET  /dashboard/pnl                   Daily realized + unrealized P&L by position
    GET  /dashboard/anomalies             Order size, win rate, tool failure anomalies
    GET  /etrade/status                   eTrade connection and auth status
    POST /etrade/auth                     eTrade OAuth flow (step 1: get URL, step 2: complete)
    POST /etrade/reconcile                Force portfolio sync from eTrade positions
    POST /kill                            Activate kill switch
    POST /reset                           Reset kill switch (ops use)

Run:
    uvicorn quant_pod.api.server:app --host 127.0.0.1 --port 8420 --reload
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from loguru import logger
from pydantic import BaseModel

from quant_pod.audit.decision_log import extract_indicator_attributions, get_decision_log
from quant_pod.audit.models import AuditQuery
from quant_pod.execution.broker_factory import get_broker, get_broker_mode
from quant_pod.execution.kill_switch import get_kill_switch
from quant_pod.execution.portfolio_state import get_portfolio_state
from quant_pod.learning.calibration import get_calibration_tracker
from quant_pod.monitoring.metrics import (
    get_metrics_content_type,
    get_metrics_text,
    record_daily_pnl,
    record_kill_switch_active,
    record_nav,
)

# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="QuantPod Local API",
    description="Single-user local API for QuantPod trading system",
    version="0.1.0",
)

# Allow localhost-only CORS for local UI development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================


class AnalyzeRequest(BaseModel):
    symbol: str
    date: str | None = None
    portfolio: dict[str, Any] | None = None
    regimes: dict[str, Any] | None = None


class ETradeAuthRequest(BaseModel):
    verifier_code: str | None = None  # None = step 1 (get URL), set = step 2 (complete)


class KillSwitchRequest(BaseModel):
    reason: str = "Manual trigger via API"


class ResetRequest(BaseModel):
    reset_by: str = "api_user"


# =============================================================================
# ROUTES
# =============================================================================


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check — returns service status and kill switch state."""
    kill = get_kill_switch()
    portfolio = get_portfolio_state()
    snapshot = portfolio.get_snapshot()

    return {
        "status": "ok",
        "broker_mode": get_broker_mode(),
        "kill_switch_active": kill.is_active(),
        "kill_switch_reason": kill.status().reason if kill.is_active() else None,
        "portfolio_equity": snapshot.total_equity,
        "open_positions": snapshot.position_count,
        "daily_pnl": snapshot.daily_pnl,
    }


@app.get("/portfolio")
def get_portfolio() -> dict[str, Any]:
    """Current portfolio snapshot."""
    portfolio = get_portfolio_state()
    snapshot = portfolio.get_snapshot()
    return snapshot.model_dump()


@app.get("/portfolio/positions")
def get_positions() -> list[dict[str, Any]]:
    """Current open positions."""
    portfolio = get_portfolio_state()
    return [p.model_dump() for p in portfolio.get_positions()]


@app.post("/analyze/{symbol}")
def analyze(symbol: str, req: AnalyzeRequest | None = None) -> dict[str, Any]:
    """
    Run regime detection + TradingDayFlow for a symbol.

    This kicks off the full agent stack (RegimeDetector → TradingCrew →
    RiskGate → PaperBroker) and returns the result.
    """
    kill = get_kill_switch()
    if kill.is_active():
        raise HTTPException(
            status_code=503,
            detail=f"Kill switch is active: {kill.status().reason}",
        )

    from quant_pod.agents.regime_detector import RegimeDetectorAgent
    from quant_pod.execution.portfolio_state import get_portfolio_state

    symbol = symbol.upper()
    trade_date = date.fromisoformat(req.date) if req and req.date else date.today()
    portfolio = (req.portfolio if req else None) or {
        "equity": get_portfolio_state().get_snapshot().total_equity,
        "context": get_portfolio_state().as_context_string(),
    }

    detector = RegimeDetectorAgent()
    regime_result = detector.detect_regime(symbol)

    if not regime_result.get("success"):
        logger.warning(f"[API] Regime detection failed for {symbol}: {regime_result.get('error')}")

    regime = {
        "trend": regime_result.get("trend_regime", "unknown"),
        "volatility": regime_result.get("volatility_regime", "unknown"),
        "confidence": regime_result.get("confidence", 0.0),
        "adx": regime_result.get("adx", 0.0),
        "atr_percentile": regime_result.get("atr_percentile", 50.0),
    }

    try:
        from quant_pod.flows.trading_day_flow import TradingDayFlow

        flow = TradingDayFlow()
        flow.state.symbols = [symbol]
        flow.state.current_date = trade_date
        flow.state.portfolio = portfolio
        flow.state.regimes = {symbol: regime}

        flow.kickoff()
        return {
            "symbol": symbol,
            "date": str(trade_date),
            "regime": regime,
            "trades_executed": len(flow.state.executed_trades),
            "decisions": flow.state.trade_decisions,
            "executed": flow.state.executed_trades,
            "errors": flow.state.errors,
            "session_id": flow._session_id,
        }
    except Exception as e:
        logger.error(f"[API] Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/trades")
def get_trades(limit: int = Query(default=50, le=500)) -> list[dict[str, Any]]:
    """Recent fills from the active broker (paper or eTrade)."""
    broker = get_broker()
    fills = broker.get_fills(limit=limit)
    return [f.model_dump() for f in fills]


@app.get("/audit")
def get_audit(
    symbol: str | None = None,
    agent_name: str | None = None,
    event_type: str | None = None,
    limit: int = Query(default=50, le=500),
) -> list[dict[str, Any]]:
    """Recent audit log entries."""
    log = get_decision_log()
    events = log.query(
        AuditQuery(
            symbol=symbol,
            agent_name=agent_name,
            event_type=event_type,
            limit=limit,
        )
    )
    return [
        {
            "event_id": e.event_id,
            "event_type": e.event_type,
            "agent_name": e.agent_name,
            "symbol": e.symbol,
            "action": e.action,
            "confidence": e.confidence,
            "output_summary": e.output_summary,
            "risk_approved": e.risk_approved,
            "created_at": e.created_at.isoformat(),
        }
        for e in events
    ]


@app.get("/audit/{event_id}/trace")
def get_audit_trace(event_id: str) -> list[dict[str, Any]]:
    """Full decision trace for a specific event."""
    log = get_decision_log()
    trace = log.get_decision_trace(event_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    return [e.model_dump() for e in trace]


@app.get("/audit/{event_id}/attribution")
def get_audit_attribution(event_id: str) -> dict[str, Any]:
    """
    SHAP-style indicator attribution for a specific decision event.

    Returns:
      - The stored indicator_attributions (if populated at recording time)
      - Derived attributions from the market_data_snapshot (as fallback)
      - IC dissent signals (ICs that disagreed with the consensus)
    """
    log = get_decision_log()
    log.query(AuditQuery(limit=1))  # Query by event_id directly
    # query() doesn't filter by event_id, so fetch via trace (single-element trace = the event itself)
    trace = log.get_decision_trace(event_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    event = trace[0]

    # Use stored attributions if present; derive from snapshot as fallback
    attributions = event.indicator_attributions
    if not attributions and event.market_data_snapshot:
        attributions = extract_indicator_attributions(
            event.market_data_snapshot, action=event.action
        )

    return {
        "event_id": event_id,
        "agent_name": event.agent_name,
        "symbol": event.symbol,
        "action": event.action,
        "confidence": event.confidence,
        "indicator_attributions": [a.model_dump() for a in attributions],
        "ic_dissent": event.ic_dissent,
        "derived": len(event.indicator_attributions) == 0,  # True if we derived vs. stored
    }


@app.get("/audit/session/{session_id}/summary")
def get_session_summary(session_id: str) -> dict[str, Any]:
    """High-level summary of a trading session."""
    log = get_decision_log()
    return log.get_session_summary(session_id)


@app.get("/regime/{symbol}")
def get_regime(symbol: str) -> dict[str, Any]:
    """Current regime detection for a symbol."""
    from quant_pod.agents.regime_detector import RegimeDetectorAgent

    detector = RegimeDetectorAgent()
    return detector.detect_regime(symbol.upper())


@app.get("/skills")
def get_skills() -> list[dict[str, Any]]:
    """
    Agent skill performance summary — includes IC/ICIR metrics.

    Returns per-agent: win-rate, prediction accuracy, IC mean, ICIR,
    rolling IC(30), IC trend, and retraining flag. Sorted by ICIR descending
    so the strongest and weakest signals appear at top and bottom.
    """
    from quant_pod.knowledge.store import KnowledgeStore
    from quant_pod.learning.skill_tracker import SkillTracker

    store = KnowledgeStore()
    tracker = SkillTracker(store)
    return tracker.ic_summary()


@app.get("/skills/degradation")
def get_skills_degradation() -> dict[str, Any]:
    """
    Alpha decay degradation report — on-demand monitoring check.

    Runs AlphaMonitor.check_all_agents() and returns a structured report
    of any agents showing IC decay, trend reversal, or retraining flags.

    Unlike /skills which shows raw metrics, this endpoint classifies each
    agent and provides human-readable alert messages and severity levels.

    Also fires Discord alerts if DISCORD_WEBHOOK_URL is configured.
    """
    from quant_pod.monitoring.alpha_monitor import get_alpha_monitor

    monitor = get_alpha_monitor()
    report = monitor.check_all_agents()

    return {
        "checked_at": report.checked_at.isoformat(),
        "n_agents_checked": report.n_agents_checked,
        "overall_status": report.overall_status,
        "alert_count": len(report.alerts),
        "alerts": [
            {
                "agent_id": a.agent_id,
                "severity": a.severity.value,
                "message": a.message,
                "rolling_ic_30": a.rolling_ic_30,
                "icir": a.icir,
                "ic_trend": a.ic_trend,
                "needs_retraining": a.needs_retraining,
                "detected_at": a.detected_at.isoformat(),
            }
            for a in report.alerts
        ],
        "all_agents": report.all_agents_ic_summary,
    }


@app.get("/orders")
def get_orders(
    status: str | None = None,
    limit: int = Query(default=50, le=500),
) -> list[dict[str, Any]]:
    """
    OMS order ledger — recent orders with lifecycle status.

    Optional ?status= filter accepts: new, submitted, acknowledged,
    partially_filled, filled, rejected, cancelled, expired.

    Includes implementation_shortfall_bps for filled orders — the
    primary metric for execution quality monitoring.
    """
    from quant_pod.execution.order_lifecycle import get_order_lifecycle

    oms = get_order_lifecycle()
    orders = oms.get_recent_orders(limit=limit)

    if status:
        orders = [o for o in orders if o.status.value == status.lower()]

    return [
        {
            "order_id": o.order_id,
            "symbol": o.symbol,
            "side": o.side,
            "quantity": o.quantity,
            "filled_quantity": o.filled_quantity,
            "status": o.status.value,
            "exec_algo": o.exec_algo.value,
            "arrival_price": o.arrival_price,
            "fill_price": o.fill_price,
            "implementation_shortfall_bps": o.implementation_shortfall_bps,
            "rejection_reason": o.rejection_reason,
            "created_at": o.created_at.isoformat(),
            "filled_at": o.filled_at.isoformat() if o.filled_at else None,
        }
        for o in orders
    ]


@app.get("/orders/summary")
def get_orders_summary() -> dict[str, Any]:
    """OMS session summary — fill rate, IS bps, status breakdown."""
    from quant_pod.execution.order_lifecycle import get_order_lifecycle

    return get_order_lifecycle().session_summary()


@app.get("/calibration")
def get_calibration() -> list[dict[str, Any]]:
    """Agent confidence calibration report."""
    tracker = get_calibration_tracker()
    return tracker.all_agents_summary()


@app.get("/heartbeat")
def get_heartbeat() -> dict[str, Any]:
    """
    Agent heartbeat monitor.

    Reports when each agent type last produced an audit event, and flags
    agents that have been silent longer than their expected cadence.

    Expected cadences:
      - ic_analysis:              should fire every session (≤ 25h)
      - super_trader_decision:    should fire every session (≤ 25h)
      - execution:                fires only when trades execute (no SLA)
      - risk_rejection:           fires only on rejections (no SLA)

    Status: "ok" | "stale" | "never_seen"
    """
    log = get_decision_log()
    now = datetime.now()
    max_silence_hours = 25  # One session = one trading day + buffer

    # Fetch the most recent event for each key event type
    monitored_types = ["ic_analysis", "pod_synthesis", "super_trader_decision"]
    agent_status: list[dict[str, Any]] = []

    for event_type in monitored_types:
        events = log.query(AuditQuery(event_type=event_type, limit=1))
        if not events:
            agent_status.append(
                {
                    "event_type": event_type,
                    "last_seen": None,
                    "hours_ago": None,
                    "status": "never_seen",
                }
            )
        else:
            last_event = events[0]
            age_hours = (now - last_event.created_at).total_seconds() / 3600
            status = "ok" if age_hours <= max_silence_hours else "stale"
            agent_status.append(
                {
                    "event_type": event_type,
                    "last_seen": last_event.created_at.isoformat(),
                    "hours_ago": round(age_hours, 1),
                    "status": status,
                    "last_agent": last_event.agent_name,
                    "last_symbol": last_event.symbol,
                }
            )

    overall = "ok"
    if any(s["status"] == "stale" for s in agent_status):
        overall = "stale"
    elif any(s["status"] == "never_seen" for s in agent_status):
        overall = "never_run"

    kill = get_kill_switch()
    return {
        "status": overall,
        "checked_at": now.isoformat(),
        "kill_switch_active": kill.is_active(),
        "agents": agent_status,
    }


@app.get("/dashboard/pnl")
def get_pnl_dashboard() -> dict[str, Any]:
    """
    P&L dashboard — daily realized and unrealized P&L by position.

    Returns:
      - today_realized_pnl: closed trades today
      - unrealized_by_symbol: mark-to-market per open position
      - total_unrealized_pnl: sum of all unrealized
      - total_equity: cash + positions_value
      - recent_trades: last 10 closed trades with P&L
    """
    portfolio = get_portfolio_state()
    snapshot = portfolio.get_snapshot()
    positions = portfolio.get_positions()

    unrealized_by_symbol = [
        {
            "symbol": p.symbol,
            "side": p.side,
            "quantity": p.quantity,
            "avg_cost": p.avg_cost,
            "current_price": p.current_price,
            "unrealized_pnl": p.unrealized_pnl,
            "unrealized_pnl_pct": (
                round(p.unrealized_pnl / (p.cost_basis or 1) * 100, 2) if p.cost_basis else 0.0
            ),
        }
        for p in positions
    ]

    broker = get_broker()
    recent_fills = broker.get_fills(limit=10)

    return {
        "as_of": datetime.now().isoformat(),
        "cash": snapshot.cash,
        "positions_value": snapshot.positions_value,
        "total_equity": snapshot.total_equity,
        "today_realized_pnl": snapshot.daily_pnl,
        "total_realized_pnl": snapshot.total_realized_pnl,
        "total_unrealized_pnl": sum(p.unrealized_pnl for p in positions),
        "position_count": snapshot.position_count,
        "largest_position_pct": round(snapshot.largest_position_pct * 100, 2),
        "unrealized_by_symbol": unrealized_by_symbol,
        "recent_fills": [f.model_dump() for f in recent_fills],
    }


@app.get("/dashboard/anomalies")
def get_anomalies() -> dict[str, Any]:
    """
    Anomaly detection across three dimensions:

    1. Order size anomalies: fills with quantity > mean + 3σ of historical fills
    2. Win rate degradation: rolling win rate < 52% over last 20 trades
    3. Tool failures: audit log events with failed tool calls in last 24h

    Each anomaly includes a severity ("warning" | "critical") and description.
    """
    anomalies: list[dict[str, Any]] = []
    broker = get_broker()
    log = get_decision_log()
    now = datetime.now()

    # ---- 1. Order size anomalies ----------------------------------------
    fills = broker.get_fills(limit=200)
    if len(fills) >= 10:
        quantities = [abs(f.filled_quantity) for f in fills]
        mean_qty = statistics.mean(quantities)
        stdev_qty = statistics.stdev(quantities) if len(quantities) > 1 else 0.0
        threshold = mean_qty + 3 * stdev_qty

        for fill in fills[-20:]:  # Check only recent fills
            if abs(fill.filled_quantity) > threshold and threshold > 0:
                anomalies.append(
                    {
                        "type": "order_size",
                        "severity": "warning",
                        "description": (
                            f"{fill.symbol} fill qty={fill.filled_quantity} exceeds "
                            f"mean+3σ ({threshold:.0f}). "
                            f"Historical mean={mean_qty:.0f}, σ={stdev_qty:.0f}."
                        ),
                        "symbol": fill.symbol,
                        "value": fill.filled_quantity,
                        "threshold": round(threshold, 2),
                        "detected_at": fill.filled_at.isoformat(),
                    }
                )

    # ---- 2. Win rate degradation (uses ClosedTrade records, not fills) -----
    portfolio = get_portfolio_state()
    recent_closed = portfolio.conn.execute(
        "SELECT realized_pnl FROM closed_trades ORDER BY closed_at DESC LIMIT 20"
    ).fetchall()
    if len(recent_closed) >= 5:
        pnls = [row[0] for row in recent_closed]
        winners = sum(1 for p in pnls if p > 0)
        win_rate = winners / len(pnls)
        if win_rate < 0.52:
            severity = "critical" if win_rate < 0.40 else "warning"
            anomalies.append(
                {
                    "type": "win_rate_degradation",
                    "severity": severity,
                    "description": (
                        f"Win rate over last {len(pnls)} trades: "
                        f"{win_rate:.1%} (below 52% threshold). "
                        f"Consider retraining or halting new entries."
                    ),
                    "value": round(win_rate, 4),
                    "threshold": 0.52,
                    "sample_size": len(pnls),
                    "detected_at": now.isoformat(),
                }
            )

    # ---- 3. Tool failures in last 24h ------------------------------------
    cutoff = now - timedelta(hours=24)
    recent_events = log.query(AuditQuery(limit=200, from_date=cutoff))
    failed_tool_events = []
    for event in recent_events:
        failed_tools = [tc for tc in event.tool_calls if not tc.success]
        if failed_tools:
            failed_tool_events.append(
                {
                    "event_id": event.event_id,
                    "agent_name": event.agent_name,
                    "symbol": event.symbol,
                    "failed_tools": [tc.tool_name for tc in failed_tools],
                    "at": event.created_at.isoformat(),
                }
            )

    if failed_tool_events:
        anomalies.append(
            {
                "type": "tool_failures",
                "severity": "warning" if len(failed_tool_events) < 5 else "critical",
                "description": (
                    f"{len(failed_tool_events)} agent events had tool failures in the last 24h. "
                    f"Agents may be operating on incomplete data."
                ),
                "count": len(failed_tool_events),
                "events": failed_tool_events[:10],  # Cap at 10 for response size
                "detected_at": now.isoformat(),
            }
        )

    return {
        "checked_at": now.isoformat(),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "status": "clean"
        if not anomalies
        else ("critical" if any(a["severity"] == "critical" for a in anomalies) else "warning"),
    }


@app.get("/etrade/status")
def etrade_status() -> dict[str, Any]:
    """
    eTrade connection status.

    Returns authentication state, account, sandbox mode, and whether
    USE_REAL_TRADING is active. Safe to call at any time — does not
    require eTrade credentials to be configured.
    """
    mode = get_broker_mode()
    if mode == "paper":
        return {
            "broker_mode": "paper",
            "etrade_active": False,
            "message": "PaperBroker active. Set USE_REAL_TRADING=true to use eTrade.",
        }

    try:
        from quant_pod.execution.etrade_broker import get_etrade_broker

        broker = get_etrade_broker()
        status = broker.auth_status()
        status["broker_mode"] = mode
        return status
    except Exception as e:
        return {
            "broker_mode": mode,
            "etrade_active": False,
            "error": str(e),
            "message": "EtradeBroker init failed — check credentials.",
        }


@app.post("/etrade/auth")
def etrade_auth(req: ETradeAuthRequest) -> dict[str, Any]:
    """
    eTrade OAuth flow.

    Step 1 — call with no body (or verifier_code=null):
        Returns auth_url. User visits it and gets a verifier code.

    Step 2 — call with verifier_code from the OAuth page:
        Completes authorization. Tokens saved to ~/.etrade_tokens.json.

    Only needed when USE_REAL_TRADING=true.
    """
    if get_broker_mode() == "paper":
        raise HTTPException(
            status_code=400,
            detail="USE_REAL_TRADING=false — eTrade auth not needed for PaperBroker.",
        )

    try:
        from quant_pod.execution.etrade_broker import get_etrade_broker

        broker = get_etrade_broker()

        if req.verifier_code is None:
            # Step 1: return OAuth URL
            auth_url = broker.get_auth_url()
            return {
                "step": 1,
                "auth_url": auth_url,
                "instructions": (
                    "Visit auth_url, authorize the app, then call this endpoint again "
                    "with the verifier_code from the redirect URL."
                ),
            }
        else:
            # Step 2: complete auth
            success = broker.complete_auth(req.verifier_code)
            if success:
                return {
                    "step": 2,
                    "authenticated": True,
                    "message": "eTrade authentication complete. Portfolio reconciled.",
                }
            else:
                raise HTTPException(
                    status_code=401, detail="eTrade auth failed — check verifier code."
                )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/etrade/reconcile")
def etrade_reconcile() -> dict[str, Any]:
    """
    Force a portfolio reconciliation against eTrade's actual positions.

    Use this if you suspect the local DuckDB state has drifted from eTrade
    (e.g., after a manual trade in the eTrade UI).

    Only available when USE_REAL_TRADING=true and authenticated.
    """
    if get_broker_mode() == "paper":
        raise HTTPException(
            status_code=400,
            detail="Reconciliation only available in eTrade mode.",
        )

    try:
        from quant_pod.execution.etrade_broker import get_etrade_broker

        broker = get_etrade_broker()

        if not broker._auth.is_authenticated():
            raise HTTPException(
                status_code=401, detail="Not authenticated — call /etrade/auth first."
            )

        broker._reconcile_on_startup()
        portfolio = get_portfolio_state()
        snapshot = portfolio.get_snapshot()

        return {
            "reconciled": True,
            "portfolio": snapshot.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/kill")
def trigger_kill_switch(req: KillSwitchRequest) -> dict[str, Any]:
    """Activate the kill switch — halts all trading immediately."""
    kill = get_kill_switch()
    kill.trigger(reason=req.reason)
    record_kill_switch_active(True)
    return {"status": "kill_switch_activated", "reason": req.reason}


@app.post("/reset")
def reset_kill_switch(req: ResetRequest) -> dict[str, Any]:
    """Reset the kill switch — allows trading to resume."""
    kill = get_kill_switch()
    if not kill.is_active():
        return {"status": "not_active", "message": "Kill switch was not active"}
    kill.reset(reset_by=req.reset_by)
    record_kill_switch_active(False)
    return {"status": "reset", "reset_by": req.reset_by}


@app.get("/monitor/intraday")
def monitor_intraday() -> dict[str, Any]:
    """
    Run an intraday monitoring cycle on demand.

    Equivalent to what runs on the hourly cron schedule. Checks:
      - Position P&L and proximity to daily loss limit
      - Regime reversals on all held symbols (vs. regime at entry)
      - Alpha decay status (IC-based)
      - IS/OOS degradation status (live Sharpe vs backtested benchmark)

    Returns structured report. Also fires Discord alert if DISCORD_WEBHOOK_URL
    is configured and action items are present.

    Safe to call at any time — read-only relative to broker state.
    """
    from quant_pod.flows.intraday_monitor_flow import IntradayMonitorFlow

    flow = IntradayMonitorFlow()
    report = flow.run()

    return {
        "status": report.overall_status,
        "run_at": report.run_at.isoformat(),
        "duration_seconds": report.duration_seconds,
        "intraday_pnl": report.intraday_pnl,
        "intraday_pnl_pct": report.intraday_pnl_pct,
        "daily_loss_pct": report.daily_loss_pct,
        "daily_loss_limit_pct": report.daily_loss_limit_pct,
        "open_positions": len(report.open_positions),
        "regime_reversals": [
            {
                "symbol": r.symbol,
                "entry_regime": r.entry_regime,
                "current_regime": r.current_regime,
                "position_side": r.position_side,
                "action_hint": r.action_hint,
            }
            for r in report.regime_reversals
        ],
        "alpha_status": report.alpha_status,
        "degradation_status": report.degradation_status,
        "action_items": report.action_items,
        "discord_alert_sent": report.discord_alert_sent,
    }


@app.get("/monitor/degradation")
def monitor_degradation(
    strategy_id: str | None = None,
    rolling_days: int = Query(default=60, ge=10, le=365),
) -> dict[str, Any]:
    """
    IS/OOS degradation report.

    Compares live rolling Sharpe (from closed trades in last `rolling_days`)
    against the registered in-sample backtested benchmark.

    Returns recommended_size_multiplier — the fraction to apply to position
    sizes. 1.0 = no change, 0.5 = halve sizes, 0.25 = quarter sizes.

    POST /monitor/degradation/benchmark to register an IS benchmark
    from your walk-forward validation results.
    """
    from quant_pod.monitoring.degradation_detector import get_degradation_detector

    detector = get_degradation_detector()
    if strategy_id:
        reports = [detector.check(strategy_id, rolling_days)]
    else:
        reports = detector.check_all(rolling_days)

    return {
        "rolling_days": rolling_days,
        "checked_at": datetime.now().isoformat(),
        "strategies": [
            {
                "strategy_id": r.strategy_id,
                "status": r.status.value,
                "live_sharpe": r.live_sharpe,
                "live_win_rate": r.live_win_rate,
                "live_max_drawdown": r.live_max_drawdown,
                "live_n_trades": r.live_n_trades,
                "sharpe_ratio_oos_vs_is": r.sharpe_ratio_oos_vs_is,
                "drawdown_ratio_vs_predicted": r.drawdown_ratio_vs_predicted,
                "recommended_size_multiplier": r.recommended_size_multiplier,
                "findings": r.findings,
                "has_benchmark": r.is_benchmark is not None,
            }
            for r in reports
        ],
        "overall_status": (
            "critical"
            if any(r.status.value == "critical" for r in reports)
            else "warning"
            if any(r.status.value == "warning" for r in reports)
            else "clean"
        ),
    }


class ISBenchmarkRequest(BaseModel):
    strategy_id: str = "default"
    predicted_annual_sharpe: float
    predicted_max_drawdown: float
    predicted_win_rate: float
    n_backtest_trades: int = 0


@app.post("/monitor/degradation/benchmark")
def register_benchmark(req: ISBenchmarkRequest) -> dict[str, Any]:
    """
    Register an in-sample (backtest) performance benchmark.

    Call this after running walk-forward validation so the degradation
    detector knows what live performance to compare against.

    Example:
        POST /monitor/degradation/benchmark
        {
          "strategy_id": "SuperTrader_SPY",
          "predicted_annual_sharpe": 1.8,
          "predicted_max_drawdown": 0.07,
          "predicted_win_rate": 0.57,
          "n_backtest_trades": 120
        }
    """
    from quant_pod.monitoring.degradation_detector import (
        ISBenchmark,
        get_degradation_detector,
    )

    detector = get_degradation_detector()
    benchmark = ISBenchmark(
        strategy_id=req.strategy_id,
        predicted_annual_sharpe=req.predicted_annual_sharpe,
        predicted_max_drawdown=req.predicted_max_drawdown,
        predicted_win_rate=req.predicted_win_rate,
        n_backtest_trades=req.n_backtest_trades,
    )
    detector.register_benchmark(benchmark)
    return {"registered": True, "strategy_id": req.strategy_id}


@app.get("/options/flow/{symbol}")
def options_flow(
    symbol: str,
    expiry_within_days: int = Query(default=45, ge=1, le=180),
) -> dict[str, Any]:
    """
    Unusual options activity (UOA) signal for a symbol.

    Fetches the current-day options chain via Alpha Vantage REALTIME_OPTIONS
    and derives:
      - flow_bias: BULLISH | BEARISH | NEUTRAL | MIXED
      - unusual_score: 0–100 (actionable above 60)
      - put_call_ratio: < 0.5 = heavy calls, > 2.0 = heavy puts
      - net_premium_usd: calls premium minus puts premium
      - n_unusual_contracts: contracts with vol/OI > 3× AND notional > $50k

    Requires ALPHA_VANTAGE_API_KEY. Returns NEUTRAL with score=0 if no data.
    """
    from quant_pod.tools.options_flow_tools import OptionsFlowClient

    client = OptionsFlowClient()
    signal = client.get_flow(symbol.upper(), expiry_within_days=expiry_within_days)

    return {
        "symbol": signal.symbol,
        "as_of": signal.as_of.isoformat(),
        "flow_bias": signal.flow_bias,
        "unusual_score": signal.unusual_score,
        "put_call_ratio": signal.put_call_ratio,
        "net_premium_usd": signal.net_premium_usd,
        "call_volume": signal.call_volume,
        "put_volume": signal.put_volume,
        "n_unusual_contracts": signal.n_unusual_contracts,
        "largest_trade": signal.largest_trade_description,
        "summary": signal.summary,
        "actionable": signal.unusual_score >= 60,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    """
    Prometheus metrics endpoint.

    Returns current metric values in text exposition format.
    Scrape with: scrape_configs: - job_name: quantpod, static_configs: [{targets: [host:8420]}]

    Also updates NAV and daily P&L gauges on each scrape so they stay current
    without requiring a dedicated background task.
    """
    portfolio = get_portfolio_state()
    snap = portfolio.get_snapshot()
    record_nav(snap.total_equity)
    record_daily_pnl(snap.daily_pnl)
    record_kill_switch_active(get_kill_switch().is_active())

    body = get_metrics_text()
    if not body:
        return "# prometheus_client not installed\n"
    return PlainTextResponse(content=body, media_type=get_metrics_content_type())
