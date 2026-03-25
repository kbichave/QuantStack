"""Phase 3 — Execution tools.

Extracted from server.py to reduce module size. These six tools handle trade
execution, position closing, order cancellation, fill queries, risk metrics,
and the decision audit trail.
"""

import asyncio
import os
import uuid as _uuid
from typing import Any

from loguru import logger

from quantstack.audit.models import AuditQuery, DecisionEvent
from quantstack.execution.broker_factory import get_broker_mode
from quantstack.execution.paper_broker import OrderRequest
from quantstack.learning.outcome_tracker import OutcomeTracker
from quantstack.mcp.server import mcp
from quantstack.mcp._state import (
    require_ctx,
    require_live_db,
    live_db_or_error,
    _serialize,
)
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain


# ---------------------------------------------------------------------------
# Helpers (private to this module, mirrored from server.py)
# ---------------------------------------------------------------------------


def _calc_quantity_from_size(
    position_size: str, equity: float, current_price: float
) -> int:
    """Convert a position_size label ('full', 'half', 'quarter') to shares."""
    fractions = {"full": 0.10, "half": 0.05, "quarter": 0.025}
    frac = fractions.get(position_size, 0.025)
    if current_price <= 0:
        return 0
    return max(1, int((equity * frac) / current_price))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@domain(Domain.EXECUTION)
@mcp.tool()
async def execute_trade(
    symbol: str,
    action: str,
    reasoning: str,
    confidence: float,
    quantity: int | None = None,
    position_size: str = "quarter",
    order_type: str = "market",
    limit_price: float | None = None,
    strategy_id: str | None = None,
    paper_mode: bool = True,
    regime_at_entry: str | None = None,
    instrument_type: str = "equity",
    strike: float | None = None,
    expiry: str | None = None,
    option_type: str | None = None,
) -> dict[str, Any]:
    """
    Execute a trade through the risk gate and broker.

    The risk gate is NEVER bypassed. reasoning and confidence are REQUIRED
    to ensure every trade has an audit trail.

    Args:
        symbol: Ticker symbol.
        action: "buy" or "sell".
        reasoning: REQUIRED. Why you are making this trade.
        confidence: REQUIRED. 0-1 confidence score.
        quantity: Number of shares/contracts. Auto-calculated from position_size if None.
        position_size: "full", "half", or "quarter" (used if quantity is None).
        order_type: "market" or "limit".
        limit_price: Required for limit orders.
        strategy_id: Links trade to a registered strategy.
        paper_mode: Must be explicitly False for live trading.
        instrument_type: "equity" (default) or "option".
        strike: Options only — strike price.
        expiry: Options only — expiration date (YYYY-MM-DD).
        option_type: Options only — "call" or "put".

    Returns:
        Dict with fill details or rejection reason.
    """
    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        # 1. Kill switch guard
        ctx.kill_switch.guard()

        # 2. Paper/live mode check
        if not paper_mode:
            use_real = os.getenv("USE_REAL_TRADING", "false").strip().lower()
            if use_real not in ("true", "1", "yes"):
                return {
                    "success": False,
                    "error": (
                        "Live trading rejected: paper_mode=False but "
                        "USE_REAL_TRADING env var is not 'true'. "
                        "Set USE_REAL_TRADING=true to enable live trading."
                    ),
                    "broker_mode": get_broker_mode(),
                }

        # 3. Get current price (from portfolio position or default)
        snapshot = ctx.portfolio.get_snapshot()
        pos = ctx.portfolio.get_position(symbol)
        current_price = pos.current_price if pos and pos.current_price > 0 else 0.0

        # If no current price from portfolio, we need one for risk calculations
        if current_price <= 0:
            return {
                "success": False,
                "error": (
                    f"No current price available for {symbol}. "
                    "Run get_portfolio_state or run_analysis first to populate prices."
                ),
            }

        # 4. Calculate quantity if not explicit
        if quantity is None or quantity <= 0:
            quantity = _calc_quantity_from_size(
                position_size, snapshot.total_equity, current_price
            )

        # 5. Risk gate check — SACRED, NEVER BYPASSED
        daily_volume = 1_000_000  # Default for paper mode
        verdict = ctx.risk_gate.check(
            symbol=symbol,
            side=action,
            quantity=quantity,
            current_price=current_price,
            daily_volume=daily_volume,
        )

        if not verdict.approved:
            violations = [v.description for v in verdict.violations]

            # Log the rejection
            ctx.audit.record(
                DecisionEvent(
                    event_id=str(_uuid.uuid4()),
                    session_id=ctx.session_id,
                    event_type="risk_rejection",
                    agent_name="ClaudeCode",
                    agent_role="strategic_brain",
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    output_summary=f"REJECTED: {'; '.join(violations)}",
                    risk_approved=False,
                    risk_violations=violations,
                    portfolio_snapshot=_serialize(snapshot) or {},
                )
            )

            return {
                "success": False,
                "risk_approved": False,
                "risk_violations": violations,
                "error": f"Risk gate rejected: {'; '.join(violations)}",
                "broker_mode": get_broker_mode(),
            }

        # Use approved_quantity (may have been scaled down)
        approved_qty = verdict.approved_quantity or quantity

        # 6. Execute via broker
        order = OrderRequest(
            symbol=symbol,
            side=action,
            quantity=approved_qty,
            order_type=order_type,
            limit_price=limit_price,
            current_price=current_price,
            daily_volume=daily_volume,
        )

        fill = ctx.broker.execute(order)

        # 7. Log to audit trail
        ctx.audit.record(
            DecisionEvent(
                event_id=str(_uuid.uuid4()),
                session_id=ctx.session_id,
                event_type="execution",
                agent_name="ClaudeCode",
                agent_role="strategic_brain",
                symbol=symbol,
                action=action,
                confidence=confidence,
                output_summary=(
                    f"{'FILLED' if not fill.rejected else 'REJECTED'}: "
                    f"{action.upper()} {fill.filled_quantity} {symbol} "
                    f"@ ${fill.fill_price:.4f} | reasoning: {reasoning[:200]}"
                ),
                output_structured={
                    "order_id": fill.order_id,
                    "fill_price": fill.fill_price,
                    "filled_quantity": fill.filled_quantity,
                    "slippage_bps": fill.slippage_bps,
                    "strategy_id": strategy_id,
                    "paper_mode": paper_mode,
                },
                risk_approved=True,
                portfolio_snapshot=_serialize(snapshot) or {},
            )
        )

        # 8. Outcome attribution — best-effort, never blocks the fill
        if strategy_id and not fill.rejected:
            try:
                tracker = OutcomeTracker()
                regime = regime_at_entry or "unknown"
                if action == "buy":
                    await asyncio.to_thread(
                        tracker.record_entry,
                        strategy_id,
                        symbol,
                        regime,
                        action,
                        fill.fill_price,
                        ctx.session_id,
                    )
                elif action == "sell":
                    await asyncio.to_thread(
                        tracker.record_exit,
                        strategy_id,
                        symbol,
                        fill.fill_price,
                    )
                    # Apply learning immediately after exit
                    await asyncio.to_thread(tracker.apply_learning, strategy_id)
            except Exception as _ot_exc:
                logger.debug(
                    f"[execute_trade] outcome attribution failed (non-critical): {_ot_exc}"
                )

        return {
            "success": not fill.rejected,
            "order_id": fill.order_id,
            "fill_price": fill.fill_price,
            "filled_quantity": fill.filled_quantity,
            "slippage_bps": round(fill.slippage_bps, 2),
            "commission": round(fill.commission, 4),
            "risk_approved": True,
            "risk_violations": [],
            "broker_mode": get_broker_mode(),
            "error": fill.reject_reason if fill.rejected else None,
        }

    except RuntimeError as e:
        # Kill switch or other runtime guard
        return {"success": False, "error": str(e), "broker_mode": get_broker_mode()}
    except Exception as e:
        logger.error(f"[quantpod_mcp] execute_trade failed: {e}")
        return {"success": False, "error": str(e), "broker_mode": get_broker_mode()}


@domain(Domain.EXECUTION)
@mcp.tool()
async def close_position(
    symbol: str,
    reasoning: str,
    quantity: int | None = None,
) -> dict[str, Any]:
    """
    Close an open position (all or partial).

    Infers the correct action (sell for longs, buy for shorts) from
    the current position side. Routes through risk gate and audit.

    Args:
        symbol: Ticker symbol of the position to close.
        reasoning: REQUIRED. Why you are closing.
        quantity: Shares to close. None = close all.

    Returns:
        Dict with fill details or error.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        pos = ctx.portfolio.get_position(symbol)
        if pos is None:
            return {"success": False, "error": f"No open position for {symbol}"}

        close_qty = quantity or abs(pos.quantity)
        close_action = "sell" if pos.side == "long" else "buy"

        return await execute_trade(
            symbol=symbol,
            action=close_action,
            reasoning=f"CLOSE: {reasoning}",
            confidence=1.0,
            quantity=close_qty,
            order_type="market",
            paper_mode=True,
        )
    except Exception as e:
        logger.error(f"[quantpod_mcp] close_position failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.EXECUTION)
@mcp.tool()
async def cancel_order(order_id: str) -> dict[str, Any]:
    """
    Cancel an open order by ID.

    Note: PaperBroker executes all orders immediately, so cancellation
    is only meaningful for limit orders that were not filled.

    Args:
        order_id: The order ID to cancel.

    Returns:
        Dict with success status.
    """
    return {
        "success": True,
        "message": (
            f"Order {order_id} cancel acknowledged. "
            "Note: PaperBroker fills all market orders immediately; "
            "unfilled limit orders are not persisted."
        ),
    }


@domain(Domain.EXECUTION)
@mcp.tool()
async def get_fills(
    symbol: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Get recent trade fills.

    Args:
        symbol: Filter by symbol. None = all symbols.
        limit: Maximum fills to return.

    Returns:
        Dict with list of fill records.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        fills = ctx.broker.get_fills(symbol=symbol, limit=limit)
        return {
            "success": True,
            "fills": [_serialize(f) for f in fills],
            "total": len(fills),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_fills failed: {e}")
        return {"success": False, "error": str(e), "fills": [], "total": 0}


@domain(Domain.EXECUTION)
@mcp.tool()
async def get_risk_metrics() -> dict[str, Any]:
    """
    Get current risk exposure, drawdown, and limits headroom.

    Returns:
        Dict with cash, equity, exposure, daily loss, and all limit values.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        snapshot = ctx.portfolio.get_snapshot()
        positions = ctx.portfolio.get_positions()
        limits = ctx.risk_gate.limits

        gross_exposure = sum(abs(p.quantity) * p.current_price for p in positions)
        equity = snapshot.total_equity or 1.0
        gross_pct = gross_exposure / equity
        daily_loss_pct = abs(min(0, snapshot.daily_pnl)) / equity if equity > 0 else 0.0

        return {
            "success": True,
            "cash": round(snapshot.cash, 2),
            "total_equity": round(snapshot.total_equity, 2),
            "positions_value": round(snapshot.positions_value, 2),
            "position_count": snapshot.position_count,
            "daily_pnl": round(snapshot.daily_pnl, 2),
            "daily_loss_pct": round(daily_loss_pct * 100, 2),
            "daily_loss_limit_pct": round(limits.daily_loss_limit_pct * 100, 2),
            "daily_headroom_pct": round(
                (limits.daily_loss_limit_pct - daily_loss_pct) * 100, 2
            ),
            "gross_exposure": round(gross_exposure, 2),
            "gross_exposure_pct": round(gross_pct * 100, 2),
            "max_gross_exposure_pct": round(limits.max_gross_exposure_pct * 100, 2),
            "largest_position_pct": round(snapshot.largest_position_pct * 100, 2),
            "max_position_pct": round(limits.max_position_pct * 100, 2),
            "kill_switch_active": ctx.kill_switch.is_active(),
            "risk_halted": ctx.risk_gate.is_halted(),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_risk_metrics failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.EXECUTION)
@mcp.tool()
async def get_audit_trail(
    session_id: str | None = None,
    symbol: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Query the decision audit trail.

    Args:
        session_id: Filter by session. None = current session.
        symbol: Filter by symbol.
        limit: Maximum entries.

    Returns:
        Dict with list of audit events.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        query = AuditQuery(
            session_id=session_id or "",
            symbol=symbol or "",
            limit=limit,
        )
        events = ctx.audit.query(query)
        return {
            "success": True,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "agent_name": e.agent_name,
                    "symbol": e.symbol,
                    "action": e.action,
                    "confidence": e.confidence,
                    "risk_approved": e.risk_approved,
                    "output_summary": (
                        e.output_summary[:300] if e.output_summary else ""
                    ),
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ],
            "total": len(events),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_audit_trail failed: {e}")
        return {"success": False, "error": str(e), "events": [], "total": 0}
