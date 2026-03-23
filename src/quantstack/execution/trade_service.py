# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Trade execution service — core logic extracted from MCP tools.

This module contains the business logic for trade execution, independent of
the MCP server. Both MCP tools and the autonomous runner call these functions
directly, passing explicit dependencies (portfolio, risk_gate, broker, audit).

Design invariant: the risk gate is NEVER bypassed.
"""

from __future__ import annotations

import asyncio
import os
import uuid as _uuid
from typing import Any

from loguru import logger

from quantstack.audit.models import DecisionEvent
from quantstack.db import db_conn
from quantstack.execution.broker_factory import get_broker_mode
from quantstack.execution.hook_registry import fire as _fire_hook
from quantstack.execution.paper_broker import OrderRequest
from quantstack.shared.serializers import serialize_for_json


def calc_quantity_from_size(
    position_size: str, equity: float, current_price: float
) -> int:
    """Convert a position_size label ('full', 'half', 'quarter') to shares."""
    fractions = {"full": 0.10, "half": 0.05, "quarter": 0.025}
    frac = fractions.get(position_size, 0.025)
    if current_price <= 0:
        return 0
    notional = equity * frac
    return max(1, int(notional / current_price))


async def execute_trade(
    *,
    portfolio,
    risk_gate,
    broker,
    audit,
    kill_switch,
    session_id: str,
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
    # v2 — position metadata for autonomous trading loop
    instrument_type: str = "equity",
    time_horizon: str = "swing",
    stop_price: float | None = None,
    target_price: float | None = None,
    trailing_stop: float | None = None,
    entry_atr: float | None = None,
    option_expiry: str | None = None,
    option_strike: float | None = None,
    option_type: str | None = None,
) -> dict[str, Any]:
    """
    Execute a trade through the risk gate and broker.

    All dependencies are passed explicitly — no MCP state imports.

    Returns:
        Dict with fill details or rejection reason.
    """
    try:
        # 1. Kill switch guard
        kill_switch.guard()

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

        # 3. Get current price
        snapshot = portfolio.get_snapshot()
        pos = portfolio.get_position(symbol)
        current_price = pos.current_price if pos and pos.current_price > 0 else 0.0

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
            quantity = calc_quantity_from_size(
                position_size, snapshot.total_equity, current_price
            )

        # 5. Risk gate check — SACRED, NEVER BYPASSED
        daily_volume = 1_000_000  # Default for paper mode
        verdict = risk_gate.check(
            symbol=symbol,
            side=action,
            quantity=quantity,
            current_price=current_price,
            daily_volume=daily_volume,
        )

        if not verdict.approved:
            violations = [v.description for v in verdict.violations]

            audit.record(
                DecisionEvent(
                    event_id=str(_uuid.uuid4()),
                    session_id=session_id,
                    event_type="risk_rejection",
                    agent_name="TradeService",
                    agent_role="execution",
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    output_summary=f"REJECTED: {'; '.join(violations)}",
                    risk_approved=False,
                    risk_violations=violations,
                    portfolio_snapshot=serialize_for_json(snapshot) or {},
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

        fill = broker.execute(order)

        # 6b. Persist position metadata (strategy context + exit levels)
        # The broker's execute() already calls portfolio.upsert_position() with basic
        # fields. We update the extra metadata columns separately so the broker layer
        # stays unaware of trading-loop concerns.
        if not fill.rejected and action in ("buy",):
            try:
                with db_conn() as _mc:
                    sets = []
                    vals: list = []
                    for col, val in [
                        ("strategy_id", strategy_id or ""),
                        ("regime_at_entry", regime_at_entry or "unknown"),
                        ("instrument_type", instrument_type),
                        ("time_horizon", time_horizon),
                        ("stop_price", stop_price),
                        ("target_price", target_price),
                        ("trailing_stop", trailing_stop),
                        ("entry_atr", entry_atr),
                        ("option_expiry", option_expiry),
                        ("option_strike", option_strike),
                        ("option_type", option_type),
                    ]:
                        if val is not None:
                            sets.append(f"{col} = ?")
                            vals.append(val)
                    if sets:
                        vals.append(symbol)
                        _mc.execute(
                            f"UPDATE positions SET {', '.join(sets)} WHERE symbol = ?",
                            vals,
                        )
            except Exception as _meta_exc:
                logger.debug(
                    f"[trade_service] position metadata update failed (non-critical): {_meta_exc}"
                )

        # 7. Log to audit trail
        audit.record(
            DecisionEvent(
                event_id=str(_uuid.uuid4()),
                session_id=session_id,
                event_type="execution",
                agent_name="TradeService",
                agent_role="execution",
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
                portfolio_snapshot=serialize_for_json(snapshot) or {},
            )
        )

        # 8. Outcome attribution — fire via hook registry (best-effort)
        if strategy_id and not fill.rejected:
            _fire_hook(
                "trade_fill",
                strategy_id=strategy_id,
                symbol=symbol,
                action=action,
                fill_price=fill.fill_price,
                session_id=session_id,
                regime_at_entry=regime_at_entry or "unknown",
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
        return {"success": False, "error": str(e), "broker_mode": get_broker_mode()}
    except Exception as e:
        logger.error(f"[trade_service] execute_trade failed: {e}")
        return {"success": False, "error": str(e), "broker_mode": get_broker_mode()}
