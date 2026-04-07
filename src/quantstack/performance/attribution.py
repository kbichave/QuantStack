"""Cycle-level P&L attribution engine.

Decomposes each trading cycle's returns into four components:
  - **Factor**: market/sector/style beta exposure
  - **Timing**: entry/exit quality vs VWAP
  - **Selection**: stock-specific alpha (residual)
  - **Cost**: slippage + commission drag

The accounting identity ``factor + timing + selection + cost == total_pnl``
is maintained by construction (selection is the residual).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import db_conn


@dataclass
class PositionAttribution:
    """Per-position P&L decomposition."""

    symbol: str
    weight: float
    total_pnl: float
    factor_pnl: float
    timing_pnl: float
    selection_pnl: float
    cost_pnl: float


@dataclass
class CycleAttribution:
    """Full-cycle P&L decomposition across four components."""

    cycle_id: str
    total_pnl: float
    factor_contribution: float
    timing_contribution: float
    selection_contribution: float
    cost_contribution: float
    per_position: list[PositionAttribution] = field(default_factory=list)
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for TradingState and DB."""
        return {
            "cycle_id": self.cycle_id,
            "total_pnl": self.total_pnl,
            "factor_contribution": self.factor_contribution,
            "timing_contribution": self.timing_contribution,
            "selection_contribution": self.selection_contribution,
            "cost_contribution": self.cost_contribution,
            "per_position": [asdict(p) for p in self.per_position],
            "computed_at": self.computed_at.isoformat(),
        }


async def compute_cycle_attribution(
    positions: list[dict],
    fills: list[dict],
    benchmark_return: float,
    sector_returns: dict[str, float] | None = None,
) -> CycleAttribution:
    """Decompose cycle P&L into four components.

    Args:
        positions: Active positions with symbol, quantity, market_value, sector,
            unrealized_pnl fields.
        fills: Cycle fills with symbol, quantity, fill_price, side,
            vwap (optional), commission (optional), slippage (optional).
        benchmark_return: Benchmark % return for the cycle period.
        sector_returns: Optional sector -> % return mapping. If missing,
            uses benchmark_return for all sectors.

    Returns:
        CycleAttribution with all four components summing to total_pnl.
    """
    sector_returns = sector_returns or {}
    cycle_id = str(uuid.uuid4())[:12]

    if not positions and not fills:
        return CycleAttribution(
            cycle_id=cycle_id,
            total_pnl=0.0,
            factor_contribution=0.0,
            timing_contribution=0.0,
            selection_contribution=0.0,
            cost_contribution=0.0,
        )

    # --- Total portfolio value for weighting ---
    total_mv = sum(abs(p.get("market_value", 0)) for p in positions)
    if total_mv == 0:
        total_mv = 1.0  # avoid division by zero

    # --- Factor contribution: weighted beta exposure ---
    factor_total = 0.0
    for p in positions:
        mv = abs(p.get("market_value", 0))
        weight = mv / total_mv
        sector = p.get("sector", "")
        sector_ret = sector_returns.get(sector, benchmark_return)
        factor_pnl = weight * sector_ret * mv
        factor_total += factor_pnl

    # --- Cost contribution: slippage + commissions ---
    cost_total = 0.0
    for f in fills:
        slippage = abs(f.get("slippage", 0.0))
        commission = abs(f.get("commission", 0.0))
        cost_total -= (slippage + commission)

    # --- Timing contribution: entry/exit quality vs VWAP ---
    timing_total = 0.0
    for f in fills:
        fill_price = f.get("fill_price", 0.0)
        vwap = f.get("vwap", fill_price)
        qty = f.get("quantity", 0)
        side = f.get("side", "buy").lower()

        if fill_price <= 0 or vwap <= 0:
            continue

        if side == "buy":
            # Bought below VWAP = positive timing
            timing_total += (vwap - fill_price) * qty
        else:
            # Sold above VWAP = positive timing
            timing_total += (fill_price - vwap) * qty

    # --- Total P&L ---
    total_pnl = sum(p.get("unrealized_pnl", 0.0) for p in positions)
    # Add realized P&L from fills
    total_pnl += sum(f.get("realized_pnl", 0.0) for f in fills)

    # --- Selection = residual (guarantees accounting identity) ---
    selection_total = total_pnl - factor_total - timing_total - cost_total

    # --- Per-position attribution ---
    per_position: list[PositionAttribution] = []
    for p in positions:
        mv = abs(p.get("market_value", 0))
        weight = mv / total_mv
        sector = p.get("sector", "")
        sector_ret = sector_returns.get(sector, benchmark_return)
        pos_factor = weight * sector_ret * mv
        pos_pnl = p.get("unrealized_pnl", 0.0)
        pos_cost = 0.0
        pos_timing = 0.0

        # Match fills to this position
        sym = p.get("symbol", "")
        for f in fills:
            if f.get("symbol") == sym:
                pos_cost -= abs(f.get("slippage", 0.0)) + abs(f.get("commission", 0.0))
                fp = f.get("fill_price", 0.0)
                vw = f.get("vwap", fp)
                q = f.get("quantity", 0)
                side = f.get("side", "buy").lower()
                if fp > 0 and vw > 0:
                    if side == "buy":
                        pos_timing += (vw - fp) * q
                    else:
                        pos_timing += (fp - vw) * q

        pos_selection = pos_pnl - pos_factor - pos_timing - pos_cost

        per_position.append(PositionAttribution(
            symbol=sym,
            weight=weight,
            total_pnl=pos_pnl,
            factor_pnl=pos_factor,
            timing_pnl=pos_timing,
            selection_pnl=pos_selection,
            cost_pnl=pos_cost,
        ))

    # --- Accounting identity check ---
    identity_sum = factor_total + timing_total + selection_total + cost_total
    gap = abs(identity_sum - total_pnl)
    if gap > 1e-6:
        logger.warning(
            "[attribution] Accounting identity gap: %.6f (sum=%.2f, total=%.2f)",
            gap, identity_sum, total_pnl,
        )
        try:
            from quantstack.tools.functions.system_alerts import emit_system_alert

            await emit_system_alert(
                category="performance_degradation",
                severity="warning",
                title=f"Attribution identity gap: {gap:.6f}",
                detail=(
                    f"factor={factor_total:.2f} + timing={timing_total:.2f} + "
                    f"selection={selection_total:.2f} + cost={cost_total:.2f} = {identity_sum:.2f} "
                    f"!= total_pnl={total_pnl:.2f}"
                ),
                source="attribution",
            )
        except Exception:
            pass

    return CycleAttribution(
        cycle_id=cycle_id,
        total_pnl=total_pnl,
        factor_contribution=factor_total,
        timing_contribution=timing_total,
        selection_contribution=selection_total,
        cost_contribution=cost_total,
        per_position=per_position,
    )


def persist_cycle_attribution(
    attribution: CycleAttribution,
    graph_cycle_number: int,
) -> None:
    """Write attribution to the cycle_attribution DB table."""
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO cycle_attribution "
                "(cycle_id, graph_cycle_number, total_pnl, "
                "factor_contribution, timing_contribution, "
                "selection_contribution, cost_contribution, "
                "per_position, computed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    attribution.cycle_id,
                    graph_cycle_number,
                    attribution.total_pnl,
                    attribution.factor_contribution,
                    attribution.timing_contribution,
                    attribution.selection_contribution,
                    attribution.cost_contribution,
                    json.dumps([asdict(p) for p in attribution.per_position]),
                    attribution.computed_at,
                ],
            )
    except Exception as e:
        logger.error("[attribution] Failed to persist: %s", e)
