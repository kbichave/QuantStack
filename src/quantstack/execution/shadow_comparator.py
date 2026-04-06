"""
Shadow mode comparator — collects exit decisions from IntradayPositionManager
and ExecutionMonitor, compares them side-by-side for validation before cutover.

Thread-safe: both systems may call record() from different async tasks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

from loguru import logger

from quantstack.holding_period import HoldingType


@dataclass
class ShadowExitRecord:
    """A single exit decision from one of the two systems."""

    symbol: str
    source: str  # "intraday_pm" or "execution_monitor"
    reason: str
    trigger_price: float
    entry_price: float
    unrealized_pnl: float
    holding_type: HoldingType
    strategy_id: str
    timestamp: datetime


@dataclass
class ShadowComparison:
    """Result of comparing two exit decisions for the same symbol."""

    symbol: str
    match: bool
    divergence_type: str | None = None  # "monitor_only", "pm_only", "reason_mismatch"
    price_delta: float | None = None
    time_delta_seconds: float | None = None
    pm_record: ShadowExitRecord | None = None
    monitor_record: ShadowExitRecord | None = None


class ShadowComparator:
    """Collects and compares exit decisions from IntradayPositionManager
    and ExecutionMonitor during shadow mode validation.

    The comparison window is configurable (default 5 seconds). If both systems
    decide to exit the same symbol within this window, they are considered
    to have made the "same" decision and the reasons/prices are compared.
    """

    def __init__(self, comparison_window_seconds: float | None = None) -> None:
        self._window = comparison_window_seconds or float(
            os.getenv("SHADOW_COMPARISON_WINDOW_SECONDS", "5")
        )
        self._lock = Lock()
        # Most recent record per (symbol, source)
        self._records: dict[tuple[str, str], ShadowExitRecord] = {}
        self._comparisons: list[ShadowComparison] = []

    def record(self, entry: ShadowExitRecord) -> None:
        """Record an exit decision from either system. Thread-safe."""
        with self._lock:
            key = (entry.symbol, entry.source)
            self._records[key] = entry

            logger.info(
                f"[ShadowMode] {entry.source} exit: {entry.symbol} "
                f"reason={entry.reason} price={entry.trigger_price:.2f} "
                f"pnl=${entry.unrealized_pnl:.2f}"
            )

            # Auto-compare only if we have records from BOTH sources for this symbol
            pm_key = (entry.symbol, "intraday_pm")
            mon_key = (entry.symbol, "execution_monitor")
            if pm_key in self._records and mon_key in self._records:
                self._try_compare(entry.symbol)

    def compare(self, symbol: str) -> ShadowComparison | None:
        """Compare the most recent decisions for a symbol, if both exist within the window."""
        with self._lock:
            return self._try_compare(symbol)

    def _try_compare(self, symbol: str) -> ShadowComparison | None:
        """Internal compare — caller must hold lock."""
        pm_key = (symbol, "intraday_pm")
        mon_key = (symbol, "execution_monitor")

        pm_rec = self._records.get(pm_key)
        mon_rec = self._records.get(mon_key)

        if pm_rec is None and mon_rec is None:
            return None

        # Only one side fired
        if pm_rec is None and mon_rec is not None:
            comp = ShadowComparison(
                symbol=symbol,
                match=False,
                divergence_type="monitor_only",
                monitor_record=mon_rec,
            )
            self._comparisons.append(comp)
            return comp

        if mon_rec is None and pm_rec is not None:
            comp = ShadowComparison(
                symbol=symbol,
                match=False,
                divergence_type="pm_only",
                pm_record=pm_rec,
            )
            self._comparisons.append(comp)
            return comp

        # Both fired — check time window
        time_delta = abs((pm_rec.timestamp - mon_rec.timestamp).total_seconds())
        price_delta = abs(pm_rec.trigger_price - mon_rec.trigger_price)

        if time_delta > self._window:
            # Outside comparison window — treat as separate events
            comp = ShadowComparison(
                symbol=symbol,
                match=False,
                divergence_type="timing_delta",
                time_delta_seconds=time_delta,
                price_delta=price_delta,
                pm_record=pm_rec,
                monitor_record=mon_rec,
            )
        elif pm_rec.reason != mon_rec.reason:
            comp = ShadowComparison(
                symbol=symbol,
                match=False,
                divergence_type="reason_mismatch",
                time_delta_seconds=time_delta,
                price_delta=price_delta,
                pm_record=pm_rec,
                monitor_record=mon_rec,
            )
        else:
            comp = ShadowComparison(
                symbol=symbol,
                match=True,
                time_delta_seconds=time_delta,
                price_delta=price_delta,
                pm_record=pm_rec,
                monitor_record=mon_rec,
            )

        self._comparisons.append(comp)

        # Clear records after comparison to prevent double-counting
        self._records.pop(pm_key, None)
        self._records.pop(mon_key, None)

        return comp

    def summary(self) -> dict:
        """Aggregate metrics: total_exits, agreements, divergences, agreement_rate."""
        with self._lock:
            total = len(self._comparisons)
            agreements = sum(1 for c in self._comparisons if c.match)
            divergences = total - agreements
            rate = (agreements / total * 100) if total > 0 else 0.0

            divergence_breakdown: dict[str, int] = {}
            for c in self._comparisons:
                if not c.match and c.divergence_type:
                    divergence_breakdown[c.divergence_type] = (
                        divergence_breakdown.get(c.divergence_type, 0) + 1
                    )

            return {
                "total_exits": total,
                "agreements": agreements,
                "divergences": divergences,
                "agreement_rate": rate,
                "divergence_breakdown": divergence_breakdown,
            }

    def flush_to_log(self) -> None:
        """Write summary to structured log and clear old records."""
        stats = self.summary()
        logger.info(
            f"[ShadowMode] Summary: {stats['total_exits']} exits, "
            f"{stats['agreements']} agreements ({stats['agreement_rate']:.1f}%), "
            f"{stats['divergences']} divergences {stats['divergence_breakdown']}"
        )
        with self._lock:
            self._comparisons.clear()
            self._records.clear()
