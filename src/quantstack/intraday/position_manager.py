# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
IntradayPositionManager — per-bar position management for intraday trading.

Responsibilities:
  1. Mark-to-market all positions on every bar
  2. Enforce trailing stops (ATR-based) and time stops
  3. Flatten all positions at a configurable time (default 15:55 ET)
  4. Track intraday trade count and P&L
  5. Emergency flatten on kill switch activation

This runs as a callback on IncrementalFeatureEngine — it sees every bar
before the signal evaluator does, so exits are processed before new entries.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

import pytz
from loguru import logger

from quantstack.data.streaming.incremental_features import IncrementalFeatures
from quantstack.core.execution.fill_tracker import FillEvent, FillTracker

ET = pytz.timezone("US/Eastern")

# ---------------------------------------------------------------------------
# Position metadata (entry tracking for stops)
# ---------------------------------------------------------------------------


@dataclass
class IntradayPositionMeta:
    """Per-position metadata for intraday stop management."""

    symbol: str
    entry_price: float
    entry_time: datetime
    entry_bar_count: int  # bar index at entry
    high_water_mark: float  # highest price since entry (for trailing stop)
    low_water_mark: float  # lowest price since entry (for short trailing stop)
    strategy_id: str = ""
    atr_at_entry: float = 0.0  # ATR at the time of entry (for stop calculation)


# ---------------------------------------------------------------------------
# IntradayPositionManager
# ---------------------------------------------------------------------------


class IntradayPositionManager:
    """Per-bar position management: MTM, stops, flatten-at-close.

    Args:
        fill_tracker: Live position tracker (in-memory, thread-safe).
        broker_execute_fn: Async callable to submit exit orders. Signature:
            ``async def execute(symbol, side, quantity, reason) -> dict``
        kill_switch_fn: Callable returning True if kill switch is active.
        flatten_time_et: Time (ET) to flatten all positions (e.g. "15:55").
        trailing_stop_atr_mult: Trailing stop distance in ATR multiples.
        max_hold_bars: Force exit after this many bars (0 = disabled).
        intraday_loss_stop_pct: Per-position loss threshold as decimal (e.g. 0.02 = 2%).
    """

    def __init__(
        self,
        fill_tracker: FillTracker,
        broker_execute_fn: Any,
        kill_switch_fn: Any = lambda: False,
        flatten_time_et: str = "15:55",
        trailing_stop_atr_mult: float = 2.0,
        max_hold_bars: int = 0,
        intraday_loss_stop_pct: float = 0.02,
    ) -> None:
        self._tracker = fill_tracker
        self._execute = broker_execute_fn
        self._kill_switch_fn = kill_switch_fn

        h, m = flatten_time_et.split(":")
        self._flatten_time = time(int(h), int(m))

        self._trailing_stop_mult = trailing_stop_atr_mult
        self._max_hold_bars = max_hold_bars
        self._loss_stop_pct = intraday_loss_stop_pct

        self._position_meta: dict[str, IntradayPositionMeta] = {}
        self._bar_count = 0
        self._trades_today = 0
        self._flattened = False
        self._lock = threading.Lock()
        self._exit_log: list[dict] = []

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def trades_today(self) -> int:
        return self._trades_today

    @property
    def intraday_pnl(self) -> float:
        return self._tracker.daily_realised_pnl()

    @property
    def is_flattened(self) -> bool:
        return self._flattened

    @property
    def exit_log(self) -> list[dict]:
        return list(self._exit_log)

    # ── Entry registration ──────────────────────────────────────────────────

    def register_entry(
        self,
        symbol: str,
        price: float,
        atr: float,
        strategy_id: str = "",
    ) -> None:
        """Call after a fill to track the position for stop management."""
        with self._lock:
            self._position_meta[symbol] = IntradayPositionMeta(
                symbol=symbol,
                entry_price=price,
                entry_time=datetime.now(ET),
                entry_bar_count=self._bar_count,
                high_water_mark=price,
                low_water_mark=price,
                atr_at_entry=atr,
                strategy_id=strategy_id,
            )
            self._trades_today += 1

    def unregister(self, symbol: str) -> None:
        """Remove position meta after exit."""
        with self._lock:
            self._position_meta.pop(symbol, None)

    # ── Pre-entry portfolio risk checks ─────────────────────────────────────

    def pre_entry_check(
        self, symbol: str, sector: str | None = None
    ) -> tuple[bool, str]:
        """Check portfolio-level constraints before opening a new position.

        Returns (approved, reason). Called by the intraday signal evaluator
        before submitting an entry order.

        Checks:
          1. Max concurrent positions (default 6)
          2. Max same-sector positions (default 3)
          3. No duplicate entries for already-held symbols
        """
        with self._lock:
            open_symbols = set(self._position_meta.keys())

        # Already holding this symbol
        if symbol in open_symbols:
            return False, f"already_holding_{symbol}"

        # Max concurrent positions
        max_positions = 6
        if len(open_symbols) >= max_positions:
            return False, f"max_positions_reached ({len(open_symbols)}/{max_positions})"

        # Sector concentration check
        if sector:
            max_sector = 3
            sector_count = sum(
                1
                for s, meta in self._position_meta.items()
                if getattr(meta, "sector", None) == sector
            )
            if sector_count >= max_sector:
                return False, f"sector_limit_{sector} ({sector_count}/{max_sector})"

        return True, "approved"

    # ── Per-bar callback ────────────────────────────────────────────────────

    async def on_features(self, features: IncrementalFeatures) -> None:
        """Called on every bar by IncrementalFeatureEngine.

        Order of operations:
          1. Mark-to-market
          2. Kill switch check
          3. Flatten-at-close time check
          4. Per-position stop checks
        """
        self._bar_count += 1
        symbol = features.symbol
        price = features.close

        # 1. Mark-to-market
        self._tracker.update_price(symbol, price)

        # 2. Kill switch
        if self._kill_switch_fn() and not self._flattened:
            logger.warning("[IntradayPM] Kill switch active — flattening all")
            await self.flatten_all(reason="kill_switch")
            return

        # 3. Flatten-at-close
        now_et = datetime.now(ET).time()
        if now_et >= self._flatten_time and not self._flattened:
            logger.info(f"[IntradayPM] Flatten time {self._flatten_time} reached")
            await self.flatten_all(reason="flatten_at_close")
            return

        # 4. Per-position stop checks
        with self._lock:
            meta = self._position_meta.get(symbol)
        if meta is None:
            return

        pos = self._tracker.get_position(symbol)
        if pos is None or pos.quantity == 0:
            self.unregister(symbol)
            return

        # Update high/low water marks
        meta.high_water_mark = max(meta.high_water_mark, price)
        meta.low_water_mark = min(meta.low_water_mark, price)

        exit_reason = self._check_stops(meta, pos, features)
        if exit_reason:
            await self._exit_position(symbol, pos.quantity, exit_reason)

    # ── Stop checks ─────────────────────────────────────────────────────────

    def _check_stops(
        self,
        meta: IntradayPositionMeta,
        pos: Any,
        features: IncrementalFeatures,
    ) -> str | None:
        """Return exit reason string if any stop is hit, else None."""
        price = features.close
        qty = pos.quantity

        # Trailing stop (ATR-based)
        if meta.atr_at_entry > 0 and self._trailing_stop_mult > 0:
            stop_dist = meta.atr_at_entry * self._trailing_stop_mult
            if qty > 0 and price < meta.high_water_mark - stop_dist:
                return f"trailing_stop (hwm={meta.high_water_mark:.2f} stop_dist={stop_dist:.2f})"
            if qty < 0 and price > meta.low_water_mark + stop_dist:
                return f"trailing_stop_short (lwm={meta.low_water_mark:.2f} stop_dist={stop_dist:.2f})"

        # Time stop
        if self._max_hold_bars > 0:
            bars_held = self._bar_count - meta.entry_bar_count
            if bars_held >= self._max_hold_bars:
                return f"time_stop ({bars_held} bars >= {self._max_hold_bars})"

        # Intraday loss stop
        if self._loss_stop_pct > 0 and meta.entry_price > 0:
            pnl_pct = (price - meta.entry_price) / meta.entry_price
            if qty > 0 and pnl_pct < -self._loss_stop_pct:
                return f"loss_stop ({pnl_pct:.2%} < -{self._loss_stop_pct:.0%})"
            if qty < 0 and pnl_pct > self._loss_stop_pct:
                return f"loss_stop_short ({pnl_pct:.2%} > {self._loss_stop_pct:.0%})"

        return None

    # ── Exit helpers ────────────────────────────────────────────────────────

    async def _exit_position(self, symbol: str, quantity: float, reason: str) -> None:
        """Submit an exit order and log it."""
        side = "sell" if quantity > 0 else "buy"
        abs_qty = abs(quantity)

        logger.info(f"[IntradayPM] Exiting {symbol}: {side} {abs_qty} — {reason}")

        try:
            result = await self._execute(
                symbol=symbol, side=side, quantity=abs_qty, reason=reason
            )
            self._exit_log.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs_qty,
                    "reason": reason,
                    "result": result,
                }
            )
        except Exception as exc:
            logger.error(f"[IntradayPM] Exit failed for {symbol}: {exc}")
            self._exit_log.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs_qty,
                    "reason": reason,
                    "error": str(exc),
                }
            )
        finally:
            self.unregister(symbol)

    async def flatten_all(self, reason: str = "flatten_at_close") -> list[dict]:
        """Market-sell/cover all open intraday positions."""
        self._flattened = True
        positions = self._tracker.get_open_positions()
        if not positions:
            logger.info("[IntradayPM] No positions to flatten")
            return []

        logger.info(f"[IntradayPM] Flattening {len(positions)} positions — {reason}")
        tasks = []
        for sym, pos in positions.items():
            if pos.quantity != 0:
                tasks.append(self._exit_position(sym, pos.quantity, reason))

        await asyncio.gather(*tasks, return_exceptions=True)
        return self._exit_log

    # ── Daily reset ─────────────────────────────────────────────────────────

    def reset_daily(self) -> None:
        """Reset for a new trading day."""
        with self._lock:
            self._position_meta.clear()
            self._bar_count = 0
            self._trades_today = 0
            self._flattened = False
            self._exit_log.clear()
