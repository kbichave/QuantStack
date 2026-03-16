"""
PreTradeRiskGate — synchronous pre-order checks run before every submission.

Checks (in order, first failure blocks the order)
--------------------------------------------------
1. KillSwitch         — file-sentinel halt (always first)
2. Max order size     — single order qty × price > max_order_value
3. Max position size  — resulting position value > max_position_value
4. Max open positions — number of symbols with non-zero qty > max_positions
5. Orders per minute  — sliding-window rate limiter
6. Daily drawdown     — today's total P&L < -max_daily_loss

Design notes
------------
- All checks are synchronous (called from the hot path before broker.place_order).
- Reads from FillTracker are lock-protected inside FillTracker; the gate itself
  has no lock because it is called from a single execution thread.
- ``check()`` raises ``RiskGateError`` with a descriptive message.
  Callers catch this, log it, and skip order submission.
- Limits are set at construction time; hot-patching is intentionally not
  supported to avoid race conditions during live trading.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from quantcore.execution.fill_tracker import FillTracker
from quantcore.execution.kill_switch import KillSwitch, KillSwitchError
from quantcore.execution.unified_models import UnifiedOrder


class RiskGateError(RuntimeError):
    """Raised when a pre-trade check fails.  Contains the rule that triggered."""

    def __init__(self, rule: str, message: str) -> None:
        self.rule = rule
        self.message = message
        super().__init__(f"[{rule}] {message}")


@dataclass
class RiskLimits:
    """Configurable hard limits for the risk gate.

    Attributes:
        max_order_value:    Maximum notional value of a single order (USD).
        max_position_value: Maximum notional value held in one symbol (USD).
        max_positions:      Maximum number of concurrent open symbol positions.
        max_orders_per_min: Maximum orders submitted per rolling 60-second window.
        max_daily_loss:     Maximum daily loss (positive number; loss triggers at
                            total P&L < -max_daily_loss).  0 = disabled.
    """

    max_order_value: float = 50_000.0
    max_position_value: float = 100_000.0
    max_positions: int = 20
    max_orders_per_min: int = 60
    max_daily_loss: float = 5_000.0  # 0 disables the check


class PreTradeRiskGate:
    """Validates orders against position, rate, and drawdown limits.

    Args:
        limits:       Hard limits for all checks.
        fill_tracker: Live position and P&L state provider.
        kill_switch:  File-sentinel kill switch (default sentinel path).
    """

    def __init__(
        self,
        limits: RiskLimits,
        fill_tracker: FillTracker,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self._limits = limits
        self._tracker = fill_tracker
        self._kill_switch = kill_switch or KillSwitch()
        # Sliding window of order submission timestamps (epoch float)
        self._order_times: deque[float] = deque()

    # ── Public interface ──────────────────────────────────────────────────────

    def check(self, order: UnifiedOrder, current_price: float) -> None:
        """Run all pre-trade checks.  Raises RiskGateError on first failure.

        Args:
            order:         The order about to be submitted.
            current_price: Latest market price for the symbol (used to compute
                           notional values without hitting the broker API).
        """
        self._check_kill_switch()
        self._check_order_size(order, current_price)
        self._check_position_size(order, current_price)
        self._check_max_positions(order)
        self._check_order_rate()
        self._check_drawdown()

    def record_submission(self) -> None:
        """Call after a successful order submission to update the rate limiter."""
        self._order_times.append(time.monotonic())

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_kill_switch(self) -> None:
        try:
            self._kill_switch.check()
        except KillSwitchError as exc:
            raise RiskGateError("KILL_SWITCH", str(exc)) from exc

    def _check_order_size(self, order: UnifiedOrder, price: float) -> None:
        notional = order.quantity * price
        if notional > self._limits.max_order_value:
            raise RiskGateError(
                "MAX_ORDER_SIZE",
                f"Order notional ${notional:,.2f} exceeds limit "
                f"${self._limits.max_order_value:,.2f} "
                f"({order.quantity} × ${price:.4f})",
            )

    def _check_position_size(self, order: UnifiedOrder, price: float) -> None:
        pos = self._tracker.get_position(order.symbol)
        current_qty = pos.quantity if pos else 0.0
        delta = order.quantity if order.side.lower() == "buy" else -order.quantity
        resulting_qty = current_qty + delta
        resulting_value = abs(resulting_qty * price)
        if resulting_value > self._limits.max_position_value:
            raise RiskGateError(
                "MAX_POSITION_SIZE",
                f"Resulting position for {order.symbol} would be "
                f"${resulting_value:,.2f} (limit ${self._limits.max_position_value:,.2f}). "
                f"Current qty: {current_qty}, order delta: {delta:+}",
            )

    def _check_max_positions(self, order: UnifiedOrder) -> None:
        pos = self._tracker.get_position(order.symbol)
        currently_open = self._tracker.position_count()
        opening_new = (pos is None or abs(pos.quantity) < 1e-9) and order.side.lower() == "buy"
        if opening_new and currently_open >= self._limits.max_positions:
            raise RiskGateError(
                "MAX_POSITIONS",
                f"Cannot open new position in {order.symbol}: "
                f"{currently_open}/{self._limits.max_positions} positions already open",
            )

    def _check_order_rate(self) -> None:
        now = time.monotonic()
        window = 60.0
        # Prune events outside the rolling window
        while self._order_times and self._order_times[0] < now - window:
            self._order_times.popleft()
        count = len(self._order_times)
        if count >= self._limits.max_orders_per_min:
            raise RiskGateError(
                "ORDER_RATE",
                f"Rate limit exceeded: {count} orders in the last 60s "
                f"(limit {self._limits.max_orders_per_min})",
            )

    def _check_drawdown(self) -> None:
        if self._limits.max_daily_loss <= 0:
            return
        pnl = self._tracker.daily_total_pnl()
        if pnl < -self._limits.max_daily_loss:
            raise RiskGateError(
                "DAILY_DRAWDOWN",
                f"Daily loss ${-pnl:,.2f} exceeds limit "
                f"${self._limits.max_daily_loss:,.2f} — trading halted for the day",
            )

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return current risk gate state for monitoring."""
        now = time.monotonic()
        window = 60.0
        while self._order_times and self._order_times[0] < now - window:
            self._order_times.popleft()
        return {
            "kill_switch": self._kill_switch.status(),
            "orders_last_60s": len(self._order_times),
            "max_orders_per_min": self._limits.max_orders_per_min,
            "open_positions": self._tracker.position_count(),
            "max_positions": self._limits.max_positions,
            "daily_pnl": round(self._tracker.daily_total_pnl(), 2),
            "max_daily_loss": self._limits.max_daily_loss,
            "net_exposure": round(self._tracker.net_exposure(), 2),
        }
