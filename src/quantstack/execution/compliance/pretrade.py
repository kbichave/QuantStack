"""
Pre-trade SEC compliance checks: PDT rule and Reg-T margin.

These checks run BEFORE order submission and are independent of the
RiskGate hard-risk checks.  They enforce SEC/FINRA regulatory rules
that the risk gate does not cover:

  - **PDT (Pattern Day Trader)**: FINRA Rule 4210 — accounts under $25K
    that execute 4+ day trades in a rolling 5-business-day window are
    flagged and restricted.  We reject at 3 (the 4th would trigger the
    flag).
  - **Reg-T Margin**: 50% initial margin for long equity, full premium
    for long options, zero additional margin for sell/close orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from quantstack.db import PgConnection
from quantstack.execution.compliance.calendar import (
    rolling_business_day_window,
    trading_day_for,
)


@dataclass
class ComplianceResult:
    """Outcome of a pre-trade compliance check."""

    approved: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PDT Checker
# ---------------------------------------------------------------------------


class PDTChecker:
    """Pattern Day Trader detection and recording.

    A "day trade" is a round-trip (open + close) in the same security on
    the same business day.  FINRA flags accounts that execute 4+ day
    trades in any 5-business-day window AND hold less than $25K equity.

    We reject pre-trade when the *next* trade would be the 4th day trade
    (count >= 3) for accounts below the threshold.
    """

    PDT_THRESHOLD_EQUITY = 25_000.0
    PDT_WINDOW_DAYS = 5
    PDT_MAX_ALLOWED = 3  # reject when count reaches this (4th would trigger flag)

    def check(
        self,
        conn: PgConnection,
        order: dict,
        account_equity: float,
        positions: list[dict],
    ) -> ComplianceResult:
        """Pre-trade PDT check.

        Args:
            conn: Live database connection.
            order: Dict with keys ``symbol``, ``side``, ``quantity``, ``timestamp``.
            account_equity: Current account equity in dollars.
            positions: List of dicts with at least ``symbol``, ``side``,
                ``opened_at`` (datetime).

        Returns:
            ComplianceResult — approved or rejected with reason.
        """
        # Only closing trades can create day trades.  A "close" is a sell
        # when we hold long, or a buy when we hold short.
        if not self._would_close_same_day(order, positions):
            return ComplianceResult(approved=True)

        # Count existing day trades in the rolling 5-business-day window.
        order_ts = order.get("timestamp") or datetime.utcnow()
        trade_date = trading_day_for(order_ts)
        window_dates = rolling_business_day_window(trade_date, self.PDT_WINDOW_DAYS)

        if not window_dates:
            return ComplianceResult(approved=True)

        window_start = window_dates[0]
        window_end = window_dates[-1]

        row = conn.execute(
            "SELECT COUNT(*) FROM day_trades "
            "WHERE trade_date >= %s AND trade_date <= %s",
            (window_start, window_end),
        ).fetchone()
        count = row[0] if row else 0

        if count >= self.PDT_MAX_ALLOWED and account_equity < self.PDT_THRESHOLD_EQUITY:
            return ComplianceResult(
                approved=False,
                reason=(
                    f"PDT rule: {count} day trades in rolling 5-day window "
                    f"and account equity ${account_equity:,.2f} < $25,000"
                ),
            )

        # Warn if approaching the limit regardless of equity.
        warnings: list[str] = []
        if count >= self.PDT_MAX_ALLOWED - 1:
            warnings.append(
                f"PDT warning: {count} day trades in window — approaching limit"
            )

        return ComplianceResult(approved=True, warnings=warnings)

    def record_if_day_trade(
        self,
        conn: PgConnection,
        fill_event: dict,
        positions: list[dict],
    ) -> bool:
        """Post-fill: record a day trade if the fill closes a same-day position.

        Args:
            conn: Live database connection.
            fill_event: Dict with ``order_id``, ``symbol``, ``side``,
                ``quantity``, ``price``, ``timestamp``.
            positions: Current position state (same schema as ``check``).

        Returns:
            True if a day trade was recorded, False otherwise.
        """
        if not self._would_close_same_day(fill_event, positions):
            return False

        fill_ts = fill_event.get("timestamp") or datetime.utcnow()
        trade_date = trading_day_for(fill_ts)

        # Find the matching position to get the opening order_id.
        open_order_id = ""
        symbol = fill_event["symbol"]
        for pos in positions:
            if pos.get("symbol") == symbol:
                open_order_id = pos.get("order_id", "")
                break

        conn.execute(
            "INSERT INTO day_trades "
            "(symbol, open_order_id, close_order_id, trade_date, quantity, account_equity) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                symbol,
                open_order_id,
                fill_event.get("order_id", ""),
                trade_date,
                fill_event.get("quantity", 0),
                0.0,  # account_equity at time of fill — caller can update
            ),
        )
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _would_close_same_day(order: dict, positions: list[dict]) -> bool:
        """Return True if this order/fill would close a position opened today."""
        symbol = order.get("symbol", "")
        side = order.get("side", "").lower()

        for pos in positions:
            if pos.get("symbol") != symbol:
                continue

            # Determine if the order is closing this position.
            pos_side = pos.get("side", "").lower()
            is_closing = (
                (pos_side in ("buy", "long") and side == "sell")
                or (pos_side in ("sell", "short") and side == "buy")
            )
            if not is_closing:
                continue

            # Check if position was opened on the same business day.
            opened_at = pos.get("opened_at")
            if opened_at is None:
                continue

            order_ts = order.get("timestamp") or datetime.utcnow()
            order_day = trading_day_for(order_ts)

            if isinstance(opened_at, datetime):
                opened_day = trading_day_for(opened_at)
            else:
                # Already a date
                opened_day = opened_at

            if order_day == opened_day:
                return True

        return False


# ---------------------------------------------------------------------------
# Margin Calculator
# ---------------------------------------------------------------------------


class MarginCalculator:
    """Reg-T initial margin check.

    Rules:
      - Long equity: 50% initial margin (Reg T).
      - Long options (buy): premium is the margin requirement.
      - Sell/close orders: no additional margin required.
    """

    REG_T_INITIAL_MARGIN = 0.50

    def check(
        self,
        conn: PgConnection,
        order: dict,
        positions: list[dict],
        account_equity: float,
    ) -> ComplianceResult:
        """Pre-trade margin check.

        Args:
            conn: Database connection (unused currently, reserved for
                future margin queries).
            order: Dict with ``symbol``, ``side``, ``quantity``, ``price``,
                and optionally ``instrument_type`` (default ``"equity"``).
            positions: Current positions list.
            account_equity: Available equity for margin.

        Returns:
            ComplianceResult with approval or rejection.
        """
        side = order.get("side", "").lower()

        # Sell/close orders free up margin — no additional requirement.
        if side in ("sell", "short_cover", "close"):
            return ComplianceResult(approved=True)

        instrument_type = order.get("instrument_type", "equity")
        quantity = order.get("quantity", 0)
        price = order.get("price", 0.0)

        if instrument_type == "options":
            # Long options: full premium is the margin.
            margin_required = quantity * price * 100  # contracts * premium * 100
        else:
            # Long equity: 50% Reg-T.
            notional = quantity * price
            margin_required = notional * self.REG_T_INITIAL_MARGIN

        if margin_required > account_equity:
            return ComplianceResult(
                approved=False,
                reason=(
                    f"Insufficient margin: required ${margin_required:,.2f} "
                    f"({instrument_type}) but only ${account_equity:,.2f} available"
                ),
            )

        return ComplianceResult(approved=True)
