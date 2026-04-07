"""
Post-trade SEC compliance: Wash Sale tracking and Tax Lot management.

These hooks run AFTER a fill is confirmed and handle IRS reporting
obligations:

  - **Wash Sale Rule** (IRC Section 1091): If you sell a security at a
    loss and buy the same (or substantially identical) security within
    30 calendar days before or after the sale, the loss is disallowed
    for tax purposes and added to the cost basis of the replacement
    shares.

  - **Tax Lot Manager** (FIFO): Tracks individual purchase lots, matches
    sells to the oldest open lots first, and computes realized P&L per
    lot.  Wash sale adjustments are applied to lot cost basis.
"""

from __future__ import annotations

from datetime import date, datetime

from quantstack.db import PgConnection
from quantstack.execution.compliance.calendar import (
    trading_day_for,
    wash_sale_window_end,
)


# ---------------------------------------------------------------------------
# Wash Sale Tracker
# ---------------------------------------------------------------------------


class WashSaleTracker:
    """IRS wash sale detection and cost basis adjustment.

    Two-phase approach:

      Phase 1 (sell at loss): When a fill is a sell and realized P&L is
      negative, insert a ``pending_wash_losses`` record with a 30-day
      window.

      Phase 2 (buy): When a fill is a buy, check ``pending_wash_losses``
      for the same symbol with an active window.  If found, flag the
      wash sale and adjust cost basis on the replacement lot.
    """

    def on_fill(self, conn: PgConnection, fill_event: dict) -> dict:
        """Post-fill hook — detect and record wash sale activity.

        Args:
            fill_event: Dict with keys ``order_id``, ``symbol``, ``side``,
                ``quantity``, ``price``, ``timestamp``, and optionally
                ``realized_pnl`` (for sells).

        Returns:
            Dict summarising the action taken:
              ``{"action": "pending_loss_created", "loss_amount": ...}``
              ``{"action": "wash_sale_flagged", "disallowed_loss": ..., "adjusted_basis": ...}``
              ``{"action": "none"}``
        """
        side = fill_event.get("side", "").lower()
        symbol = fill_event["symbol"]
        fill_ts = fill_event.get("timestamp") or datetime.utcnow()
        fill_date = trading_day_for(fill_ts) if isinstance(fill_ts, datetime) else fill_ts

        if side == "sell":
            return self._handle_sell(conn, fill_event, symbol, fill_date)
        elif side == "buy":
            return self._handle_buy(conn, fill_event, symbol, fill_date)

        return {"action": "none"}

    def check_pending(self, conn: PgConnection, symbol: str) -> str | None:
        """Pre-trade advisory: warn if buying into a pending wash window.

        Returns a warning string if there is an unresolved pending loss
        for this symbol, or None.
        """
        today = date.today()
        row = conn.execute(
            "SELECT loss_amount, window_end FROM pending_wash_losses "
            "WHERE symbol = %s AND resolved = FALSE AND window_end >= %s "
            "ORDER BY window_end DESC LIMIT 1",
            (symbol, today),
        ).fetchone()

        if row is not None:
            return (
                f"Wash sale warning: pending loss ${row[0]:,.2f} on {symbol} "
                f"(window ends {row[1]})"
            )
        return None

    # ------------------------------------------------------------------
    # Phase 1: Sell at loss
    # ------------------------------------------------------------------

    def _handle_sell(
        self,
        conn: PgConnection,
        fill_event: dict,
        symbol: str,
        fill_date: date,
    ) -> dict:
        """If the sell realizes a loss, create a pending_wash_losses record."""
        realized_pnl = fill_event.get("realized_pnl", 0.0)
        if realized_pnl is None or realized_pnl >= 0:
            return {"action": "none"}

        loss_amount = abs(realized_pnl)
        window_end = wash_sale_window_end(fill_date)

        conn.execute(
            "INSERT INTO pending_wash_losses "
            "(symbol, loss_amount, sell_order_id, sell_date, window_end) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                symbol,
                loss_amount,
                fill_event.get("order_id", ""),
                fill_date,
                window_end,
            ),
        )

        return {"action": "pending_loss_created", "loss_amount": loss_amount}

    # ------------------------------------------------------------------
    # Phase 2: Buy within wash window
    # ------------------------------------------------------------------

    def _handle_buy(
        self,
        conn: PgConnection,
        fill_event: dict,
        symbol: str,
        fill_date: date,
    ) -> dict:
        """Check if this buy falls within a pending wash window."""
        row = conn.execute(
            "SELECT id, loss_amount, sell_date, window_end "
            "FROM pending_wash_losses "
            "WHERE symbol = %s AND resolved = FALSE AND window_end >= %s "
            "ORDER BY sell_date ASC LIMIT 1",
            (symbol, fill_date),
        ).fetchone()

        if row is None:
            return {"action": "none"}

        pending_id = row[0]
        disallowed_loss = row[1]
        sell_date = row[2]
        window_end = row[3]

        # Mark the pending loss as resolved.
        conn.execute(
            "UPDATE pending_wash_losses "
            "SET resolved = TRUE, resolved_by_order_id = %s "
            "WHERE id = %s",
            (fill_event.get("order_id", ""), pending_id),
        )

        # Compute adjusted cost basis: original purchase price + disallowed loss per share.
        buy_price = fill_event.get("price", 0.0)
        buy_qty = fill_event.get("quantity", 1)
        adjusted_basis = buy_price + (disallowed_loss / buy_qty)

        # Record the wash sale flag.
        conn.execute(
            "INSERT INTO wash_sale_flags "
            "(loss_trade_id, replacement_order_id, disallowed_loss, "
            "adjusted_cost_basis, wash_window_start, wash_window_end) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                pending_id,
                fill_event.get("order_id", ""),
                disallowed_loss,
                adjusted_basis,
                sell_date,
                window_end,
            ),
        )

        # Adjust cost basis on matching tax lot (if TaxLotManager created one).
        conn.execute(
            "UPDATE tax_lots SET "
            "cost_basis = %s, wash_sale_adjustment = %s "
            "WHERE order_id = %s AND symbol = %s AND status = 'open'",
            (
                adjusted_basis,
                disallowed_loss,
                fill_event.get("order_id", ""),
                symbol,
            ),
        )

        return {
            "action": "wash_sale_flagged",
            "disallowed_loss": disallowed_loss,
            "adjusted_basis": adjusted_basis,
        }


# ---------------------------------------------------------------------------
# Tax Lot Manager
# ---------------------------------------------------------------------------


class TaxLotManager:
    """FIFO tax lot tracking.

    Every buy creates a new lot.  Every sell consumes the oldest open
    lots first (FIFO), computing realized P&L per lot including any
    wash sale cost-basis adjustments.
    """

    def on_fill(self, conn: PgConnection, fill_event: dict) -> dict:
        """Post-fill hook — create or consume tax lots.

        Args:
            fill_event: Dict with ``order_id``, ``symbol``, ``side``,
                ``quantity``, ``price``, ``timestamp``.

        Returns:
            Dict summarising the action:
              Buy:  ``{"action": "lot_created", "lot_id": ...}``
              Sell: ``{"action": "lots_consumed", "realized_pnl": ..., "lots_closed": ...}``
        """
        side = fill_event.get("side", "").lower()
        symbol = fill_event["symbol"]

        if side == "buy":
            return self._create_lot(conn, fill_event, symbol)
        elif side == "sell":
            return self._consume_lots(conn, fill_event, symbol)

        return {"action": "none"}

    # ------------------------------------------------------------------
    # Buy → new lot
    # ------------------------------------------------------------------

    def _create_lot(
        self,
        conn: PgConnection,
        fill_event: dict,
        symbol: str,
    ) -> dict:
        fill_ts = fill_event.get("timestamp") or datetime.utcnow()
        acquired_date = (
            trading_day_for(fill_ts) if isinstance(fill_ts, datetime) else fill_ts
        )
        quantity = fill_event.get("quantity", 0)
        price = fill_event.get("price", 0.0)

        conn.execute(
            "INSERT INTO tax_lots "
            "(symbol, quantity, original_quantity, cost_basis, acquired_date, order_id) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (symbol, quantity, quantity, price, acquired_date, fill_event.get("order_id", "")),
        )

        # Retrieve the lot_id of the just-inserted row.
        row = conn.execute(
            "SELECT lot_id FROM tax_lots "
            "WHERE order_id = %s AND symbol = %s "
            "ORDER BY lot_id DESC LIMIT 1",
            (fill_event.get("order_id", ""), symbol),
        ).fetchone()

        lot_id = row[0] if row else None
        return {"action": "lot_created", "lot_id": lot_id}

    # ------------------------------------------------------------------
    # Sell → FIFO consumption
    # ------------------------------------------------------------------

    def _consume_lots(
        self,
        conn: PgConnection,
        fill_event: dict,
        symbol: str,
    ) -> dict:
        sell_qty = fill_event.get("quantity", 0)
        sell_price = fill_event.get("price", 0.0)
        fill_ts = fill_event.get("timestamp") or datetime.utcnow()
        close_date = (
            trading_day_for(fill_ts) if isinstance(fill_ts, datetime) else fill_ts
        )

        # Fetch open lots in FIFO order (oldest first).
        rows = conn.execute(
            "SELECT lot_id, quantity, cost_basis, wash_sale_adjustment "
            "FROM tax_lots "
            "WHERE symbol = %s AND status = 'open' "
            "ORDER BY acquired_date ASC, lot_id ASC",
            (symbol,),
        ).fetchall()

        remaining = sell_qty
        total_realized_pnl = 0.0
        lots_closed = 0

        for row in rows:
            if remaining <= 0:
                break

            lot_id = row[0]
            lot_qty = row[1]
            cost_basis = row[2]

            consumed = min(lot_qty, remaining)
            pnl = consumed * (sell_price - cost_basis)
            total_realized_pnl += pnl
            remaining -= consumed

            if consumed == lot_qty:
                # Lot fully consumed — close it.
                conn.execute(
                    "UPDATE tax_lots SET "
                    "quantity = 0, status = 'closed', closed_date = %s, "
                    "exit_price = %s, realized_pnl = %s "
                    "WHERE lot_id = %s",
                    (close_date, sell_price, pnl, lot_id),
                )
                lots_closed += 1
            else:
                # Partial consumption — reduce quantity, keep open.
                conn.execute(
                    "UPDATE tax_lots SET quantity = %s WHERE lot_id = %s",
                    (lot_qty - consumed, lot_id),
                )

        return {
            "action": "lots_consumed",
            "realized_pnl": total_realized_pnl,
            "lots_closed": lots_closed,
            "unfilled_quantity": remaining,
        }
