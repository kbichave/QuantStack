# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Fill legs recording and VWAP computation.

Every broker fill (paper or live) is decomposed into one or more fill legs
stored in the ``fill_legs`` table.  The summary row in ``fills`` is updated
with the VWAP across all legs after each new leg is recorded.

This module provides two entry points:

- ``record_fill_leg()`` — insert a single fill leg and return its sequence
  number (auto-incrementing per order_id).
- ``compute_fill_vwap()`` — quantity-weighted average price across all legs
  for a given order.

Both operate on the ``fill_legs`` table created by the execution layer
migration in ``db.py``.
"""

from __future__ import annotations

from quantstack.db import PgConnection


def record_fill_leg(
    conn: PgConnection,
    order_id: str,
    quantity: int,
    price: float,
    venue: str | None = None,
) -> int:
    """Insert a fill leg and return its leg_sequence number.

    The leg_sequence is auto-incremented per order_id: the first leg for an
    order gets sequence 1, the second gets 2, etc.  This is computed via a
    subquery (``COALESCE(MAX(leg_sequence), 0) + 1``) rather than a database
    sequence so that numbering is scoped to each order independently.

    Args:
        conn: Active PgConnection (caller manages transaction boundaries).
        order_id: The order this leg belongs to.
        quantity: Number of shares filled in this leg.
        price: Execution price for this leg.
        venue: Optional execution venue identifier (e.g. "paper", "alpaca").

    Returns:
        The assigned leg_sequence number (1-based).
    """
    conn.execute(
        """
        INSERT INTO fill_legs (order_id, leg_sequence, quantity, price, venue)
        VALUES (
            ?,
            COALESCE(
                (SELECT MAX(leg_sequence) FROM fill_legs WHERE order_id = ?),
                0
            ) + 1,
            ?, ?, ?
        )
        """,
        [order_id, order_id, quantity, price, venue],
    )

    row = conn.execute(
        "SELECT MAX(leg_sequence) FROM fill_legs WHERE order_id = ?",
        [order_id],
    ).fetchone()
    return int(row[0])


def compute_fill_vwap(conn: PgConnection, order_id: str) -> float:
    """Compute VWAP across all fill legs for an order.

    VWAP = SUM(quantity * price) / SUM(quantity)

    Args:
        conn: Active PgConnection.
        order_id: The order to compute VWAP for.

    Returns:
        The volume-weighted average price.

    Raises:
        ValueError: If no fill legs exist for the given order_id.
    """
    row = conn.execute(
        """
        SELECT SUM(quantity * price), SUM(quantity)
        FROM fill_legs
        WHERE order_id = ?
        """,
        [order_id],
    ).fetchone()

    if row is None or row[0] is None or row[1] is None or float(row[1]) == 0:
        raise ValueError(f"No fill legs found for order_id={order_id!r}")

    return float(row[0]) / float(row[1])
