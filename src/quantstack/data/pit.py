# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Point-in-time query helper — prevents look-ahead bias.

Ensures that research and signal generation only uses data that was
publicly available on a given date. Rows with NULL available_date are
excluded (conservative — prevents look-ahead from data without a known
publication date).

Usage:
    from quantstack.data.pit import pit_query

    rows = pit_query("financial_statements", "AAPL", as_of=date(2025, 6, 15))
"""

from __future__ import annotations

from datetime import date

from loguru import logger

from quantstack.db import pg_conn

_ALLOWED_TABLES = frozenset({
    "financial_statements",
    "earnings_calendar",
    "insider_trades",
    "institutional_holdings",
})


def pit_query(
    table: str,
    symbol: str,
    as_of: date,
    columns: str = "*",
    extra_where: str = "",
    extra_params: list | None = None,
    order_by: str = "available_date DESC",
    limit: int | None = None,
) -> list[dict]:
    """Query with point-in-time filtering: WHERE available_date <= as_of.

    Only returns rows where available_date is not NULL and <= as_of.
    Table name is validated against a whitelist to prevent SQL injection.

    Args:
        table: Table name (must be in _ALLOWED_TABLES)
        symbol: Stock symbol to filter on
        as_of: Only return data available on or before this date
        columns: Column selection (default "*")
        extra_where: Additional WHERE clause (AND-joined), use %s placeholders
        extra_params: Parameters for extra_where placeholders
        order_by: ORDER BY clause
        limit: Optional row limit

    Returns:
        List of row dicts, ordered by available_date descending by default.
    """
    if table not in _ALLOWED_TABLES:
        raise ValueError(
            f"pit_query: table '{table}' not in allowed list: {sorted(_ALLOWED_TABLES)}"
        )

    params: list = [symbol, as_of]
    where = "symbol = %s AND available_date IS NOT NULL AND available_date <= %s"

    if extra_where:
        where += f" AND ({extra_where})"
        if extra_params:
            params.extend(extra_params)

    sql = f"SELECT {columns} FROM {table} WHERE {where} ORDER BY {order_by}"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    with pg_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    logger.debug(
        "pit_query | table=%s symbol=%s as_of=%s rows=%d",
        table, symbol, as_of, len(rows),
    )
    return [dict(r) for r in rows]


def backfill_available_dates() -> dict[str, int]:
    """Backfill available_date for existing rows where it is NULL.

    Conservative defaults:
      - financial_statements: reported_date + 1 day
      - insider_trades: filed_date (SEC Form 4 is public immediately)
      - institutional_holdings: filed_date + 45 days (13F delay)
      - earnings_calendar: announcement_date

    Returns:
        Dict mapping table name to number of rows updated.
    """
    updates: dict[str, int] = {}

    backfill_rules = [
        (
            "financial_statements",
            """UPDATE financial_statements
               SET available_date = reported_date + INTERVAL '1 day'
               WHERE available_date IS NULL AND reported_date IS NOT NULL""",
        ),
        (
            "insider_trades",
            """UPDATE insider_trades
               SET available_date = filed_date
               WHERE available_date IS NULL AND filed_date IS NOT NULL""",
        ),
        (
            "institutional_holdings",
            """UPDATE institutional_holdings
               SET available_date = filed_date + INTERVAL '45 days'
               WHERE available_date IS NULL AND filed_date IS NOT NULL""",
        ),
        (
            "earnings_calendar",
            """UPDATE earnings_calendar
               SET available_date = announcement_date
               WHERE available_date IS NULL AND announcement_date IS NOT NULL""",
        ),
    ]

    with pg_conn() as conn:
        for table_name, sql in backfill_rules:
            try:
                result = conn.execute(sql)
                count = result.rowcount if result.rowcount else 0
                updates[table_name] = count
                if count > 0:
                    logger.info(
                        "pit_backfill | %s: updated %d rows", table_name, count
                    )
            except Exception as exc:
                logger.warning("pit_backfill | %s failed: %s", table_name, exc)
                updates[table_name] = 0

    return updates
