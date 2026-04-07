# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for tool implementations — response formatting, data access,
parsing, and serialization utilities.

Tool modules import from here rather than defining their own helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.config.settings import Settings
from quantstack.config.timeframes import Timeframe
from quantstack.data.pg_storage import PgDataStore
from quantstack.shared.serializers import serialize_for_json


# ---------------------------------------------------------------------------
# DataStore read/write separation
# ---------------------------------------------------------------------------


def _get_reader() -> PgDataStore:
    """Return a PgDataStore for read operations.

    PgDataStore is stateless — each method opens and closes a pg_conn()
    context manager, so there is no persistent connection to share or reuse.
    """
    return PgDataStore()


def set_shared_reader(reader: Any) -> None:
    """No-op: PgDataStore is stateless, no persistent connection to share."""


def _get_writer() -> PgDataStore:
    """Return a PgDataStore for write operations.  Caller may call .close() (no-op)."""
    return PgDataStore()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_TF_MAP = {
    "1m": Timeframe.M1,
    "m1": Timeframe.M1,
    "1min": Timeframe.M1,
    "5m": Timeframe.M5,
    "m5": Timeframe.M5,
    "5min": Timeframe.M5,
    "15m": Timeframe.M15,
    "m15": Timeframe.M15,
    "15min": Timeframe.M15,
    "30m": Timeframe.M30,
    "m30": Timeframe.M30,
    "30min": Timeframe.M30,
    "1h": Timeframe.H1,
    "h1": Timeframe.H1,
    "hourly": Timeframe.H1,
    "4h": Timeframe.H4,
    "h4": Timeframe.H4,
    "1d": Timeframe.D1,
    "d1": Timeframe.D1,
    "daily": Timeframe.D1,
    "d": Timeframe.D1,
    "1w": Timeframe.W1,
    "w1": Timeframe.W1,
    "weekly": Timeframe.W1,
    "w": Timeframe.W1,
}


def _parse_timeframe(tf_str: str) -> Timeframe:
    """Parse a timeframe string to ``Timeframe`` enum.  Defaults to D1."""
    return _TF_MAP.get(tf_str.lower(), Timeframe.D1)


# ---------------------------------------------------------------------------
# Serialization helpers (quantcore-specific wrappers)
# ---------------------------------------------------------------------------


def _dataframe_to_dict(df: pd.DataFrame, max_rows: int = 100) -> dict[str, Any]:
    """Convert DataFrame to serializable dict with truncation."""
    if df.empty:
        return {"data": [], "columns": [], "rows": 0}

    truncated = len(df) > max_rows
    if truncated:
        df = df.tail(max_rows)

    data = df.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "data": data.reset_index().to_dict(orient="records"),
        "columns": list(df.columns),
        "rows": len(df),
        "truncated": truncated,
    }


def _serialize_result(obj: Any) -> Any:
    """Serialize various result types to JSON-compatible format.

    Delegates to ``shared.serializers.serialize_for_json`` for most types,
    with special handling for the quantcore-specific DataFrame truncation format.
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return _dataframe_to_dict(obj)
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    # Fall through to the generic serializer for everything else
    return serialize_for_json(obj)


# ---------------------------------------------------------------------------
# Tool health tracking
# ---------------------------------------------------------------------------


def track_tool_health(
    tool_name: str,
    success: bool,
    latency_ms: float,
    error: str | None = None,
) -> None:
    """Record a tool invocation result in the tool_health table.

    Fire-and-forget: errors are logged but never propagated so that health
    tracking never breaks the tool call itself.

    Uses an upsert (ON CONFLICT) to atomically increment counters and update
    the running average latency without a separate SELECT.
    """
    try:
        from quantstack.db import pg_conn

        now = datetime.now(timezone.utc)
        success_inc = 1 if success else 0
        failure_inc = 0 if success else 1
        status = "active" if success else "error"
        last_error = error if not success else None

        with pg_conn() as conn:
            conn.execute(
                """
                INSERT INTO tool_health
                    (tool_name, invocation_count, success_count, failure_count,
                     avg_latency_ms, last_invoked, last_error, status, updated_at)
                VALUES (%s, 1, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tool_name) DO UPDATE SET
                    invocation_count = tool_health.invocation_count + 1,
                    success_count    = tool_health.success_count + EXCLUDED.success_count,
                    failure_count    = tool_health.failure_count + EXCLUDED.failure_count,
                    avg_latency_ms   = (tool_health.avg_latency_ms * tool_health.invocation_count + %s)
                                       / (tool_health.invocation_count + 1),
                    last_invoked     = EXCLUDED.last_invoked,
                    last_error       = COALESCE(EXCLUDED.last_error, tool_health.last_error),
                    status           = EXCLUDED.status,
                    updated_at       = EXCLUDED.updated_at
                """,
                [
                    tool_name, success_inc, failure_inc, latency_ms,
                    now, last_error, status, now,
                    latency_ms,
                ],
            )
    except Exception as exc:
        logger.warning("[ToolHealth] Failed to record health for %s: %s", tool_name, exc)
