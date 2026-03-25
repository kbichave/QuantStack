# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP helpers — shared state and utilities used by all tool modules.

Tool modules import from here rather than defining their own helpers.
The ``mcp`` singleton and ``lifespan`` live in ``server.py``; this module
holds only stateless helpers and the read/write DataStore accessors.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.config.settings import Settings
from quantstack.config.timeframes import Timeframe
from quantstack.data.pg_storage import PgDataStore
from quantstack.shared.serializers import serialize_for_json


@dataclass
class ServerContext:
    """Shared context for MCP server, set during lifespan."""

    settings: Settings
    data_store: PgDataStore | None = None
    feature_factory: Any = None
    data_registry: Any = None


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
