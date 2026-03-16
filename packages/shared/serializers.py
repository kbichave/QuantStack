# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unified serialization for MCP tool responses.

Combines the previously duplicated ``_dc_to_dict`` (broker MCPs),
``_serialize`` (quant_pod), and ``_serialize_result`` (quantcore) into
a single recursive converter.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any


def serialize_for_json(obj: Any, *, max_rows: int = 500) -> Any:
    """Recursively convert *obj* to JSON-safe Python primitives.

    Handles (in priority order):
      - None
      - Pydantic v2 models (``model_dump``)
      - dataclasses (``asdict``)
      - ``datetime`` → ISO-8601 string
      - NumPy scalars/arrays (lazy import)
      - Pandas DataFrame/Series (lazy import, truncated to *max_rows*)
      - dicts, lists, tuples (recursive)
      - everything else → pass-through

    Args:
        obj: Value to serialize.
        max_rows: Maximum DataFrame rows before truncation (default 500).
    """
    if obj is None:
        return None

    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")

    # dataclass
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        return {k: serialize_for_json(v, max_rows=max_rows) for k, v in d.items()}

    # datetime
    if isinstance(obj, datetime):
        return obj.isoformat()

    # NumPy — lazy import so shared stays lightweight
    try:
        import numpy as np

        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    # Pandas — lazy import
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return obj.head(max_rows).to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except ImportError:
        pass

    # Containers — recurse
    if isinstance(obj, dict):
        return {k: serialize_for_json(v, max_rows=max_rows) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(v, max_rows=max_rows) for v in obj]

    return obj
