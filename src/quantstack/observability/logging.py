# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Structured JSON logging for QuantPod.

Adds a JSON sink to loguru alongside the default human-readable stderr output.
Every JSON record includes:
  - timestamp (ISO 8601 UTC)
  - level
  - message
  - module / function / line
  - trace_id (from TraceContext, if active)
  - trace metadata (symbol, run_id, etc.)
  - extra fields from loguru's `logger.bind()`

Output goes to a JSONL file (one JSON object per line) for easy ingestion
by ELK, Datadog, or grep + jq.

Usage:
    from quantstack.observability import configure_logging

    # At process startup (once):
    configure_logging(json_path="~/.quant_pod/logs/quantpod.jsonl")

    # Then use loguru normally:
    from loguru import logger
    logger.info("Trade executed", extra_field="value")
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from quantstack.observability.trace import TraceContext

# Sentinel to prevent double-configuration
_configured = False


def configure_logging(
    json_path: str | None = None,
    json_level: str = "DEBUG",
    stderr_level: str = "INFO",
    rotation: str = "50 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure structured JSON logging alongside human-readable stderr.

    Args:
        json_path: Path to the JSONL log file. Defaults to
                   ~/.quant_pod/logs/quantpod.jsonl. Set to None to
                   disable file logging (stderr only).
        json_level: Minimum level for JSON file output.
        stderr_level: Minimum level for stderr (human-readable) output.
        rotation: Log file rotation size.
        retention: How long to keep rotated log files.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # Remove loguru's default handler so we can replace it
    logger.remove()

    # Human-readable stderr (same format as before, with trace_id when available)
    logger.add(
        sys.stderr,
        level=stderr_level,
        format=_stderr_format,
        colorize=True,
    )

    # JSON file output
    if json_path is not None:
        resolved = Path(os.path.expanduser(json_path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(resolved),
            level=json_level,
            format=_json_format,
            rotation=rotation,
            retention=retention,
            serialize=False,  # We handle serialization in _json_format
        )
        logger.info(f"[OBSERVABILITY] JSON logging enabled → {resolved}")


def _stderr_format(record: dict) -> str:
    """Human-readable format with optional trace_id prefix."""
    trace_id = TraceContext.get_trace_id()
    trace_prefix = f"[{trace_id[:8]}] " if trace_id else ""

    level = record["level"].name
    time = record["time"].strftime("%H:%M:%S.%f")[:-3]
    message = record["message"]

    return (
        f"<level>{level:<8}</level> "
        f"<cyan>{time}</cyan> "
        f"{trace_prefix}"
        f"{message}\n"
    )


def _json_format(record: dict) -> str:
    """
    Serialize a loguru record to a single-line JSON string.

    Includes trace context, structured fields, and source location.
    """
    trace_id = TraceContext.get_trace_id()
    trace_meta = TraceContext.get_metadata()

    entry: dict[str, Any] = {
        "timestamp": record["time"].astimezone(timezone.utc).isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    # Trace context
    if trace_id:
        entry["trace_id"] = trace_id
    if trace_meta:
        entry["trace"] = trace_meta

    # Extra fields from logger.bind()
    extra = record.get("extra", {})
    if extra:
        entry["extra"] = {k: _safe_serialize(v) for k, v in extra.items()}

    # Exception info
    if record["exception"] is not None:
        exc = record["exception"]
        entry["exception"] = {
            "type": exc.type.__name__ if exc.type else None,
            "value": str(exc.value) if exc.value else None,
        }

    return json.dumps(entry, default=str) + "\n"


def _safe_serialize(value: Any) -> Any:
    """Convert non-serializable values to strings."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    return str(value)
