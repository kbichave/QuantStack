# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Observability layer — structured logging, trace IDs, and JSON output.

Provides:
  - TraceContext: thread-local trace ID propagation (signal → decision → order → fill)
  - configure_logging(): JSON structured logging alongside human-readable output
  - Loguru integration: adds trace_id to all log records automatically

Usage (at process startup):
    from quantstack.observability import configure_logging, TraceContext

    configure_logging(json_path="~/.quantstack/logs/quantstack.jsonl")

    # At the start of a trading pass:
    with TraceContext.new_trace(symbol="SPY") as trace_id:
        logger.info("Starting analysis")   # trace_id is auto-injected
        # ... signal → decision → order → fill
        # All log lines within this block share the same trace_id

    # Or manually:
    TraceContext.set("abc123", symbol="SPY")
    logger.info("manual trace")
    TraceContext.clear()
"""

from quantstack.observability.logging import configure_logging
from quantstack.observability.trace import TraceContext

__all__ = ["configure_logging", "TraceContext"]
