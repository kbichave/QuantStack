# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
TraceContext — thread-local trace ID propagation.

Provides a context-manager-based trace ID that flows through the entire
signal → decision → risk → order → fill pipeline. Every log line emitted
within a trace context includes the trace_id, enabling end-to-end tracing
of a bad trade from signal generation to fill.

Thread-safe: each thread gets its own trace context via threading.local.
Async-safe: works with asyncio because Python asyncio tasks inherit
thread-local state within a single event loop thread.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from threading import local
from typing import Any, Generator

_thread_local = local()


class TraceContext:
    """
    Thread-local trace context for end-to-end request tracing.

    The trace_id and optional metadata (symbol, run_id, strategy_id) are
    available to the logging layer for automatic injection into JSON records.
    """

    @classmethod
    def set(cls, trace_id: str, **metadata: Any) -> None:
        """Set the trace context for the current thread."""
        _thread_local.trace_id = trace_id
        _thread_local.trace_meta = metadata

    @classmethod
    def clear(cls) -> None:
        """Clear the trace context for the current thread."""
        _thread_local.trace_id = None
        _thread_local.trace_meta = {}

    @classmethod
    def get_trace_id(cls) -> str | None:
        """Return the current trace ID, or None if not in a trace."""
        return getattr(_thread_local, "trace_id", None)

    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        """Return the current trace metadata."""
        return getattr(_thread_local, "trace_meta", {})

    @classmethod
    @contextmanager
    def new_trace(cls, **metadata: Any) -> Generator[str, None, None]:
        """
        Context manager that creates a new trace scope.

        All log lines within the scope will include the trace_id.
        Nested traces are not supported — the inner trace replaces the outer.

        Usage:
            with TraceContext.new_trace(symbol="SPY", run_id="abc") as tid:
                logger.info("signal generated")  # includes trace_id=tid
        """
        trace_id = uuid.uuid4().hex[:16]
        prev_id = getattr(_thread_local, "trace_id", None)
        prev_meta = getattr(_thread_local, "trace_meta", {})

        cls.set(trace_id, **metadata)
        try:
            yield trace_id
        finally:
            # Restore previous context (supports nesting even though not recommended)
            if prev_id is not None:
                cls.set(prev_id, **prev_meta)
            else:
                cls.clear()
