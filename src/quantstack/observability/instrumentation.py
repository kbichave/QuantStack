"""Centralized Langfuse instrumentation setup for LangGraph.

Must be called once per process before any graph is invoked.
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


def setup_instrumentation() -> None:
    """Initialize Langfuse instrumentation.

    Validates env vars and pre-warms the Langfuse client singleton.
    Raises ValueError if required env vars are missing.
    """
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")

    if not secret_key:
        raise ValueError(
            "LANGFUSE_SECRET_KEY is not set. "
            "Generate keys at http://localhost:3000 and add to .env"
        )
    if not public_key:
        raise ValueError(
            "LANGFUSE_PUBLIC_KEY is not set. "
            "Generate keys at http://localhost:3000 and add to .env"
        )

    from quantstack.observability.tracing import _get_langfuse
    lf = _get_langfuse()
    if lf is not None:
        logger.info("Langfuse instrumentation initialized")
    else:
        logger.warning("Langfuse client initialization failed — tracing disabled")


@contextmanager
def langfuse_trace_context(
    session_id: str,
    tags: list[str],
    name: str = "graph_cycle",
) -> Generator:
    """Create a Langfuse trace context for a graph invocation cycle.

    Usage:
        with langfuse_trace_context("trading-2026-04-02-cycle-3", ["trading"]) as trace:
            # trace is a Langfuse trace object (or None if tracing disabled)
            result = await graph.ainvoke(state)

    Args:
        session_id: Groups related traces (e.g., "trading-2026-04-02-cycle-3").
        tags: Categorization tags (e.g., ["trading", "paper"]).
        name: Trace name (e.g., "trading_cycle", "research_cycle").

    Yields:
        A Langfuse trace object, or None if tracing is disabled.
    """
    from quantstack.observability.tracing import _get_langfuse
    lf = _get_langfuse()
    if lf is None:
        yield None
        return

    try:
        trace = lf.trace(
            name=name,
            session_id=session_id,
            tags=tags,
        )
        yield trace
        trace.update(status_message="success")
    except Exception as exc:
        logger.debug("Langfuse trace context failed: %s", exc)
        yield None
