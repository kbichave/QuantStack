"""Centralized Langfuse instrumentation setup for LangGraph.

Langfuse v4 uses OpenTelemetry for automatic LLM call tracing.
Must be called once per process before any graph is invoked.
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

_initialized = False


def setup_instrumentation() -> None:
    """Initialize Langfuse OTEL instrumentation.

    Langfuse v4 auto-instruments LangChain, LiteLLM, and other LLM libs
    via OpenTelemetry. This captures all LLM calls (input/output/tokens/cost)
    automatically — no callback handlers needed.
    """
    global _initialized
    if _initialized:
        return

    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")

    if not secret_key or not public_key:
        raise ValueError(
            "LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY must be set. "
            "Generate keys at http://localhost:3100 and add to .env"
        )

    try:
        from langfuse import Langfuse

        # Auth check — validates keys against the Langfuse server
        lf = Langfuse()
        lf.auth_check()
        logger.info("Langfuse instrumentation initialized (v4 OTEL)")
        _initialized = True
    except Exception as exc:
        logger.warning("Langfuse auth check failed: %s — tracing may be degraded", exc)
        _initialized = True  # Don't retry, proceed without tracing


@contextmanager
def langfuse_trace_context(
    session_id: str,
    tags: list[str],
    name: str = "graph_cycle",
) -> Generator:
    """Create a Langfuse trace context for a graph invocation cycle.

    In v4, uses @observe-style context with start_observation/flush.
    All LLM calls within the context are auto-captured via OTEL.
    """
    try:
        from langfuse import Langfuse
        lf = Langfuse()
        # v4: use observe-style tracing
        trace_id = lf.create_trace_id()
        yield trace_id
        lf.flush()
    except Exception as exc:
        logger.debug("Langfuse trace context failed: %s", exc)
        yield None
