"""Centralized Langfuse instrumentation setup for LangGraph.

Uses Langfuse v2 direct API for trace creation. The CallbackHandler
approach is incompatible with langchain v1.x, so we create traces
manually and flush at cycle end.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

_initialized = False
_langfuse_client = None


def setup_instrumentation() -> None:
    """Initialize Langfuse and validate credentials."""
    global _initialized, _langfuse_client
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

        lf = Langfuse()
        lf.auth_check()
        _langfuse_client = lf
        logger.info("Langfuse instrumentation initialized (v2 direct API)")
        _initialized = True
    except Exception as exc:
        logger.warning("Langfuse auth check failed: %s — tracing disabled", exc)
        _initialized = True


@contextmanager
def langfuse_trace_context(
    session_id: str,
    tags: list[str],
    name: str = "graph_cycle",
) -> Generator[Any, None, None]:
    """Create a Langfuse trace for a graph invocation cycle.

    Yields a trace object (or None if Langfuse unavailable).
    Graph nodes and tool calls within the cycle are logged as events/spans.
    """
    if _langfuse_client is None:
        yield None
        return

    trace = None
    try:
        trace = _langfuse_client.trace(
            name=name,
            session_id=session_id,
            tags=tags,
        )
        yield trace
    except Exception as exc:
        logger.debug("Langfuse trace context failed: %s", exc)
        yield None
    finally:
        if trace is not None:
            try:
                trace.event(name="cycle_end")
            except Exception:
                pass
        try:
            _langfuse_client.flush()
        except Exception:
            pass


def log_node_execution(
    trace: Any,
    node_name: str,
    duration_seconds: float,
    metadata: dict | None = None,
) -> None:
    """Log a graph node execution as a Langfuse span on the current trace."""
    if trace is None:
        return
    try:
        trace.span(
            name=node_name,
            metadata=metadata or {},
            input={"duration_seconds": round(duration_seconds, 2)},
        )
    except Exception:
        pass
