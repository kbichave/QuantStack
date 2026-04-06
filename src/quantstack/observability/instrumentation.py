"""Centralized Langfuse instrumentation setup for LangGraph.

Uses Langfuse v2 direct API for trace creation. A ContextVar makes the
active trace available to agent_executor without threading it through
every function signature.
"""

import contextvars
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

_initialized = False
_langfuse_client = None

# ContextVar so any code running inside a graph cycle can access the trace.
_active_trace: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_active_trace", default=None
)


def get_active_trace() -> Any:
    """Return the current Langfuse trace (or None if outside a cycle)."""
    return _active_trace.get()


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
    Also sets the ContextVar so agent_executor and other code can
    access the trace via ``get_active_trace()``.
    """
    if _langfuse_client is None:
        yield None
        return

    trace = None
    token = None
    try:
        trace = _langfuse_client.trace(
            name=name,
            session_id=session_id,
            tags=tags,
        )
        token = _active_trace.set(trace)
    except Exception as exc:
        logger.debug("Langfuse trace creation failed: %s", exc)

    try:
        yield trace
    finally:
        if token is not None:
            _active_trace.reset(token)
        if trace is not None:
            try:
                trace.event(name="cycle_end")
            except Exception as exc:
                logger.debug("Langfuse cycle_end event failed: %s", exc)
        try:
            _langfuse_client.flush()
        except Exception as exc:
            logger.debug("Langfuse flush failed: %s", exc)


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
    except Exception as exc:
        logger.debug("Langfuse node span failed: %s", exc)


def log_llm_call(
    agent_name: str,
    model_name: str,
    input_messages: list[Any],
    output_content: str,
    duration_seconds: float,
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
) -> None:
    """Log an LLM call as a Langfuse GENERATION observation."""
    trace = _active_trace.get()
    if trace is None:
        return
    try:
        # Compact message representation for Langfuse input
        input_repr = []
        for m in input_messages[-3:]:  # last 3 messages to keep it small
            role = type(m).__name__.replace("Message", "").lower()
            content = m.content if hasattr(m, "content") else str(m)
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "..."
            input_repr.append({"role": role, "content": content})

        output_repr = {"content": output_content[:1000] if output_content else ""}
        if tool_calls:
            output_repr["tool_calls"] = [
                {"name": tc.get("name"), "args_keys": list(tc.get("args", {}).keys())}
                for tc in tool_calls[:5]
            ]

        trace.generation(
            name=f"llm/{agent_name}",
            model=model_name,
            input=input_repr,
            output=output_repr,
            usage=usage or {},
            metadata={"agent": agent_name, "duration_s": round(duration_seconds, 2)},
        )
    except Exception as exc:
        logger.debug("Langfuse generation log failed: %s", exc)


def log_tool_call(
    agent_name: str,
    tool_name: str,
    tool_args: dict,
    result: str,
    duration_seconds: float,
    success: bool = True,
) -> None:
    """Log a tool call as a Langfuse SPAN observation."""
    trace = _active_trace.get()
    if trace is None:
        return
    try:
        trace.span(
            name=f"tool/{tool_name}",
            input={"args": {k: str(v)[:200] for k, v in (tool_args or {}).items()}},
            output={"result": result[:500] if result else "", "success": success},
            metadata={"agent": agent_name, "duration_s": round(duration_seconds, 2)},
        )
    except Exception as exc:
        logger.debug("Langfuse tool span failed: %s", exc)
