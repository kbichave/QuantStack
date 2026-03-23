# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Langfuse tracing for prompt optimization loops.

Provides a thin wrapper around langfuse-python that traces:
- TextGrad backward passes (LLM calls, critiques, proposals)
- OPRO meta-prompt generation + scoring
- HypothesisJudge verdicts
- Research orchestrator steps

All tracing is best-effort: if langfuse is not installed or not
configured, operations proceed silently without tracing.

Setup:
    pip install langfuse
    export LANGFUSE_PUBLIC_KEY=pk-...
    export LANGFUSE_SECRET_KEY=sk-...
    export LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted

See: https://langfuse.com/docs/get-started
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

from loguru import logger

from langfuse import Langfuse

# Lazy-init singleton
_langfuse = None
_init_attempted = False


def _get_langfuse():
    """Lazy-init Langfuse client. Returns None if not configured."""
    global _langfuse, _init_attempted
    if _init_attempted:
        return _langfuse
    _init_attempted = True

    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        logger.debug("[Tracing] LANGFUSE_PUBLIC_KEY not set — tracing disabled")
        return None

    try:
        _langfuse = Langfuse()
        logger.info("[Tracing] Langfuse initialized")
        return _langfuse
    except Exception as exc:
        logger.warning(f"[Tracing] Langfuse init failed: {exc}")
        return None


class TracingSpan:
    """Thin wrapper around a langfuse span/generation. No-ops if langfuse is unavailable."""

    def __init__(self, span: Any = None):
        self._span = span

    def update(self, **kwargs) -> None:
        if self._span:
            try:
                self._span.update(**kwargs)
            except Exception:
                pass

    def end(self, **kwargs) -> None:
        if self._span:
            try:
                self._span.end(**kwargs)
            except Exception:
                pass

    def generation(self, **kwargs) -> "TracingSpan":
        if self._span:
            try:
                return TracingSpan(self._span.generation(**kwargs))
            except Exception:
                pass
        return TracingSpan(None)

    def span(self, **kwargs) -> "TracingSpan":
        if self._span:
            try:
                return TracingSpan(self._span.span(**kwargs))
            except Exception:
                pass
        return TracingSpan(None)


@contextmanager
def trace_optimization(
    name: str,
    metadata: dict | None = None,
) -> Generator[TracingSpan, None, None]:
    """Context manager for tracing an optimization operation.

    Usage:
        with trace_optimization("textgrad_backward", {"trade_id": 123}) as trace:
            trace.generation(name="critique", input=prompt, output=critique)
            ...
    """
    lf = _get_langfuse()
    if lf is None:
        yield TracingSpan(None)
        return

    try:
        trace = lf.trace(
            name=name,
            metadata=metadata or {},
            tags=["optimization"],
        )
        span = TracingSpan(trace)
        yield span
        trace.update(status_message="success")
    except Exception as exc:
        logger.debug(f"[Tracing] trace_optimization failed: {exc}")
        yield TracingSpan(None)


def trace_textgrad_critique(
    trade_id: int,
    node_name: str,
    prompt: str,
    critique: str,
    model: str,
) -> None:
    """Record a TextGrad critique as a langfuse generation."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        trace = lf.trace(
            name="textgrad_critique",
            metadata={"trade_id": trade_id, "node_name": node_name},
            tags=["textgrad", "optimization"],
        )
        trace.generation(
            name=f"critique_{node_name}",
            model=model,
            input=prompt,
            output=critique,
            metadata={"trade_id": trade_id},
        )
    except Exception as exc:
        logger.debug(f"[Tracing] TextGrad trace failed: {exc}")


def trace_opro_generation(
    node_name: str,
    meta_prompt: str,
    candidate_text: str,
    fitness: float,
    model: str,
) -> None:
    """Record an OPRO candidate generation as a langfuse generation."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        trace = lf.trace(
            name="opro_generation",
            metadata={"node_name": node_name, "fitness": fitness},
            tags=["opro", "optimization"],
        )
        trace.generation(
            name=f"opro_{node_name}",
            model=model,
            input=meta_prompt,
            output=candidate_text,
            metadata={"fitness": fitness},
        )
    except Exception as exc:
        logger.debug(f"[Tracing] OPRO trace failed: {exc}")


def trace_judge_verdict(
    hypothesis_name: str,
    approved: bool,
    score: float,
    flags: list[str],
) -> None:
    """Record a HypothesisJudge verdict as a langfuse event."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        trace = lf.trace(
            name="judge_verdict",
            metadata={
                "hypothesis": hypothesis_name,
                "approved": approved,
                "score": score,
                "flags": flags,
            },
            tags=["judge", "optimization"],
        )
        trace.update(
            status_message="approved" if approved else "rejected",
        )
    except Exception as exc:
        logger.debug(f"[Tracing] Judge trace failed: {exc}")


def trace_research_critique(
    symbol: str,
    sharpe: float,
    node_name: str,
    critique: str,
    model: str,
) -> None:
    """Record a research chain critique as a langfuse generation."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        trace = lf.trace(
            name="research_critique",
            metadata={"symbol": symbol, "sharpe": sharpe, "node_name": node_name},
            tags=["textgrad", "research", "optimization"],
        )
        trace.generation(
            name=f"research_critique_{node_name}",
            model=model,
            input=f"Failed research: {symbol}, Sharpe={sharpe:.2f}",
            output=critique,
        )
    except Exception as exc:
        logger.debug(f"[Tracing] Research critique trace failed: {exc}")


def flush() -> None:
    """Flush any pending langfuse events. Call at process exit."""
    lf = _get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass
