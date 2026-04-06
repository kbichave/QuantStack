# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Langfuse tracing for QuantStack.

Provides:
- Lazy-init Langfuse singleton
- TracingSpan wrapper (no-ops if Langfuse unavailable)
- Optimization trace helpers (TextGrad, OPRO, Judge, Research)
- Business event trace helpers (provider failover, strategy lifecycle,
  self-healing, capital allocation, safety boundary)

All tracing is best-effort: if Langfuse is not configured,
operations proceed silently without tracing.

Setup:
    pip install langfuse
    export LANGFUSE_PUBLIC_KEY=pk-...
    export LANGFUSE_SECRET_KEY=sk-...
    export LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

from loguru import logger

from langfuse import Langfuse

_std_logger = logging.getLogger(__name__)

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
        _langfuse.auth_check()
        logger.info("[Tracing] Langfuse initialized")
        return _langfuse
    except Exception as exc:
        logger.warning(f"[Tracing] Langfuse init failed: {exc}")
        return None


def _create_event(name: str, metadata: dict | None = None, tags: list[str] | None = None) -> None:
    """Create a Langfuse event (v4 API). Best-effort, never raises."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.create_event(name=name, metadata=metadata or {})
    except Exception:
        _std_logger.debug("Failed to create Langfuse event: %s", name, exc_info=True)


def shutdown() -> None:
    """Flush pending events and shut down the Langfuse client.

    Stops the background worker thread. Call at process exit.
    """
    lf = _get_langfuse()
    if lf:
        try:
            lf.shutdown()
        except Exception as exc:
            _std_logger.debug("[Tracing] shutdown failed: %s", exc)


class TracingSpan:
    """Thin wrapper around a langfuse span/generation. No-ops if langfuse is unavailable."""

    def __init__(self, span: Any = None):
        self._span = span

    def update(self, **kwargs) -> None:
        if self._span:
            try:
                self._span.update(**kwargs)
            except Exception as exc:
                _std_logger.debug("[Tracing] span update failed: %s", exc)

    def end(self, **kwargs) -> None:
        if self._span:
            try:
                self._span.end(**kwargs)
            except Exception as exc:
                _std_logger.debug("[Tracing] span end failed: %s", exc)

    def generation(self, **kwargs) -> "TracingSpan":
        if self._span:
            try:
                return TracingSpan(self._span.generation(**kwargs))
            except Exception as exc:
                _std_logger.debug("[Tracing] span generation failed: %s", exc)
        return TracingSpan(None)

    def span(self, **kwargs) -> "TracingSpan":
        if self._span:
            try:
                return TracingSpan(self._span.span(**kwargs))
            except Exception as exc:
                _std_logger.debug("[Tracing] span creation failed: %s", exc)
        return TracingSpan(None)


@contextmanager
def trace_optimization(
    name: str,
    metadata: dict | None = None,
) -> Generator[TracingSpan, None, None]:
    """Context manager for tracing an optimization operation."""
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
        except Exception as exc:
            _std_logger.debug("[Tracing] flush failed: %s", exc)


# --- Business event trace helpers ---


def trace_provider_failover(
    original_provider: str,
    fallback_provider: str,
    error: str,
    tier: str,
) -> None:
    """Log a provider failover event to Langfuse."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="provider_failover",
            metadata={
                "original_provider": original_provider,
                "fallback_provider": fallback_provider,
                "error": error,
                "tier": tier,
            },
            tags=["failover", "llm"],
        )
    except Exception:
        _std_logger.debug("Failed to trace provider failover", exc_info=True)


def trace_strategy_lifecycle(
    strategy_id: str,
    action: str,
    reasoning: str,
    evidence: dict[str, Any],
) -> None:
    """Log a strategy promotion/retirement/extension decision."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="strategy_lifecycle",
            metadata={
                "strategy_id": strategy_id,
                "action": action,
                "reasoning": reasoning,
            },
            input=evidence,
            tags=["strategy", action],
        )
    except Exception:
        _std_logger.debug("Failed to trace strategy lifecycle", exc_info=True)


def trace_self_healing_event(
    event_type: str,
    details: dict[str, Any],
) -> None:
    """Log a self-healing event (watchdog trigger, restart, etc.)."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="self_healing",
            metadata={"event_type": event_type, **details},
            tags=["self_healing", event_type],
        )
    except Exception:
        _std_logger.debug("Failed to trace self-healing event", exc_info=True)


def trace_capital_allocation(
    symbol: str,
    recommended_size_pct: float,
    reasoning: str,
    portfolio_context: dict[str, Any],
) -> None:
    """Log a capital allocation / position sizing decision."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="capital_allocation",
            metadata={
                "symbol": symbol,
                "recommended_size_pct": recommended_size_pct,
                "reasoning": reasoning,
            },
            input=portfolio_context,
            tags=["risk", "allocation"],
        )
    except Exception:
        _std_logger.debug("Failed to trace capital allocation", exc_info=True)


def trace_safety_boundary_trigger(
    symbol: str,
    llm_recommendation: dict[str, Any],
    gate_limit: str,
    gate_value: float,
) -> None:
    """Log when the programmatic safety gate overrides an LLM decision."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="safety_boundary_trigger",
            metadata={
                "symbol": symbol,
                "llm_recommendation": llm_recommendation,
                "gate_limit": gate_limit,
                "gate_value": gate_value,
            },
            tags=["safety", "override"],
        )
        _std_logger.warning(
            "Safety gate triggered for %s: %s exceeded (%.4f)",
            symbol, gate_limit, gate_value,
        )
    except Exception:
        _std_logger.debug("Failed to trace safety boundary trigger", exc_info=True)


def trace_agent_decision(
    agent_name: str,
    decision_type: str,
    symbol: str | None,
    verdict: str,
    reasoning: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a Claude agent decision to Langfuse.

    Generic trace for any agent decision not covered by the specialized
    helpers (capital_allocation, strategy_lifecycle, etc.).
    """
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name=f"agent_decision:{agent_name}",
            metadata={
                "agent": agent_name,
                "decision_type": decision_type,
                "symbol": symbol,
                "verdict": verdict,
                "reasoning": reasoning,
                **(metadata or {}),
            },
            tags=[agent_name, decision_type],
        )
    except Exception:
        _std_logger.debug("Failed to trace agent decision for %s", agent_name, exc_info=True)


# ---------------------------------------------------------------------------
# Tool Search tracing helpers
# ---------------------------------------------------------------------------

def trace_tool_search_event(
    agent_name: str,
    query: str,
    result_count: int,
    tool_names_returned: list[str],
    latency_ms: float | None = None,
) -> None:
    """Log a BM25 tool search event to LangFuse."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="tool_search:search",
            metadata={
                "agent": agent_name,
                "query": query,
                "result_count": result_count,
                "tool_names_returned": tool_names_returned,
                "latency_ms": latency_ms,
            },
            tags=["tool_search", "search"],
        )
    except Exception:
        _std_logger.debug("Failed to trace tool search event for %s", agent_name, exc_info=True)


def trace_tool_discovery_event(
    agent_name: str,
    tool_name: str,
    search_query: str,
) -> None:
    """Log when a deferred tool is discovered via search and subsequently called."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="tool_search:discovery",
            metadata={
                "agent": agent_name,
                "tool_name": tool_name,
                "search_query": search_query,
            },
            tags=["tool_search", "discovery"],
        )
    except Exception:
        _std_logger.debug("Failed to trace tool discovery for %s", agent_name, exc_info=True)


def trace_tool_search_miss_event(
    agent_name: str,
    query: str,
    reason: str,
) -> None:
    """Log when a tool search returns no useful results."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="tool_search:miss",
            metadata={
                "agent": agent_name,
                "query": query,
                "reason": reason,
            },
            tags=["tool_search", "miss"],
        )
    except Exception:
        _std_logger.debug("Failed to trace tool search miss for %s", agent_name, exc_info=True)


# ---------------------------------------------------------------------------
# Work-item trace helpers (WI-1, WI-5, WI-6, WI-7, WI-8)
# ---------------------------------------------------------------------------


def trace_quality_evaluation(
    trade_id: int,
    scores: dict,
    model_used: str,
    latency_ms: float,
) -> None:
    """Trace a trade quality evaluation from WI-1."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="trade_quality_evaluation",
            metadata={
                "trade_id": trade_id,
                "overall_score": scores.get("overall_score"),
                "model_used": model_used,
                "latency_ms": latency_ms,
            },
            tags=["quality", "evaluation"],
        )
    except Exception:
        _std_logger.debug("Failed to trace quality evaluation", exc_info=True)


def trace_thinking_enabled(
    agent_name: str,
    thinking_config: dict,
    model_id: str,
) -> None:
    """Trace when extended thinking is activated for an agent (WI-5)."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="thinking_enabled",
            metadata={
                "agent_name": agent_name,
                "thinking_config": thinking_config,
                "model_id": model_id,
            },
            tags=["thinking", "llm"],
        )
    except Exception:
        _std_logger.debug("Failed to trace thinking_enabled", exc_info=True)


def trace_parallel_branch_timing(
    graph_name: str,
    branch_name: str,
    duration_seconds: float,
) -> None:
    """Trace parallel branch execution duration (WI-6)."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="parallel_branch_timing",
            metadata={
                "graph_name": graph_name,
                "branch_name": branch_name,
                "duration_seconds": duration_seconds,
            },
            tags=["parallel", "timing"],
        )
    except Exception:
        _std_logger.debug("Failed to trace parallel branch timing", exc_info=True)


def trace_fanout_worker(
    symbol: str,
    worker_index: int,
    duration_seconds: float,
    success: bool,
    error: str | None = None,
) -> None:
    """Trace per-symbol worker execution in research fan-out (WI-7)."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="fanout_worker",
            metadata={
                "symbol": symbol,
                "worker_index": worker_index,
                "duration_seconds": duration_seconds,
                "success": success,
                "error": error,
            },
            tags=["fanout", "research"],
        )
    except Exception:
        _std_logger.debug("Failed to trace fanout worker", exc_info=True)


def trace_hypothesis_loop(
    loop_count: int,
    final_confidence: float,
    max_attempts_hit: bool,
) -> None:
    """Trace hypothesis self-critique loop metrics (WI-8)."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="hypothesis_loop",
            metadata={
                "loop_count": loop_count,
                "final_confidence": final_confidence,
                "max_attempts_hit": max_attempts_hit,
            },
            tags=["hypothesis", "self_critique"],
        )
    except Exception:
        _std_logger.debug("Failed to trace hypothesis loop", exc_info=True)


def trace_tool_search_fallback(
    agent_name: str,
    error: str,
    tools_loaded: int,
) -> None:
    """Log when tool search falls back to full loading."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="tool_search:fallback",
            metadata={
                "agent": agent_name,
                "error": error,
                "tools_loaded": tools_loaded,
            },
            tags=["tool_search", "fallback"],
        )
    except Exception:
        _std_logger.debug("Failed to trace tool search fallback for %s", agent_name, exc_info=True)
