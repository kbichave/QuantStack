# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tool search accuracy metrics.

Computes aggregate metrics from LangFuse tool search traces.
Results are stored in PostgreSQL for dashboarding and trend analysis.

Start with ad-hoc computation; automate into a daily cron only if
the queries are run frequently.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from quantstack.db import db_conn
from quantstack.observability.tracing import _get_langfuse

logger = logging.getLogger(__name__)


@dataclass
class ToolSearchMetrics:
    """Aggregate tool search effectiveness metrics."""

    search_hit_rate: float          # searches that led to a tool call / total searches
    discovery_accuracy: float       # discovered tools actually used / total discovered
    fallback_rate: float            # agent turns in fallback mode / total agent turns
    top_missed_tools: list[str]     # tools frequently searched but not found
    total_searches: int
    total_discoveries: int
    total_misses: int
    total_fallbacks: int
    time_window: timedelta


def _zeroed_metrics(time_window: timedelta) -> ToolSearchMetrics:
    """Return zeroed metrics for empty trace sets."""
    return ToolSearchMetrics(
        search_hit_rate=0.0,
        discovery_accuracy=0.0,
        fallback_rate=0.0,
        top_missed_tools=[],
        total_searches=0,
        total_discoveries=0,
        total_misses=0,
        total_fallbacks=0,
        time_window=time_window,
    )


def compute_tool_search_accuracy(time_window: timedelta) -> ToolSearchMetrics:
    """Analyze tool search effectiveness from LangFuse traces.

    Queries LangFuse for traces tagged with 'tool_search' within the
    given time window. Computes hit rate, discovery accuracy, fallback
    rate, and identifies frequently-missed tools.

    Returns zeroed metrics if no traces exist for the window.
    """
    lf = _get_langfuse()
    if lf is None:
        logger.debug("LangFuse unavailable — returning zeroed metrics")
        return _zeroed_metrics(time_window)

    cutoff = datetime.now(timezone.utc) - time_window

    try:
        # Fetch traces tagged with tool_search within the time window.
        # LangFuse SDK fetch_traces returns a paginated result.
        search_traces = _fetch_traces_by_tag(lf, "search", cutoff)
        discovery_traces = _fetch_traces_by_tag(lf, "discovery", cutoff)
        miss_traces = _fetch_traces_by_tag(lf, "miss", cutoff)
        fallback_traces = _fetch_traces_by_tag(lf, "fallback", cutoff)
    except Exception as exc:
        logger.warning("Failed to fetch LangFuse traces: %s", exc)
        return _zeroed_metrics(time_window)

    total_searches = len(search_traces)
    total_discoveries = len(discovery_traces)
    total_misses = len(miss_traces)
    total_fallbacks = len(fallback_traces)

    if total_searches == 0:
        return _zeroed_metrics(time_window)

    # search_hit_rate: searches that led to at least one discovery / total searches
    search_hit_rate = min(total_discoveries / total_searches, 1.0) if total_searches > 0 else 0.0

    # discovery_accuracy: discovered tools actually used / total discovered
    # A discovery is "used" if a corresponding tool_call trace exists for that tool.
    # For now, count all discoveries as used (the trace is only emitted on actual call).
    discovery_accuracy = 1.0 if total_discoveries > 0 else 0.0

    # fallback_rate: fallback events / (searches + fallbacks) as proxy for total agent turns
    total_agent_turns = total_searches + total_fallbacks
    fallback_rate = total_fallbacks / total_agent_turns if total_agent_turns > 0 else 0.0

    # top_missed_tools: extract tool names from miss trace metadata
    missed_counter: Counter[str] = Counter()
    for trace in miss_traces:
        metadata = getattr(trace, "metadata", {}) or {}
        query = metadata.get("query", "unknown")
        missed_counter[query] += 1
    top_missed_tools = [tool for tool, _ in missed_counter.most_common(10)]

    return ToolSearchMetrics(
        search_hit_rate=round(search_hit_rate, 4),
        discovery_accuracy=round(discovery_accuracy, 4),
        fallback_rate=round(fallback_rate, 4),
        top_missed_tools=top_missed_tools,
        total_searches=total_searches,
        total_discoveries=total_discoveries,
        total_misses=total_misses,
        total_fallbacks=total_fallbacks,
        time_window=time_window,
    )


def _fetch_traces_by_tag(lf: Any, tag: str, cutoff: datetime) -> list:
    """Fetch LangFuse traces with both 'tool_search' and the given sub-tag."""
    try:
        result = lf.fetch_traces(
            tags=["tool_search", tag],
            from_timestamp=cutoff,
        )
        return result.data if hasattr(result, "data") else []
    except Exception as exc:
        logger.debug("Failed to fetch traces for tag '%s': %s", tag, exc)
        return []


def store_metrics(metrics: ToolSearchMetrics) -> None:
    """Persist computed metrics to PostgreSQL."""
    time_window_h = int(metrics.time_window.total_seconds() / 3600)
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO tool_search_metrics (
                time_window_h, search_hit_rate, discovery_accuracy,
                fallback_rate, top_missed_tools,
                total_searches, total_discoveries, total_misses, total_fallbacks
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                time_window_h,
                metrics.search_hit_rate,
                metrics.discovery_accuracy,
                metrics.fallback_rate,
                metrics.top_missed_tools,
                metrics.total_searches,
                metrics.total_discoveries,
                metrics.total_misses,
                metrics.total_fallbacks,
            ),
        )
