"""Weekend parallel research coordinator.

Launches 4 research streams in parallel via LangGraph's Send API during the
weekend window (Friday 20:00 ET to Monday 04:00 ET).  Each stream runs
independently with its own error isolation.  Results merge via an
``operator.add`` reducer on ``weekend_research_results``, then a synthesis
node consolidates findings into prioritized research tasks.

Budget: $50 per weekend cycle, split across streams.
"""

from __future__ import annotations

import logging
import operator
from datetime import datetime, time
from typing import Annotated
from zoneinfo import ZoneInfo

from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from quantstack.research.streams.cross_asset_signals import run_cross_asset_signals
from quantstack.research.streams.factor_mining import run_factor_mining
from quantstack.research.streams.portfolio_construction import run_portfolio_construction
from quantstack.research.streams.regime_research import run_regime_research

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# Weekend window boundaries
_FRIDAY_START = time(20, 0)   # Friday 20:00 ET
_MONDAY_END = time(4, 0)      # Monday 04:00 ET

_STREAM_CONFIGS = [
    {"stream_name": "factor_mining", "runner": run_factor_mining},
    {"stream_name": "regime_research", "runner": run_regime_research},
    {"stream_name": "cross_asset_signals", "runner": run_cross_asset_signals},
    {"stream_name": "portfolio_construction", "runner": run_portfolio_construction},
]


# ---------------------------------------------------------------------------
# State models
# ---------------------------------------------------------------------------

class StreamResult(BaseModel):
    """Output from a single research stream."""

    stream_name: str
    findings: list[dict] = Field(default_factory=list)
    experiments_run: int = 0
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)


class WeekendResearchState(BaseModel):
    """Top-level state for the weekend research coordinator graph."""

    start_time: str = ""
    end_time: str = ""
    budget_remaining: float = 50.0
    weekend_research_results: Annotated[list[dict], operator.add] = []
    synthesis_tasks: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Time window helpers
# ---------------------------------------------------------------------------

def is_weekend_window(now: datetime | None = None) -> bool:
    """Return True if *now* falls within the weekend research window.

    Window: Friday 20:00 ET through Monday 04:00 ET (inclusive of boundaries).
    """
    if now is None:
        now = datetime.now(ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ET)

    weekday = now.weekday()  # Mon=0 .. Sun=6
    t = now.time()

    # Friday (4) at or after 20:00
    if weekday == 4 and t >= _FRIDAY_START:
        return True
    # All of Saturday (5) and Sunday (6)
    if weekday in (5, 6):
        return True
    # Monday (0) before 04:00
    if weekday == 0 and t < _MONDAY_END:
        return True
    return False


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def fan_out_streams(state: WeekendResearchState) -> list[Send]:
    """Return 4 Send() calls, one per research stream."""
    sends = []
    for cfg in _STREAM_CONFIGS:
        sends.append(
            Send(
                "run_stream",
                {"stream_name": cfg["stream_name"], "budget_remaining": state.budget_remaining},
            )
        )
    return sends


def run_stream(state: dict) -> dict:
    """Dispatch to the correct stream runner based on stream_name.

    Catches exceptions so one stream failure does not crash the graph.
    """
    stream_name = state.get("stream_name", "")
    runner_map = {cfg["stream_name"]: cfg["runner"] for cfg in _STREAM_CONFIGS}
    runner = runner_map.get(stream_name)

    if runner is None:
        return {
            "weekend_research_results": [{
                "stream_name": stream_name,
                "findings": [],
                "experiments_run": 0,
                "cost_usd": 0.0,
                "errors": [f"Unknown stream: {stream_name}"],
            }],
        }

    try:
        return runner(state)
    except Exception as exc:
        logger.exception("Stream %s failed", stream_name)
        return {
            "weekend_research_results": [{
                "stream_name": stream_name,
                "findings": [],
                "experiments_run": 0,
                "cost_usd": 0.0,
                "errors": [f"{stream_name} crashed: {exc}"],
            }],
        }


def synthesis_node(state: WeekendResearchState) -> dict:
    """Review all stream results and produce prioritized research tasks.

    In production this will call a Sonnet LLM to rank and synthesize.
    For now it deterministically collects findings and creates tasks.
    """
    results = state.weekend_research_results
    tasks: list[dict] = []

    for result in results:
        sr = StreamResult(**result)
        for finding in sr.findings:
            tasks.append({
                "source_stream": sr.stream_name,
                "finding": finding,
                "priority": "high" if finding.get("status") == "pending_validation" else "low",
                "action": "validate_and_backtest",
            })

    total_cost = sum(StreamResult(**r).cost_usd for r in results)
    total_errors = []
    for r in results:
        total_errors.extend(StreamResult(**r).errors)

    if total_errors:
        logger.warning("Weekend research had %d errors: %s", len(total_errors), total_errors)

    return {
        "synthesis_tasks": tasks,
        "budget_remaining": state.budget_remaining - total_cost,
        "end_time": datetime.now(ET).isoformat(),
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_weekend_graph() -> StateGraph:
    """Construct the weekend research StateGraph.

    Topology::

        fan_out_streams -> [run_stream x4] -> synthesis_node -> END

    ``fan_out_streams`` is a conditional edge that emits 4 ``Send()`` messages.
    All stream results merge into ``weekend_research_results`` via the
    ``operator.add`` reducer, then ``synthesis_node`` consolidates.
    """
    graph = StateGraph(WeekendResearchState)

    graph.add_node("run_stream", run_stream)
    graph.add_node("synthesis", synthesis_node)

    # Entry: fan out to 4 parallel streams
    graph.set_conditional_entry_point(fan_out_streams)

    # After all streams complete, synthesize
    graph.add_edge("run_stream", "synthesis")
    graph.add_edge("synthesis", END)

    return graph
