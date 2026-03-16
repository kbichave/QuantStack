# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Enhancement 1 — Granular IC Access tools.

Tools:
  - list_ics          — catalog of all 13 ICs and 6 pod managers
  - run_ic            — run a single IC in isolation (2-agent minimal crew)
  - run_pod           — run a pod + its ICs, or pod manager over pre-computed IC outputs
  - run_crew_subset   — run custom IC subset through pod managers to assistant
  - get_last_ic_output — retrieve cached IC output from last crew run (30-min TTL)
"""

import asyncio
import time
from typing import Any

from loguru import logger

from quant_pod.mcp.server import mcp
from quant_pod.mcp._state import (
    require_ctx,
    require_live_db,
    live_db_or_error,
    _serialize,
    ic_cache_set,
    ic_cache_get,
    populate_ic_cache_from_result,
)

# =============================================================================
# Constants
# =============================================================================

# Human-readable descriptions for each IC (used by list_ics and IDEs)
_IC_DESCRIPTIONS: dict[str, str] = {
    "data_ingestion_ic": "Fetch OHLCV market data; assess data quality and coverage",
    "market_snapshot_ic": "Current price, volume, key indicator snapshot",
    "regime_detector_ic": "Market regime classification (trend + volatility)",
    "trend_momentum_ic": "RSI, MACD, ADX, SMA — trend and momentum metrics",
    "volatility_ic": "ATR, Bollinger Bands, volatility regime",
    "structure_levels_ic": "Support and resistance levels, pivot points",
    "statarb_ic": "ADF stationarity test, information coefficient, mean-reversion signals",
    "options_vol_ic": "Implied volatility, Greeks, skew, term structure",
    "risk_limits_ic": "VaR, stress tests, position limit checks",
    "calendar_events_ic": "Earnings dates, FOMC, CPI, macro event calendar",
    "news_sentiment_ic": "News sentiment score, recent headline risk",
    "options_flow_ic": "Unusual options activity, put/call ratio, institutional flow",
    "fundamentals_ic": "P/E, EPS growth, revenue, sector comparison",
}

# Valid IC names for input validation
_VALID_IC_NAMES = list(_IC_DESCRIPTIONS.keys())

# Valid pod names for input validation
_VALID_POD_NAMES = [
    "data_pod_manager",
    "market_monitor_pod_manager",
    "technicals_pod_manager",
    "quant_pod_manager",
    "risk_pod_manager",
    "alpha_signals_pod_manager",
]


# =============================================================================
# Private Helpers
# =============================================================================


# Standard crew inputs for minimal runs (symbol is injected at call time)
def _minimal_crew_inputs(symbol: str, regime: dict[str, Any]) -> dict[str, Any]:
    from datetime import date

    return {
        "symbol": symbol,
        "current_date": str(date.today()),
        "regime": regime,
        "regime_str": (
            f"Trend: {regime.get('trend', 'unknown')}, "
            f"Volatility: {regime.get('volatility', 'normal')}, "
            f"Confidence: {regime.get('confidence', 0.5):.0%}"
        ),
        "portfolio": {},
        "historical_context": "",
        "asset_class": "equities",
        "instrument_type": "equity",
        "task_intent": "analysis",
        "task_scope": "equities/equity:analysis",
    }


async def _detect_regime_for_symbol(symbol: str) -> dict[str, Any]:
    """Lightweight regime detection for use in IC/pod runners."""
    try:
        from quant_pod.agents.regime_detector import RegimeDetectorAgent

        detector = RegimeDetectorAgent(symbols=[symbol])
        result = await asyncio.get_event_loop().run_in_executor(
            None, detector.detect_regime, symbol
        )
        if result.get("success"):
            return {
                "trend": result.get("trend_regime", "unknown"),
                "volatility": result.get("volatility_regime", "normal"),
                "confidence": result.get("confidence", 0.5),
            }
    except Exception:
        pass
    return {"trend": "unknown", "volatility": "normal", "confidence": 0.5}


# =============================================================================
# MCP Tools
# =============================================================================


@mcp.tool()
async def list_ics() -> dict[str, Any]:
    """
    Return the catalog of all available IC agents and pod managers.

    Each IC entry includes its name, description, which pod it reports to,
    its capabilities, and which asset classes it supports.

    Returns:
        Dict with 'ics' list and 'pods' list.
    """
    try:
        from quant_pod.crews.registry import (
            IC_REGISTRY,
            POD_DEPENDENCIES,
            POD_MANAGER_REGISTRY,
        )

        # Build IC → pod reverse map
        pod_of_ic: dict[str, str] = {"data_ingestion_ic": "data_pod_manager"}
        for pod, ics in POD_DEPENDENCIES.items():
            for ic in ics:
                if ic not in pod_of_ic:
                    pod_of_ic[ic] = pod

        ics = [
            {
                "name": ic_name,
                "description": _IC_DESCRIPTIONS.get(ic_name, ""),
                "pod": pod_of_ic.get(ic_name, "unknown"),
                "capabilities": list(meta.get("capabilities", set())),
                "asset_classes": sorted(meta.get("asset_classes", set())),
            }
            for ic_name, meta in IC_REGISTRY.items()
        ]

        pods = [
            {
                "name": pod_name,
                "capabilities": list(meta.get("capabilities", set())),
                "constituent_ics": POD_DEPENDENCIES.get(pod_name, []),
            }
            for pod_name, meta in POD_MANAGER_REGISTRY.items()
        ]

        return {"success": True, "ics": ics, "pods": pods, "total_ics": len(ics)}

    except Exception as e:
        logger.error(f"[quantpod_mcp] list_ics failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def run_ic(
    ic_name: str,
    symbol: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single IC agent in isolation and return its raw output.

    Runs data_ingestion_ic first (as data prerequisite) if ic_name is not
    itself the data IC, then runs the requested IC as a minimal 2-agent crew.
    Output is cached for 30 minutes.

    Cost: ~1 LLM call for data IC + 1 for target IC (cheaper than full crew).

    Args:
        ic_name: IC to run. Use list_ics() to see valid names.
        symbol: Ticker symbol to analyze.
        params: Optional extra inputs forwarded to the crew.

    Returns:
        Dict with ic_name, symbol, regime_context, raw_output, elapsed_seconds.
    """
    if ic_name not in _VALID_IC_NAMES:
        return {
            "success": False,
            "error": f"Unknown IC '{ic_name}'. Valid: {_VALID_IC_NAMES}",
        }

    start = time.monotonic()

    try:
        regime = await _detect_regime_for_symbol(symbol)

        from quant_pod.crewai_compat import Crew, Process
        from quant_pod.crews.trading_crew import TradingCrew

        tc = TradingCrew()
        ic_factories = tc._ic_agent_factories()
        ic_task_factories = tc._ic_task_factories()

        if ic_name not in ic_factories:
            return {"success": False, "error": f"No factory for IC '{ic_name}'"}

        # Build minimal agent/task list: data IC + target IC
        agents, tasks = [], []
        if ic_name != "data_ingestion_ic":
            agents.append(ic_factories["data_ingestion_ic"]())
            tasks.append(ic_task_factories["data_ingestion_ic"]())

        agents.append(ic_factories[ic_name]())
        tasks.append(ic_task_factories[ic_name]())

        minimal_crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            cache=True,
        )

        inputs = _minimal_crew_inputs(symbol, regime)
        if params:
            inputs.update(params)

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: minimal_crew.kickoff(inputs=inputs)
        )

        # Extract target IC output (last task in the crew)
        raw_output = ""
        if hasattr(result, "tasks_output") and result.tasks_output:
            last = result.tasks_output[-1]
            raw_output = last.raw if hasattr(last, "raw") else str(last)
        elif hasattr(result, "raw"):
            raw_output = str(result.raw)
        else:
            raw_output = str(result)

        ic_cache_set(symbol, ic_name, raw_output)

        elapsed = time.monotonic() - start
        return {
            "success": True,
            "ic_name": ic_name,
            "symbol": symbol,
            "regime_context": regime,
            "raw_output": raw_output,
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"[quantpod_mcp] run_ic({ic_name}, {symbol}) failed: {e}")
        return {
            "success": False,
            "ic_name": ic_name,
            "symbol": symbol,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


@mcp.tool()
async def run_pod(
    pod_name: str,
    symbol: str,
    ic_outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Run a single pod manager with its constituent ICs.

    If ic_outputs is provided (dict of {ic_name: raw_output}), those results
    are injected as context and only the pod manager LLM is invoked.
    If ic_outputs is empty/None, constituent ICs are run first.

    Args:
        pod_name: Pod to run. Use list_ics() to see valid names.
        symbol: Ticker symbol.
        ic_outputs: Optional pre-computed IC outputs to skip re-running ICs.

    Returns:
        Dict with pod_name, symbol, raw_output, constituent_ic_outputs (truncated).
    """
    if pod_name not in _VALID_POD_NAMES:
        return {
            "success": False,
            "error": f"Unknown pod '{pod_name}'. Valid: {_VALID_POD_NAMES}",
        }

    start = time.monotonic()

    try:
        from quant_pod.crewai_compat import Crew, Process
        from quant_pod.crews.registry import POD_DEPENDENCIES
        from quant_pod.crews.trading_crew import TradingCrew

        regime = await _detect_regime_for_symbol(symbol)
        constituent_ics = POD_DEPENDENCIES.get(pod_name, [])
        collected: dict[str, str] = dict(ic_outputs or {})

        tc = TradingCrew()
        ic_factories = tc._ic_agent_factories()
        ic_task_factories = tc._ic_task_factories()
        pod_factories = tc._pod_manager_factories()
        pod_task_factories = tc._pod_task_factories()

        if pod_name not in pod_factories:
            return {"success": False, "error": f"No factory for pod '{pod_name}'"}

        if not collected:
            # Run data_ingestion_ic + constituent ICs + pod manager
            ics_to_run = list(constituent_ics)
            if "data_ingestion_ic" not in ics_to_run:
                ics_to_run.insert(0, "data_ingestion_ic")

            agents = [ic_factories[ic]() for ic in ics_to_run if ic in ic_factories]
            tasks = [ic_task_factories[ic]() for ic in ics_to_run if ic in ic_task_factories]
            agents.append(pod_factories[pod_name]())
            tasks.append(pod_task_factories[pod_name]())

            pod_crew = Crew(
                agents=agents, tasks=tasks, process=Process.sequential, verbose=False, cache=True
            )
            inputs = _minimal_crew_inputs(symbol, regime)

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pod_crew.kickoff(inputs=inputs)
            )

            raw_output = ""
            if hasattr(result, "tasks_output") and result.tasks_output:
                # Last task = pod manager output
                last = result.tasks_output[-1]
                raw_output = last.raw if hasattr(last, "raw") else str(last)
                # Cache per-IC outputs
                for i, ic_nm in enumerate(ics_to_run):
                    if i < len(result.tasks_output):
                        to = result.tasks_output[i]
                        ic_raw = to.raw if hasattr(to, "raw") else str(to)
                        collected[ic_nm] = ic_raw
                        ic_cache_set(symbol, ic_nm, ic_raw)
            elif hasattr(result, "raw"):
                raw_output = str(result.raw)
            else:
                raw_output = str(result)

        else:
            # Use pre-computed IC outputs — only invoke the pod manager
            combined_context = "\n\n".join(
                f"## {ic_nm} Output:\n{out}" for ic_nm, out in collected.items()
            )
            agents = [pod_factories[pod_name]()]
            tasks = [pod_task_factories[pod_name]()]
            pod_crew = Crew(
                agents=agents, tasks=tasks, process=Process.sequential, verbose=False, cache=True
            )

            override_inputs = _minimal_crew_inputs(symbol, regime)
            override_inputs["historical_context"] = combined_context

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pod_crew.kickoff(inputs=override_inputs)
            )
            raw_output = ""
            if hasattr(result, "raw"):
                raw_output = str(result.raw)
            elif hasattr(result, "tasks_output") and result.tasks_output:
                last = result.tasks_output[-1]
                raw_output = last.raw if hasattr(last, "raw") else str(last)
            else:
                raw_output = str(result)

        elapsed = time.monotonic() - start
        return {
            "success": True,
            "pod_name": pod_name,
            "symbol": symbol,
            "constituent_ics": constituent_ics,
            "raw_output": raw_output,
            "ic_outputs_preview": {k: v[:200] + "..." for k, v in collected.items()},
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"[quantpod_mcp] run_pod({pod_name}, {symbol}) failed: {e}")
        return {
            "success": False,
            "pod_name": pod_name,
            "symbol": symbol,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


@mcp.tool()
async def run_crew_subset(
    ic_names: list[str],
    symbol: str,
) -> dict[str, Any]:
    """
    Run a custom subset of ICs through their pod managers to the assistant.

    The assistant synthesizes a partial DailyBrief scoped to only the specified
    ICs. Useful for targeted, cheaper analysis (e.g., only regime + volatility
    ICs for a quick pre-screen before committing to a full run).

    data_ingestion_ic is always auto-added as a prerequisite.
    Pod managers are auto-selected based on which ICs are included.

    Args:
        ic_names: List of IC names to run. Use list_ics() for valid names.
        symbol: Ticker symbol.

    Returns:
        Dict with partial_daily_brief, ics_run, pods_activated, elapsed_seconds.
    """
    invalid = [ic for ic in ic_names if ic not in _VALID_IC_NAMES]
    if invalid:
        return {
            "success": False,
            "error": f"Unknown ICs: {invalid}. Valid: {_VALID_IC_NAMES}",
        }

    start = time.monotonic()

    try:
        from quant_pod.crewai_compat import Crew, Process
        from quant_pod.crews.assembler import PodSelection
        from quant_pod.crews.registry import POD_DEPENDENCIES
        from quant_pod.crews.trading_crew import TradingCrew

        regime = await _detect_regime_for_symbol(symbol)

        # Auto-select pod managers for requested ICs
        activated_pods = [
            pod
            for pod, pod_ics in POD_DEPENDENCIES.items()
            if any(ic in ic_names for ic in pod_ics)
        ]

        # Always include data IC
        full_ic_list = list(ic_names)
        if "data_ingestion_ic" not in full_ic_list:
            full_ic_list.insert(0, "data_ingestion_ic")

        roster = PodSelection(
            asset_class="equities",
            ic_agents=full_ic_list,
            pod_managers=activated_pods,
            profile_used="subset",
        )

        tc = TradingCrew()
        agents = tc._build_agents(roster, stop_at_assistant=True)
        tasks = tc._build_tasks(roster, stop_at_assistant=True)

        subset_crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            cache=True,
        )

        inputs = _minimal_crew_inputs(symbol, regime)
        inputs["historical_context"] = f"Subset analysis — ICs: {full_ic_list}"

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: subset_crew.kickoff(inputs=inputs)
        )

        brief = None
        if hasattr(result, "pydantic") and result.pydantic is not None:
            brief = _serialize(result.pydantic)
        elif hasattr(result, "json_dict") and result.json_dict is not None:
            brief = result.json_dict
        elif hasattr(result, "raw"):
            brief = {"raw_output": str(result.raw)}

        # Cache per-IC outputs from the subset run
        if hasattr(result, "tasks_output") and result.tasks_output:
            for i, ic_nm in enumerate(full_ic_list):
                if i < len(result.tasks_output):
                    to = result.tasks_output[i]
                    ic_cache_set(symbol, ic_nm, to.raw if hasattr(to, "raw") else str(to))

        elapsed = time.monotonic() - start
        return {
            "success": True,
            "symbol": symbol,
            "ics_run": full_ic_list,
            "pods_activated": activated_pods,
            "partial_daily_brief": brief,
            "regime_used": regime,
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"[quantpod_mcp] run_crew_subset({ic_names}, {symbol}) failed: {e}")
        return {
            "success": False,
            "ic_names": ic_names,
            "symbol": symbol,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


@mcp.tool()
async def get_last_ic_output(
    symbol: str,
    ic_name: str,
) -> dict[str, Any]:
    """
    Retrieve the cached raw output from the last IC run for a symbol.

    The cache is populated by run_analysis, run_ic, run_pod, and
    run_crew_subset. Entries expire after 30 minutes.

    Useful for /reflect sessions analyzing which ICs were right without
    re-running the full crew.

    Args:
        ic_name: IC whose output to retrieve.
        symbol: Ticker symbol.

    Returns:
        Dict with raw_output, or cache_miss=True if absent/expired.
    """
    cached = ic_cache_get(symbol, ic_name)
    if cached is None:
        return {
            "success": True,
            "ic_name": ic_name,
            "symbol": symbol,
            "cache_miss": True,
            "note": "No cached output. Run run_analysis, run_ic, or run_crew_subset first.",
        }
    return {
        "success": True,
        "ic_name": ic_name,
        "symbol": symbol,
        "cache_miss": False,
        "raw_output": cached,
    }
