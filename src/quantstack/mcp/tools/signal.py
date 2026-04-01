# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3 MCP tools — SignalEngine-backed analysis.

Tools:
  - get_signal_brief      — run SignalEngine for a single symbol (~2–5 sec)
  - run_multi_signal_brief — run SignalEngine for multiple symbols in parallel

These replace run_analysis (TradingCrew / Ollama) as the primary analysis
tools.  The output schema is identical to run_analysis so all existing
skill files, autonomous runners, and consumers work without change.

run_analysis remains available as a fallback (see analysis.py).
"""

import time
from typing import Any

from loguru import logger

from quantstack.mcp._state import (
    _read_memory_file,
    _serialize,
    ic_cache_set,
    require_ctx,
)
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.tools._registry import domain
from quantstack.mcp.domains import Domain
# SignalEngine deferred — pulls litellm+transformers (~1.5s).


# =============================================================================
# TOOL 1: get_signal_brief
# =============================================================================


@domain(Domain.SIGNALS, Domain.INTEL)
@tool_def()
async def get_signal_brief(
    symbol: str,
    regime: dict[str, Any] | None = None,
    include_strategy_context: bool = True,
) -> dict[str, Any]:
    """
    Run all 15 signal collectors for a single symbol and return a unified DailyBrief.

    WHEN TO USE: First tool to call when evaluating any symbol — entries, exits,
    or monitoring. Provides the composite signal picture that every downstream
    decision (debate, sizing, instrument selection) depends on.
    WHEN NOT TO USE: Do not call for symbols you have already briefed this
    iteration unless regime has changed.  Use run_multi_signal_brief when
    scanning 3+ symbols.
    SIGNAL TIER: Aggregates all tiers (tier_1_retail through tier_4_regime_macro).
    WORKFLOW: get_regime (optional) → THIS → trade-debater / position-monitor
    RELATED: run_multi_signal_brief, get_regime, get_portfolio_state

    Args:
        symbol: Ticker symbol (e.g., "SPY", "XOM").
        regime: Pre-computed regime dict.  If None, detected automatically by
                the regime collector using WeeklyRegimeClassifier.
        include_strategy_context: Inject strategy_registry.md into strategic_notes
                                  (same injection as run_analysis does).

    Returns:
        {
            success: bool,
            daily_brief: dict,   # matches DailyBrief schema
            regime_used: dict,
            elapsed_seconds: float,
            engine: "signal_engine_v1",   # distinguishes from run_analysis output
            collector_failures: list[str],
        }
    """
    start = time.monotonic()
    symbol = symbol.upper().strip()

    # Context is required even in degraded mode — SignalEngine reads from DataStore
    # directly and does not need the portfolio DB.  Only execution tools need live DB.
    try:
        require_ctx()
    except RuntimeError as exc:
        return {
            "success": False,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    try:
        from quantstack.signal_engine import SignalEngine  # noqa: PLC0415
        engine = SignalEngine()
        brief = await engine.run(symbol, regime=regime)

        # Override strategic_notes if context injection is requested.
        # This mirrors the strategy_context injection in run_analysis.
        if include_strategy_context:
            session_notes = _read_memory_file("session_handoffs.md", max_chars=800)
            if session_notes:
                brief.strategic_notes = (
                    f"{brief.strategic_notes}\n\n---\n{session_notes}".strip()
                )

        # Populate IC cache with per-collector outputs so /reflect and
        # get_last_ic_output continue to work.  The cache key format is
        # "{symbol}::{ic_name}" — we map collector names to their IC equivalents.
        _populate_signal_cache(symbol, brief)

        elapsed = round(time.monotonic() - start, 2)
        return {
            "success": True,
            "daily_brief": _serialize(brief.model_dump()),
            "regime_used": brief.regime_detail or {},
            "elapsed_seconds": elapsed,
            "engine": brief.engine_version,
            "collector_failures": brief.collector_failures,
        }

    except Exception as exc:
        elapsed = round(time.monotonic() - start, 2)
        logger.error(f"[quantpod_mcp] get_signal_brief({symbol}) failed: {exc}")
        return {
            "success": False,
            "error": str(exc),
            "elapsed_seconds": elapsed,
            "collector_failures": [],
        }


# =============================================================================
# TOOL 2: run_multi_signal_brief
# =============================================================================


@domain(Domain.SIGNALS)
@tool_def()
async def run_multi_signal_brief(
    symbols: list[str],
    regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run SignalEngine for multiple symbols in parallel with bounded concurrency.

    WHEN TO USE: Scanning a watchlist or universe of 3+ symbols at once.
    Prefer this over looping get_signal_brief — it batches API calls and
    runs up to 5 symbols concurrently.
    WHEN NOT TO USE: For a single symbol use get_signal_brief (avoids
    list overhead). Do not use for > 20 symbols in one call — split into
    batches yourself to stay within Alpha Vantage rate limits.
    SIGNAL TIER: Aggregates all tiers per symbol.
    WORKFLOW: get_regime (optional, shared) → THIS → filter candidates → trade-debater
    RELATED: get_signal_brief, get_regime

    Args:
        symbols: List of ticker symbols (e.g., ["SPY", "XOM", "MSFT"]).
        regime: Pre-computed regime dict applied to all symbols.
                If None, each symbol detects its own regime.

    Returns:
        {
            results: {symbol: {success, daily_brief, ...}},
            symbols_succeeded: list[str],
            symbols_failed: list[str],
            elapsed_seconds: float,
        }
    """
    start = time.monotonic()

    try:
        require_ctx()
    except RuntimeError as exc:
        return {"success": False, "error": str(exc)}

    if not symbols:
        return {
            "results": {},
            "symbols_succeeded": [],
            "symbols_failed": [],
            "elapsed_seconds": 0.0,
        }

    try:
        from quantstack.signal_engine import SignalEngine  # noqa: PLC0415
        engine = SignalEngine()
        clean_symbols = [s.upper().strip() for s in symbols]
        briefs = await engine.run_multi(clean_symbols, max_concurrent=5)

        results = {}
        succeeded = []
        failed = []

        for sym, brief in zip(clean_symbols, briefs):
            if brief.overall_confidence == 0.0 and "all" in brief.collector_failures:
                failed.append(sym)
                results[sym] = {"success": False, "error": "all collectors failed"}
            else:
                succeeded.append(sym)
                results[sym] = {
                    "success": True,
                    "daily_brief": _serialize(brief.model_dump()),
                    "regime_used": brief.regime_detail or {},
                    "collector_failures": brief.collector_failures,
                }
                _populate_signal_cache(sym, brief)

        return {
            "results": results,
            "symbols_succeeded": succeeded,
            "symbols_failed": failed,
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    except Exception as exc:
        logger.error(f"[quantpod_mcp] run_multi_signal_brief failed: {exc}")
        return {
            "success": False,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }


# =============================================================================
# Helpers
# =============================================================================


def _populate_signal_cache(symbol: str, brief: Any) -> None:
    """
    Populate the IC output cache from SignalBrief collector data.

    Maps SignalEngine collector names to their IC-agent equivalents so
    get_last_ic_output and /reflect sessions see fresh data.
    """
    try:
        # Each symbol_brief's observations serve as the "IC output" for the cache.
        if not brief.symbol_briefs:
            return
        sb = brief.symbol_briefs[0]

        # Map collector → IC name for cache compatibility
        collector_to_ic = {
            "technical": ["trend_momentum_ic", "volatility_ic", "market_snapshot_ic"],
            "regime": ["regime_detector_ic"],
            "volume": ["structure_levels_ic"],
            "risk": ["risk_limits_ic"],
            "events": ["calendar_events_ic"],
            "fundamentals": ["fundamentals_ic"],
        }

        # Use the symbol_brief's market_summary as a stand-in IC output.
        summary_text = (
            f"SignalEngine output for {symbol}: {sb.market_summary}\n"
            f"Bias: {sb.consensus_bias} (conviction: {sb.consensus_conviction:.2f})\n"
            f"Observations: {'; '.join(sb.key_observations[:3])}"
        )

        for ic_names in collector_to_ic.values():
            for ic_name in ic_names:
                ic_cache_set(symbol, ic_name, summary_text)

    except Exception as exc:
        logger.debug(
            f"[quantpod_mcp] signal cache population failed (non-critical): {exc}"
        )


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
