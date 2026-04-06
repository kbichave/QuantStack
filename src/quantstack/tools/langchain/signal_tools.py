"""Signal analysis tools for LangGraph agents."""

import json
import time
from typing import Annotated, Any

from langchain_core.tools import tool
from loguru import logger
from pydantic import Field

from quantstack.tools._state import (
    _read_memory_file,
    _serialize,
    ic_cache_set,
    require_ctx,
)


def _populate_signal_cache(symbol: str, brief: Any) -> None:
    """Populate the IC output cache from SignalBrief collector data."""
    try:
        if not brief.symbol_briefs:
            return
        sb = brief.symbol_briefs[0]
        collector_to_ic = {
            "technical": ["trend_momentum_ic", "volatility_ic", "market_snapshot_ic"],
            "regime": ["regime_detector_ic"],
            "volume": ["structure_levels_ic"],
            "risk": ["risk_limits_ic"],
            "events": ["calendar_events_ic"],
            "fundamentals": ["fundamentals_ic"],
        }
        summary_text = (
            f"SignalEngine output for {symbol}: {sb.market_summary}\n"
            f"Bias: {sb.consensus_bias} (conviction: {sb.consensus_conviction:.2f})\n"
            f"Observations: {'; '.join(sb.key_observations[:3])}"
        )
        for ic_names in collector_to_ic.values():
            for ic_name in ic_names:
                ic_cache_set(symbol, ic_name, summary_text)
    except Exception as exc:
        logger.debug(f"signal cache population failed (non-critical): {exc}")


@tool
async def signal_brief(
    symbol: Annotated[str, Field(description="Ticker symbol to analyze, e.g. 'AAPL', 'SPY', 'TSLA'")],
) -> str:
    """Retrieve a comprehensive technical signal brief for a stock or ETF symbol. Use when you need trend direction, momentum indicators, support/resistance levels, regime classification, and fundamental context for a single ticker. Returns JSON containing technical analysis, fundamental snapshot, momentum scores, regime detection, consensus bias, conviction level, and key observations. Provides the foundation for entry/exit decisions and strategy evaluation. Synonyms: quote, analysis, overview, snapshot, market data, indicator summary."""
    start = time.monotonic()
    symbol = symbol.upper().strip()

    try:
        require_ctx()
    except RuntimeError as exc:
        return json.dumps({"success": False, "error": str(exc)})

    try:
        # Deferred import: SignalEngine pulls litellm+transformers (~1.5s)
        from quantstack.signal_engine import SignalEngine

        engine = SignalEngine()
        brief = await engine.run(symbol)

        session_notes = _read_memory_file("session_handoffs.md", max_chars=800)
        if session_notes:
            brief.strategic_notes = (
                f"{brief.strategic_notes}\n\n---\n{session_notes}".strip()
            )

        _populate_signal_cache(symbol, brief)

        result = {
            "success": True,
            "daily_brief": _serialize(brief.model_dump()),
            "regime_used": brief.regime_detail or {},
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "engine": brief.engine_version,
            "collector_failures": brief.collector_failures,
        }
    except Exception as exc:
        logger.error(f"signal_brief({symbol}) failed: {exc}")
        result = {
            "success": False,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
    return json.dumps(result, default=str)


@tool
async def multi_signal_brief(
    symbols: Annotated[list[str], Field(description="List of ticker symbols to scan in parallel, e.g. ['AAPL', 'MSFT', 'GOOG']")],
) -> str:
    """Retrieve signal briefs for multiple stock or ETF symbols concurrently in a single batch call. Use when scanning a watchlist, comparing tickers, or screening a portfolio for entry candidates. Returns JSON with per-symbol technical analysis, regime data, momentum scores, and collector status. Provides succeeded/failed symbol lists for reliability tracking. Synonyms: batch scan, watchlist analysis, multi-ticker overview, parallel screening, bulk signal fetch."""
    start = time.monotonic()

    try:
        require_ctx()
    except RuntimeError as exc:
        return json.dumps({"success": False, "error": str(exc)})

    if not symbols:
        return json.dumps({
            "results": {},
            "symbols_succeeded": [],
            "symbols_failed": [],
            "elapsed_seconds": 0.0,
        })

    try:
        from quantstack.signal_engine import SignalEngine

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

        result = {
            "results": results,
            "symbols_succeeded": succeeded,
            "symbols_failed": failed,
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
    except Exception as exc:
        logger.error(f"multi_signal_brief failed: {exc}")
        result = {
            "success": False,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
    return json.dumps(result, default=str)
