# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Data acquisition tool — wraps AcquisitionPipeline for MCP access.

Provides full-stack historical data ingestion via Alpha Vantage:
OHLCV, financials, macro, insider, institutional, options, news, etc.
"""

from typing import Any

from loguru import logger

from quantstack.mcp.server import mcp


@mcp.tool()
async def acquire_historical_data(
    phases: list[str] | None = None,
    symbols: list[str] | None = None,
    m5_lookback_months: int = 24,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run the full-stack data acquisition pipeline via Alpha Vantage.

    Phases (all idempotent, safe to re-run):
      ohlcv_5min, ohlcv_daily, financials, earnings_history, macro,
      insider, institutional, corporate_actions, options, news, fundamentals.

    Each phase checks the DB before calling the API — only fetches missing data.

    Args:
        phases: Which phases to run. Default: all 11 phases.
        symbols: Which symbols to acquire. Default: full liquid universe (~50).
        m5_lookback_months: Months of 5-min history on cold start (default 24).
        dry_run: If True, return estimated API call counts without making any calls.

    Returns:
        {
            "success": True,
            "symbols_count": 50,
            "phases": ["ohlcv_daily", ...],
            "reports": [{"phase": "ohlcv_daily", "ok": 48, "skip": 2, "fail": 0, "secs": 12.3}, ...],
            "total_ok": 48,
            "total_fail": 0,
        }
    """
    try:
        from quantstack.data.acquisition_pipeline import (
            ALL_PHASES,
            AcquisitionPipeline,
        )
        from quantstack.data.fetcher import AlphaVantageClient
        from quantstack.data.storage import DataStore
        from quantstack.data.universe import INITIAL_LIQUID_UNIVERSE

        selected_phases = phases or list(ALL_PHASES)
        selected_symbols = (
            [s.upper() for s in symbols]
            if symbols
            else list(INITIAL_LIQUID_UNIVERSE.keys())
        )

        # Validate phases
        invalid = [p for p in selected_phases if p not in ALL_PHASES]
        if invalid:
            return {
                "success": False,
                "error": f"Invalid phases: {invalid}. Valid: {ALL_PHASES}",
            }

        if dry_run:
            estimates = _estimate_calls(
                selected_phases, selected_symbols, m5_lookback_months
            )
            total = sum(estimates.values())
            return {
                "success": True,
                "dry_run": True,
                "symbols_count": len(selected_symbols),
                "phases": selected_phases,
                "estimated_api_calls": estimates,
                "total_calls": total,
                "estimated_minutes_at_75rpm": round(total / 75, 1),
            }

        av_client = AlphaVantageClient()
        store = DataStore(persistent=True)

        alpaca = None
        try:
            from quantstack.data.adapters.alpaca import AlpacaAdapter

            alpaca = AlpacaAdapter()
        except Exception:
            pass

        pipeline = AcquisitionPipeline(av_client=av_client, store=store, alpaca=alpaca)

        try:
            reports = await pipeline.run(
                symbols=selected_symbols,
                phases=selected_phases,
                m5_lookback_months=m5_lookback_months,
            )
        finally:
            store.close()

        report_dicts = [
            {
                "phase": r.phase,
                "ok": r.succeeded,
                "skip": r.skipped,
                "fail": r.failed,
                "secs": round(r.elapsed_seconds, 1),
                "errors": r.errors[:5] if r.errors else [],
            }
            for r in reports
        ]

        total_ok = sum(r.succeeded for r in reports)
        total_fail = sum(r.failed for r in reports)

        return {
            "success": total_fail == 0,
            "symbols_count": len(selected_symbols),
            "phases": selected_phases,
            "reports": report_dicts,
            "total_ok": total_ok,
            "total_fail": total_fail,
        }

    except Exception as e:
        logger.error(f"[acquisition] acquire_historical_data failed: {e}")
        return {"success": False, "error": str(e)}


# ---- dry-run estimator (mirrors script logic) ----

_CALLS_PER_SYMBOL = {
    "ohlcv_5min": None,
    "ohlcv_daily": 1,
    "financials": 3,
    "earnings_history": 1,
    "macro": 0,
    "insider": 1,
    "institutional": 1,
    "corporate_actions": 2,
    "options": 1,
    "news": 0,
    "fundamentals": 1,
}
_MACRO_CALLS = 9


def _estimate_calls(
    phases: list[str], symbols: list[str], months: int
) -> dict[str, int]:
    n = len(symbols)
    estimates: dict[str, int] = {}
    for phase in phases:
        if phase == "ohlcv_5min":
            estimates[phase] = n * months
        elif phase == "macro":
            estimates[phase] = _MACRO_CALLS
        elif phase == "news":
            estimates[phase] = -(-n // 5)  # ceil div by batch size
        else:
            estimates[phase] = n * _CALLS_PER_SYMBOL.get(phase, 1)
    return estimates
