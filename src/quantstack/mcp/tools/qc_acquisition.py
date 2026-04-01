# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Data acquisition tool — wraps AcquisitionPipeline for MCP access.

Provides full-stack historical data ingestion via Alpha Vantage:
OHLCV, financials, macro, insider, institutional, options, news, etc.
"""

from typing import Any

from loguru import logger

from quantstack.data.acquisition_pipeline import ALL_PHASES, AcquisitionPipeline
from quantstack.data.adapters.alpaca import AlpacaAdapter
from quantstack.data.fetcher import AlphaVantageClient
from quantstack.data.pg_storage import PgDataStore
from quantstack.universe import INITIAL_LIQUID_UNIVERSE
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain



@domain(Domain.DATA)
@tool_def()
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
        store = PgDataStore()

        alpaca = None
        try:
            alpaca = AlpacaAdapter()
        except Exception as exc:
            logger.debug(f"[qc_acquisition] AlpacaAdapter init failed, proceeding without Alpaca: {exc}")

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


@domain(Domain.DATA)
@tool_def()
async def register_ticker(
    symbol: str,
    description: str = "",
    group: str = "general",
    acquire_data: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Register a ticker: fetch its metadata from Alpha Vantage, store in DB,
    and optionally acquire all 12 phases of historical data.

    This is the programmatic counterpart to the /register-ticker Claude skill.
    Use this tool when an agent needs to register a new ticker at runtime
    without a code change to universe.py (e.g., research loop discovers a
    new candidate mid-session).

    To persist the ticker to universe.py (code), use /register-ticker instead.

    Args:
        symbol: Ticker symbol (case-insensitive, e.g. "HIMS").
        description: Optional one-line override for what the company does.
                     If blank, auto-filled from the first sentence of the AV description.
        group: Logical category — "speculative" | "macro_etf" | "large_cap" | "general".
        acquire_data: If True, run all 12 acquisition phases after registering metadata.
        dry_run: If True, fetch and return metadata only — no DB writes, no acquisition.

    Returns:
        {
            "symbol": "HIMS",
            "name": "Hims & Hers Health Inc",
            "sector": "Healthcare",
            "industry": "Drug Manufacturers",
            "description": "Hims & Hers Health is a telehealth company...",
            "group": "speculative",
            "already_in_db": False,
            "dry_run": False,
            "acquisition": {"total_ok": 12, "total_fail": 0, "reports": [...]},
        }
    """
    symbol = symbol.upper().strip()

    try:
        av_client = AlphaVantageClient()
        overview = av_client.fetch_company_overview(symbol)

        if not overview or "Symbol" not in overview:
            return {
                "success": False,
                "symbol": symbol,
                "error": f"Alpha Vantage returned no data for '{symbol}'. Check the ticker is valid.",
            }

        name = overview.get("Name", symbol)
        sector = overview.get("Sector", "Unknown")
        industry = overview.get("Industry", "")

        # Build short description: use caller-supplied override, or first sentence from AV
        if not description:
            raw_desc = overview.get("Description", "") or ""
            description = raw_desc.split(". ")[0][:200] if raw_desc else ""

        already_in_db = False
        if not dry_run:
            store = PgDataStore()
            try:
                with store._use_conn() as conn:
                    row = conn.execute(
                        "SELECT symbol FROM company_overview WHERE symbol = ?", [symbol]
                    ).fetchone()
                    already_in_db = row is not None

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO company_overview
                            (symbol, name, sector, industry, description,
                             market_cap, dividend_yield, ex_dividend_date,
                             fifty_two_week_high, fifty_two_week_low, beta, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        [
                            symbol, name, sector, industry, description,
                            overview.get("MarketCapitalization"),
                            overview.get("DividendYield"),
                            None,  # ex_dividend_date — let fundamentals phase handle dates
                            overview.get("52WeekHigh"),
                            overview.get("52WeekLow"),
                            overview.get("Beta"),
                        ],
                    )
            finally:
                store.close()

        result: dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "industry": industry,
            "description": description,
            "group": group,
            "already_in_db": already_in_db,
            "dry_run": dry_run,
        }

        if acquire_data and not dry_run:
            acq = await acquire_historical_data(symbols=[symbol])
            result["acquisition"] = {
                "total_ok": acq.get("total_ok", 0),
                "total_fail": acq.get("total_fail", 0),
                "reports": acq.get("reports", []),
            }
        elif dry_run:
            estimates = _estimate_calls(list(ALL_PHASES), [symbol], 24)
            result["acquisition_estimate"] = {
                "phases": list(ALL_PHASES),
                "estimated_api_calls": estimates,
                "total_calls": sum(estimates.values()),
            }

        return result

    except Exception as e:
        logger.error(f"[register_ticker] {symbol}: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


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


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
