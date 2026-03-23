"""
Full-stack historical data acquisition via Alpha Vantage (adjusted).

Phases:
  ohlcv_5min        — 5-min adjusted bars (24 months, delta-only)
  ohlcv_daily       — daily adjusted bars (20 years, delta-only)
  financials        — income stmt + balance sheet + cash flow
  earnings_history  — actual vs estimated EPS history
  macro             — CPI, GDP, Fed funds, 10yr yield, unemployment, NFP, etc.
  insider           — insider buy/sell transactions
  institutional     — 13F institutional holdings
  corporate_actions — dividends + stock splits
  options           — historical options chains
  news              — news sentiment (30-day rolling)
  fundamentals      — company overview (sector, beta, ratios, etc.)

Usage:
  # Full cold-start (all 50 symbols, all phases):
  python scripts/acquire_historical_data.py

  # OHLCV only:
  python scripts/acquire_historical_data.py --phases ohlcv_5min ohlcv_daily

  # 5 symbols, all phases:
  python scripts/acquire_historical_data.py --symbols SPY QQQ IWM NVDA TSLA

  # Daily cron (incremental top-up):
  python scripts/acquire_historical_data.py --phases ohlcv_5min ohlcv_daily news

  # Dry run (shows call estimates, no API calls):
  python scripts/acquire_historical_data.py --dry-run

Env:
  ALPHA_VANTAGE_API_KEY      required
  ALPHA_VANTAGE_RATE_LIMIT   set to 75 for $49.99/mo plan (default 5)
  ALPACA_API_KEY / SECRET    optional (fallback OHLCV)
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from quantstack.data.adapters.alpaca import AlpacaAdapter
from quantstack.data.acquisition_pipeline import (
    ALL_PHASES,
    M5_LOOKBACK_MONTHS,
    AcquisitionPipeline,
)
from quantstack.data.fetcher import AlphaVantageClient
from quantstack.data.storage import DataStore
from quantstack.data.universe import INITIAL_LIQUID_UNIVERSE

# AV calls per phase per symbol (used in dry-run estimates)
_CALLS_PER_SYMBOL = {
    "ohlcv_5min": None,  # computed separately (months × symbols)
    "ohlcv_1h": 24,      # 24 monthly slices via intraday_extended
    "ohlcv_daily": 1,
    "financials": 3,
    "earnings_history": 1,
    "macro": 0,  # global — not per symbol
    "insider": 1,
    "institutional": 1,
    "corporate_actions": 2,
    "options": 1,
    "news": 0,  # batched — computed separately
    "fundamentals": 1,
}
_MACRO_CALLS = 9  # fixed regardless of symbol count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-stack data acquisition via Alpha Vantage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbols", nargs="+", metavar="SYM", default=None)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=ALL_PHASES,
        default=ALL_PHASES,
        metavar="PHASE",
        help=f"Phases: {ALL_PHASES}",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=M5_LOOKBACK_MONTHS,
        metavar="N",
        help=f"Months of 5-min history on cold start (default {M5_LOOKBACK_MONTHS})",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


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


async def main() -> int:
    args = _parse_args()

    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    symbols = (
        [s.upper() for s in args.symbols]
        if args.symbols
        else list(INITIAL_LIQUID_UNIVERSE.keys())
    )

    if args.dry_run:
        estimates = _estimate_calls(args.phases, symbols, args.months)
        total = sum(estimates.values())
        mins_at_75 = total / 75
        logger.info("=== DRY RUN ===")
        logger.info(f"Symbols ({len(symbols)}): {', '.join(symbols)}")
        logger.info(f"Phases:  {args.phases}")
        logger.info(f"{'Phase':<22} {'AV calls':>9}")
        logger.info("-" * 33)
        for phase, calls in estimates.items():
            logger.info(f"  {phase:<20} {calls:>9}")
        logger.info("-" * 33)
        logger.info(
            f"  {'TOTAL':<20} {total:>9}  (~{mins_at_75:.1f} min at 75 req/min)"
        )
        return 0

    logger.info(f"Symbols: {len(symbols)} | Phases: {args.phases}")

    av_client = AlphaVantageClient()
    store = DataStore(persistent=True)

    alpaca = None
    try:
        alpaca = AlpacaAdapter()
    except Exception as exc:
        logger.warning(f"AlpacaAdapter init failed: {exc}")

    pipeline = AcquisitionPipeline(av_client=av_client, store=store, alpaca=alpaca)

    try:
        reports = await pipeline.run(
            symbols=symbols, phases=args.phases, m5_lookback_months=args.months
        )
    finally:
        store.close()

    # Summary table
    logger.info("=" * 58)
    logger.info(f"{'Phase':<22} {'ok':>4} {'skip':>5} {'fail':>5} {'secs':>8}")
    logger.info("-" * 58)
    for r in reports:
        logger.info(
            f"{r.phase:<22} {r.succeeded:>4} {r.skipped:>5} {r.failed:>5} {r.elapsed_seconds:>7.1f}s"
        )
    logger.info("=" * 58)

    total_fail = sum(r.failed for r in reports)
    if total_fail > 0 and args.verbose:
        for r in reports:
            for err in r.errors[:5]:
                logger.debug(f"  {r.phase}: {err}")

    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
