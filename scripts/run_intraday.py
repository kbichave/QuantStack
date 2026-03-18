#!/usr/bin/env python3
# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
CLI entry point for the live intraday trading loop.

Usage:
    python scripts/run_intraday.py --symbols SPY,QQQ --timeframe M1 --provider alpaca
    python scripts/run_intraday.py --symbols SPY --provider paper --dry-run
    python scripts/run_intraday.py  # loads symbols from active intraday strategies

Environment:
    ALPACA_API_KEY / ALPACA_SECRET_KEY  — for Alpaca streaming
    POLYGON_API_KEY                     — for Polygon tick streaming
    USE_REAL_TRADING=true               — required for live (non-paper) execution
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantPod Live Intraday Trading Loop")
    parser.add_argument(
        "--symbols", type=str, default="",
        help="Comma-separated symbols (e.g. SPY,QQQ,AAPL). Empty = load from strategies.",
    )
    parser.add_argument(
        "--timeframe", type=str, default="M1", choices=["S5", "M1", "M5", "M15"],
        help="Bar granularity (default: M1).",
    )
    parser.add_argument(
        "--provider", type=str, default="alpaca",
        choices=["alpaca", "polygon", "ibkr", "paper"],
        help="Streaming data provider (default: alpaca).",
    )
    parser.add_argument(
        "--flatten-time", type=str, default="15:55",
        help="ET time to flatten all positions (default: 15:55).",
    )
    parser.add_argument(
        "--entry-cutoff", type=str, default="15:30",
        help="ET time to stop new entries (default: 15:30).",
    )
    parser.add_argument(
        "--max-trades", type=int, default=50,
        help="Max trades per day (default: 50).",
    )
    parser.add_argument(
        "--trailing-stop-atr", type=float, default=2.0,
        help="Trailing stop in ATR multiples (default: 2.0).",
    )
    parser.add_argument(
        "--max-hold-bars", type=int, default=0,
        help="Force exit after N bars (0 = disabled).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Initialize all components but don't start streaming.",
    )
    parser.add_argument(
        "--paper", action="store_true", default=True,
        help="Force paper mode (default: True).",
    )
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None

    from quant_pod.intraday import LiveIntradayLoop

    loop = LiveIntradayLoop(
        symbols=symbols,
        timeframe=args.timeframe,
        provider=args.provider,
        paper_mode=args.paper,
        flatten_time_et=args.flatten_time,
        entry_cutoff_et=args.entry_cutoff,
        max_trades_per_day=args.max_trades,
        trailing_stop_atr_mult=args.trailing_stop_atr,
        max_hold_bars=args.max_hold_bars,
        dry_run=args.dry_run,
    )

    report = asyncio.run(loop.run())

    logger.info(
        f"Session report: bars={report.bars_processed} trades={report.trades_submitted} "
        f"filled={report.trades_filled} flattened={report.positions_flattened} "
        f"pnl=${report.realized_pnl:.2f} duration={report.session_duration_seconds:.0f}s"
    )
    if report.errors:
        logger.warning(f"Errors: {report.errors}")


if __name__ == "__main__":
    main()
