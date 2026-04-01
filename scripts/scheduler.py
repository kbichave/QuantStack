#!/usr/bin/env python3
# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantStack Scheduler — deterministic jobs that run on a cron schedule.

The research loop and trading loop (tmux) handle all LLM-based work.
This scheduler runs ONLY the LLM-free Python jobs that need fixed timing:

  08:00 Mon-Fri — Daily data refresh (Alpha Vantage cache update)
  16:10 Mon-Fri — Daily P&L attribution (equity snapshot + benchmark)
  18:00 Sun     — Strategy lifecycle weekly (gap analysis, backtest, promote)
  09:00 1st/mo  — Strategy lifecycle monthly (validate live, retire degraded)

No Claude Code sessions. No LLM calls. Those are the research/trading loops' job.

Usage:
    python scripts/scheduler.py [--dry-run]  # Print schedule without running
    python scripts/scheduler.py              # Start the scheduler daemon
    python scripts/scheduler.py --run-now data_refresh
    python scripts/scheduler.py --run-now strategy_lifecycle_weekly
    python scripts/scheduler.py --run-now strategy_lifecycle_monthly
    python scripts/scheduler.py --cron       # Print equivalent cron entries

Requirements:
    APScheduler: pip install quantstack[scheduler]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from quantstack.autonomous.strategy_lifecycle import StrategyLifecycle
from quantstack.db import open_db

logger = logging.getLogger("quantstack.scheduler")

WORKDIR = Path(os.getenv("QUANTSTACK_WORKDIR", Path(__file__).parent.parent))
TIMEZONE = "US/Eastern"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def run_data_refresh(dry_run: bool = False) -> None:
    """Full daily data refresh — all 12 phases for all universe symbols.

    All phases are idempotent: existing rows are skipped, only deltas fetched.
    At 75 req/min (AV premium) incremental runs take ~5-15 min after cold start.
    """
    label = "data_refresh"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    # No --phases flag = defaults to all 12 phases
    cmd = [sys.executable, "scripts/acquire_historical_data.py"]

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {' '.join(cmd)}")
        return

    try:
        result = subprocess.run(
            cmd, cwd=str(WORKDIR), timeout=7200, check=False,
        )
        if result.returncode != 0:
            logger.warning(f"'{label}' exited with code {result.returncode}")
        else:
            logger.info(f"'{label}' completed successfully")
    except subprocess.TimeoutExpired:
        logger.error(f"'{label}' timed out after 120 minutes")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_eod_data_refresh(dry_run: bool = False) -> None:
    """EOD data refresh — OHLCV close bar, options chains, news, macro.

    Runs after market close (16:30 ET) to capture the final close bar and
    refresh options chains, news sentiment, and macro data. Separate from
    the 08:00 full refresh so intraday data stays current for the trading loop.

    After completion, triggers feature recomputation for universe symbols.
    If > 15% of symbols fail, logs an error to session_handoffs.md.
    """
    label = "eod_data_refresh"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    cmd = [
        sys.executable, "scripts/acquire_historical_data.py",
        "--phases", "ohlcv_daily", "options", "news", "macro",
    ]

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {' '.join(cmd)}")
        return

    try:
        result = subprocess.run(
            cmd, cwd=str(WORKDIR), timeout=3600, check=False,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"'{label}' exited with code {result.returncode}")
            # Check failure rate from stderr — log to session_handoffs.md if > 15%
            _maybe_log_data_failure(result.stderr, label)
        else:
            logger.info(f"'{label}' completed successfully")
    except subprocess.TimeoutExpired:
        logger.error(f"'{label}' timed out after 60 minutes")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def _maybe_log_data_failure(stderr: str, label: str) -> None:
    """Log a high-failure-rate data refresh error to session_handoffs.md."""
    try:
        # Count failure lines vs total — heuristic: look for 'failed' in stderr
        lines = stderr.splitlines()
        failures = sum(1 for l in lines if "failed" in l.lower() or "error" in l.lower())
        total = max(len(lines), 1)
        if failures / total > 0.15:
            handoffs = WORKDIR / ".claude" / "memory" / "session_handoffs.md"
            with open(handoffs, "a") as f:
                f.write(
                    f"\n## {label} high failure rate — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    f"Failure rate: {failures}/{total} lines contain errors. "
                    f"Check data provider connectivity.\n"
                )
    except Exception:
        pass


def run_credit_regime_revalidation(dry_run: bool = False) -> None:
    """Re-validate credit regime with fresh EOD data (16:45 ET Mon-Fri).

    Reads HY/IG spread proxies (HYG/LQD), yield curve, dollar direction
    from the freshly-updated OHLCV cache and writes the result to the
    system_state table. Unblocks long entries if credit_regime has tightened
    (e.g., moved from "widening" to "stable" or "contracting").
    """
    label = "credit_regime_revalidation"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {label}")
        return

    try:
        import asyncio

        from quantstack.data.pg_storage import PgDataStore
        from quantstack.mcp._helpers import set_shared_reader
        from quantstack.mcp.tools.macro_signals import get_credit_market_signals

        store = PgDataStore()
        set_shared_reader(store)

        signals = asyncio.run(get_credit_market_signals())
        regime = signals.get("credit_regime", "unknown")

        conn = open_db()
        conn.execute(
            """
            INSERT INTO system_state (key, value, updated_at)
            VALUES ('credit_regime', %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """,
            [regime],
        )
        conn.commit()
        conn.close()

        logger.info(f"[{label}] credit_regime={regime} "
                    f"hy_spread_zscore={signals.get('hy_spread_zscore')}")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_autoresclaw_weekly(dry_run: bool = False) -> None:
    """AutoResearchClaw weekly deep research (Sunday 20:00 ET).

    Reads top 3 pending tasks from research_queue and launches AutoResearchClaw
    sequentially. Requires Docker. Output artifacts land in reports/autoresclaw/YYYY-MM-DD/.

    Task sources:
      - trade-reflector (loss > 1%)       → task_type=bug_fix
      - DriftDetector (CRITICAL severity) → task_type=ml_arch_search
      - Research loop (coverage gap)      → task_type=strategy_hypothesis
      - Human (manual insert)             → any task_type
    """
    label = "autoresclaw_weekly"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    cmd = [sys.executable, "scripts/autoresclaw_runner.py", "--limit", "3"]

    if dry_run:
        cmd.append("--dry-run")
        print(f"\n[DRY RUN] Would run at {timestamp}: {' '.join(cmd)}")
        return

    try:
        result = subprocess.run(
            cmd, cwd=str(WORKDIR), timeout=14400, check=False,
        )
        if result.returncode == 0:
            logger.info(f"'{label}' completed — all tasks succeeded")
        elif result.returncode == 2:
            logger.warning(f"'{label}' completed with partial failures (see autoresclaw log)")
        else:
            logger.error(f"'{label}' exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        logger.error(f"'{label}' timed out after 4 hours")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_strategy_lifecycle_weekly(dry_run: bool = False) -> None:
    """Weekly lifecycle: gap analysis, candidate generation, backtest, promote to forward_testing.

    Runs Sunday 18:00 ET so results are ready before Monday open.
    """
    label = "strategy_lifecycle_weekly"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {label}")
        return

    try:
        conn = open_db()
        lifecycle = StrategyLifecycle(conn)
        report = asyncio.run(lifecycle.run_weekly())
        logger.info(
            f"[{label}] gaps={len(report.gaps_found)} generated={report.candidates_generated} "
            f"passed={report.candidates_passed} promoted={len(report.promotions)} "
            f"errors={len(report.errors)}"
        )
        for err in report.errors:
            logger.warning(f"[{label}] candidate error: {err}")
        conn.close()
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_strategy_lifecycle_monthly(dry_run: bool = False) -> None:
    """Monthly lifecycle: validate live strategies, retire degraded ones.

    Runs on the 1st of each month at 09:00 ET.
    """
    label = "strategy_lifecycle_monthly"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {label}")
        return

    try:
        conn = open_db()
        lifecycle = StrategyLifecycle(conn)
        report = asyncio.run(lifecycle.run_monthly())
        logger.info(f"[{label}] retired={len(report.retirements)}: {report.retirements}")
        conn.close()
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_daily_attribution(dry_run: bool = False) -> None:
    """Daily P&L attribution — equity snapshot, strategy P&L, benchmark comparison.

    Runs EquityTracker.snapshot_daily() and BenchmarkTracker.update_benchmark()
    after market close. Idempotent: safe to retry (skips if snapshot exists).
    """
    label = "daily_attribution"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {label}")
        return

    try:
        from quantstack.db import open_db
        from quantstack.performance.benchmark import BenchmarkTracker
        from quantstack.performance.equity_tracker import EquityTracker

        conn = open_db()

        # 1. Daily equity snapshot + per-strategy P&L
        equity_result = EquityTracker(conn).snapshot_daily()
        logger.info(f"[{label}] Equity snapshot: {equity_result}")

        # 2. Benchmark close price
        bench = BenchmarkTracker(conn)
        bench_result = bench.update_benchmark("SPY")
        logger.info(f"[{label}] Benchmark update: {bench_result}")

        # 3. Rolling comparison (30d, 60d, 90d)
        comp_result = bench.compute_comparison("SPY", [30, 60, 90])
        logger.info(f"[{label}] Benchmark comparison: {len(comp_result)} windows computed")

        conn.close()
        logger.info(f"'{label}' completed successfully")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_memory_compaction(dry_run: bool = False) -> None:
    """Trim oversized memory files to prevent context window bloat.

    Runs Sunday 17:00 ET — one hour before strategy lifecycle — so the
    research loop starts the week with a compact context window.

    Excess lines are appended to {filename}.archive.md so nothing is lost.
    """
    label = "memory_compaction"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    # Lines-per-file limits (same as start.sh compaction).
    limits = {
        "workshop_lessons.md": 100,
        "ml_experiment_log.md": 120,
        "trade_journal.md": 150,
        "ml_research_program.md": 80,
    }
    memory_dir = WORKDIR / ".claude" / "memory"

    for filename, max_lines in limits.items():
        filepath = memory_dir / filename
        if not filepath.exists():
            continue
        lines = filepath.read_text().splitlines()
        if len(lines) <= max_lines:
            logger.info(f"[{label}] {filename}: {len(lines)} lines (OK)")
            continue

        archive_path = memory_dir / f"{filename}.archive.md"
        excess = lines[:-max_lines]
        keep = lines[-max_lines:]

        if dry_run:
            logger.info(f"[{label}] DRY RUN: would compact {filename} {len(lines)} → {max_lines} lines")
            continue

        with open(archive_path, "a") as f:
            f.write(f"\n\n## Archived {datetime.now().isoformat()}\n\n")
            f.write("\n".join(excess))
        filepath.write_text("\n".join(keep))
        logger.info(
            f"[{label}] {filename}: compacted {len(lines)} → {max_lines} lines "
            f"({len(excess)} archived)"
        )


def reset_av_daily_counter(dry_run: bool = False) -> None:
    """Reset the Alpha Vantage daily call counter at midnight ET.

    Stored in system_state as av_daily_calls_{YYYY-MM-DD}.  The fetcher
    reads today's key; yesterday's key is never read again, so we only
    need to ensure today's key starts at 0.  Writing a fresh row is
    sufficient — we don't need to delete old keys (they expire naturally).
    """
    label = "av_daily_counter_reset"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would reset AV daily counter at {timestamp}")
        return

    try:
        conn = open_db()
        today_key = f"av_daily_calls_{datetime.now().strftime('%Y-%m-%d')}"
        conn.execute(
            """
            INSERT INTO system_state (key, value, updated_at)
            VALUES (%s, '0', NOW())
            ON CONFLICT (key) DO UPDATE SET value = '0', updated_at = NOW()
            """,
            [today_key],
        )
        conn.commit()
        conn.close()
        logger.info(f"[{label}] Reset {today_key} to 0")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_intraday_quote_refresh(dry_run: bool = False) -> None:
    """Refresh real-time quotes for held positions and hot watchlist via Alpaca.

    Alpaca market data is free and unlimited (no per-call quota like AV).
    Updates positions.current_price and unrealized_pnl so the trading loop
    always has fresh prices without burning AV calls.

    Runs every 5 min Mon-Fri 9:30-16:00 ET.
    """
    label = "intraday_quote_refresh"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {label}")
        return

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockSnapshotRequest
        from quantstack.config.settings import get_settings

        settings = get_settings()
        api_key = settings.alpaca.api_key
        secret_key = settings.alpaca.secret_key
        if not api_key or not secret_key:
            logger.warning(f"[{label}] ALPACA_API_KEY/SECRET not set — skipping")
            return

        client = StockHistoricalDataClient(api_key, secret_key)

        conn = open_db()

        # 1. Get held positions
        rows = conn.execute(
            "SELECT symbol, quantity, avg_cost, side FROM positions"
        ).fetchall()
        held_symbols = [r[0] for r in rows]
        positions_map = {r[0]: {"qty": r[1], "avg_cost": r[2], "side": r[3]} for r in rows}

        # 2. Get active watchlist (top 20 by score, excluding held)
        watch_rows = conn.execute(
            """SELECT symbol FROM universe
               WHERE is_active = TRUE AND symbol NOT IN (
                   SELECT symbol FROM positions
               )
               ORDER BY symbol LIMIT 20"""
        ).fetchall()
        watch_symbols = [r[0] for r in watch_rows]

        all_symbols = list(set(held_symbols + watch_symbols))
        if not all_symbols:
            logger.info(f"[{label}] No positions or watchlist symbols — skipping")
            conn.close()
            return

        # 3. Batch snapshot (1 API call for all symbols)
        request = StockSnapshotRequest(symbol_or_symbols=all_symbols)
        snapshots = client.get_stock_snapshot(request)

        if not snapshots:
            logger.warning(f"[{label}] Empty snapshot response")
            conn.close()
            return

        # 4. Update positions with fresh prices
        updated = 0
        for symbol, snap in snapshots.items():
            if symbol not in positions_map:
                continue
            price = float(snap.latest_trade.price) if snap.latest_trade else None
            if price is None:
                continue
            pos = positions_map[symbol]
            if pos["side"] == "long":
                pnl = (price - pos["avg_cost"]) * pos["qty"]
            else:
                pnl = (pos["avg_cost"] - price) * abs(pos["qty"])

            conn.execute(
                """UPDATE positions
                   SET current_price = %s, unrealized_pnl = %s, last_updated = NOW()
                   WHERE symbol = %s""",
                [price, pnl, symbol],
            )
            updated += 1

        conn.commit()
        conn.close()

        logger.info(
            f"[{label}] Refreshed {len(snapshots)} snapshots, "
            f"updated {updated} positions (held={len(held_symbols)}, "
            f"watch={len(watch_symbols)})"
        )
    except ImportError:
        logger.warning(f"[{label}] alpaca-py not installed — skipping")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_av_intraday_5min(dry_run: bool = False) -> None:
    """Fetch latest 5-min adjusted bars from Alpha Vantage for all universe symbols.

    AV provides split/dividend-adjusted prices (Alpaca IEX does not).
    Uses outputsize=compact (latest 100 bars per symbol, 1 AV call each).
    At 75 req/min and ~45 active symbols, completes in ~40s.

    Runs every 15 min Mon-Fri 9:45-16:00 ET. Complements the Alpaca IEX
    5-min feed (which runs every 5 min but is unadjusted).
    """
    label = "av_intraday_5min"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {label}")
        return

    try:
        from quantstack.data.fetcher import AlphaVantageClient
        from quantstack.data.pg_storage import PgDataStore, Timeframe
        from quantstack.db import pg_conn

        # Get active universe
        with pg_conn() as conn:
            rows = conn.execute(
                "SELECT symbol FROM universe WHERE is_active = TRUE"
            ).fetchall()
        symbols = [r[0] for r in rows]
        if not symbols:
            logger.info(f"[{label}] Empty universe — skipping")
            return

        av = AlphaVantageClient()
        store = PgDataStore()
        total_saved = 0
        failed = 0

        for symbol in symbols:
            try:
                df = av.fetch_intraday(symbol, interval="5min", outputsize="compact")
                if df.empty:
                    continue
                rows_saved = store.save_ohlcv(df, symbol, Timeframe.M5)
                total_saved += rows_saved
            except Exception as exc:
                failed += 1
                logger.debug(f"[{label}] {symbol} failed: {exc}")

        logger.info(
            f"[{label}] Saved {total_saved} bars for {len(symbols)} symbols "
            f"({failed} failed, {len(symbols) - failed} ok)"
        )
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_intraday_5min_bars(dry_run: bool = False) -> None:
    """Fetch latest 5-min OHLCV bars for all universe symbols via Alpaca IEX.

    Alpaca IEX is free, unlimited, and 15-min delayed — same delay as AV but
    no call quota. Saves to the ohlcv table (timeframe='5M') so the trading
    loop and options strategies always have near-real-time intraday bars.

    Batches symbols in groups of 50 (Alpaca limit per request).
    Runs every 5 min Mon-Fri 9:30-16:00 ET.
    """
    label = "intraday_5min_bars"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {label}")
        return

    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        from quantstack.db import open_db, pg_conn

        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            logger.warning(f"[{label}] ALPACA keys not set — skipping")
            return

        client = StockHistoricalDataClient(api_key, secret_key)

        # Get active universe symbols
        with pg_conn() as conn:
            rows = conn.execute(
                "SELECT symbol FROM universe WHERE is_active = TRUE"
            ).fetchall()
        symbols = [r[0] for r in rows]
        if not symbols:
            logger.info(f"[{label}] Empty universe — skipping")
            return

        # Fetch last 10 minutes of 5-min bars (2 bars per symbol)
        end = datetime.now(tz=__import__('datetime').timezone.utc)
        start = end - __import__('datetime').timedelta(minutes=10)
        total_saved = 0

        # Batch in groups of 50
        for i in range(0, len(symbols), 50):
            batch = symbols[i:i + 50]
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                    start=start,
                    end=end,
                    feed=DataFeed.IEX,
                )
                bars = client.get_stock_bars(req)

                if not bars or not bars.data:
                    continue

                # Save to ohlcv table
                with pg_conn() as conn:
                    for sym, sym_bars in bars.data.items():
                        for b in sym_bars:
                            conn.execute(
                                """
                                INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume)
                                VALUES (%s, '5M', %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                                    open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                                    close=EXCLUDED.close, volume=EXCLUDED.volume
                                """,
                                [sym, b.timestamp, float(b.open), float(b.high),
                                 float(b.low), float(b.close), float(b.volume)],
                            )
                            total_saved += 1
            except Exception as batch_exc:
                logger.warning(f"[{label}] Batch {i}-{i+len(batch)} failed: {batch_exc}")

        logger.info(f"[{label}] Saved {total_saved} 5-min bars for {len(symbols)} symbols")
    except ImportError:
        logger.warning(f"[{label}] alpaca-py not installed — skipping")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_community_intel_weekly(dry_run: bool = False) -> None:
    """Community intelligence weekly scan (Sunday 19:00 ET).

    Runs the community-intel agent to discover new quant techniques, tools,
    and alpha factors from Reddit r/algotrading, GitHub trending repos,
    arXiv preprints, X/Twitter quant accounts, and quant newsletters.

    Discoveries are inserted into research_queue (task_type=strategy_hypothesis)
    and summarized in session_handoffs.md. Runs before AutoResearchClaw at 20:00
    so newly queued items are available for that night's deep research run.
    """
    label = "community_intel_weekly"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    log_file = WORKDIR / "data" / "logs" / "community_intel.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    agent_prompt = WORKDIR / ".claude" / "agents" / "community-intel.md"
    if not agent_prompt.exists():
        logger.error(f"[{label}] Agent prompt not found: {agent_prompt}")
        return

    cmd = f"cat {agent_prompt} | claude --model haiku 2>&1 | tee -a {log_file}"

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {cmd}")
        return

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=str(WORKDIR), timeout=3600, check=False,
        )
        if result.returncode == 0:
            logger.info(f"'{label}' completed successfully")
        else:
            logger.warning(f"'{label}' exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        logger.error(f"'{label}' timed out after 1 hour")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

JOBS = [
    # ── Intraday (market hours) ──────────────────────────────────────────
    # Quotes via Alpaca IEX feed (free, unlimited, 15-min delayed).
    # Updates positions.current_price and unrealized_pnl every 15 min.
    # 15-min delayed is far better than stale-since-08:00 for P&L and stop monitoring.
    {"trigger": {"minute": "*/15", "hour": "9-16", "day_of_week": "mon-fri"}, "func": run_intraday_quote_refresh, "label": "intraday_quote_refresh_15m"},
    # 5-min adjusted OHLCV bars via Alpha Vantage (compact=100 bars/call).
    # Options strategies need intraday bars for entry/exit timing.
    # AV provides adjusted prices; runs every 5 min during market hours.
    {"trigger": {"minute": "*/5", "hour": "9-16", "day_of_week": "mon-fri"}, "func": run_av_intraday_5min, "label": "av_intraday_5min_5m"},
    # Credit regime: check every 2h during market hours (spreads can gap intraday).
    # A morning widening event would block long entries for 6+ hours without this.
    {"trigger": {"hour": "10,12,14", "minute": 0, "day_of_week": "mon-fri"}, "func": run_credit_regime_revalidation, "label": "credit_regime_intraday_2h"},
    {"trigger": {"hour": 16, "minute": 45, "day_of_week": "mon-fri"}, "func": run_credit_regime_revalidation, "label": "credit_regime_eod_16:45"},

    # ── Daily ────────────────────────────────────────────────────────────
    # Full AV data refresh every weekday morning — all 12 phases, all universe symbols.
    {"trigger": {"hour": 8, "minute": 0, "day_of_week": "mon-fri"}, "func": run_data_refresh, "label": "data_refresh_08:00"},
    # EOD close bar + options/news/macro refresh after market close.
    {"trigger": {"hour": 16, "minute": 30, "day_of_week": "mon-fri"}, "func": run_eod_data_refresh, "label": "eod_data_refresh_16:30"},
    # Daily P&L attribution (equity snapshot + benchmark comparison).
    {"trigger": {"hour": 16, "minute": 10, "day_of_week": "mon-fri"}, "func": run_daily_attribution, "label": "daily_attribution_16:10"},
    # AV daily counter reset — ensures quota guard starts at 0 each trading day.
    {"trigger": {"hour": 0, "minute": 1}, "func": reset_av_daily_counter, "label": "av_counter_reset_midnight"},

    # ── Weekly ───────────────────────────────────────────────────────────
    # Memory compaction — Sunday + Wednesday to prevent mid-week bloat from 720 research sessions/day.
    {"trigger": {"hour": 17, "minute": 0, "day_of_week": "sun"}, "func": run_memory_compaction, "label": "memory_compaction_sun17:00"},
    {"trigger": {"hour": 17, "minute": 0, "day_of_week": "wed"}, "func": run_memory_compaction, "label": "memory_compaction_wed17:00"},
    # Strategy lifecycle — deterministic, no LLM, runs outside market hours.
    {"trigger": {"hour": 18, "minute": 0, "day_of_week": "sun"}, "func": run_strategy_lifecycle_weekly, "label": "strategy_lifecycle_weekly_sun18:00"},
    # Community intelligence scan — discovers new quant techniques from Reddit/GitHub/arXiv.
    {"trigger": {"hour": 19, "minute": 0, "day_of_week": "sun"}, "func": run_community_intel_weekly, "label": "community_intel_weekly_sun19:00"},
    # AutoResearchClaw deep research — processes research_queue (requires Docker).
    {"trigger": {"hour": 20, "minute": 0, "day_of_week": "sun"}, "func": run_autoresclaw_weekly, "label": "autoresclaw_weekly_sun20:00"},

    # ── Monthly ──────────────────────────────────────────────────────────
    {"trigger": {"hour": 9, "minute": 0, "day": "1"}, "func": run_strategy_lifecycle_monthly, "label": "strategy_lifecycle_monthly_1st09:00"},
]


def _check_data_freshness_and_sync(dry_run: bool = False) -> None:
    """On startup, check if OHLCV data is stale and trigger sync if needed."""
    label = "startup_data_freshness_check"
    try:
        conn = open_db()
        row = conn.execute(
            "SELECT MAX(timestamp) FROM ohlcv WHERE symbol = 'SPY' AND timeframe IN ('1D', '1d', 'daily', 'D1')"
        ).fetchone()
        conn.close()

        if not row or not row[0]:
            logger.warning(f"[{label}] No OHLCV data — triggering full data refresh")
            run_data_refresh(dry_run=dry_run)
            return

        latest = row[0]
        if hasattr(latest, 'replace'):
            latest = latest.replace(tzinfo=None)
        age_days = (datetime.now() - latest).days

        if age_days > 1:
            logger.info(f"[{label}] Data is {age_days} days stale — triggering incremental sync")
            run_data_refresh(dry_run=dry_run)
        else:
            logger.info(f"[{label}] Data is fresh ({age_days} days old) — no sync needed")
    except Exception as exc:
        logger.error(f"[{label}] Failed: {exc}")


def start_scheduler(dry_run: bool = False) -> None:
    """Start the APScheduler daemon."""
    # Startup freshness check — covers missed 08:00 job
    _check_data_freshness_and_sync(dry_run=dry_run)

    scheduler = BlockingScheduler(timezone=TIMEZONE)

    for job in JOBS:
        trigger = CronTrigger(**job["trigger"], timezone=TIMEZONE)
        scheduler.add_job(
            job["func"], trigger=trigger, args=[dry_run],
            id=job["label"], name=job["label"],
            misfire_grace_time=300,
        )
        logger.info(f"Scheduled '{job['label']}' {job['trigger']} {TIMEZONE}")

    print(
        f"\nQuantStack Scheduler started (workdir={WORKDIR})\n"
        f"Scheduled {len(JOBS)} jobs. Press Ctrl+C to stop.\n\n"
        f"  */5   Mon-Fri 9-16 — 5-min adjusted OHLCV bars via Alpha Vantage\n"
        f"  */15  Mon-Fri 9-16 — Position quote refresh via Alpaca (P&L + stops)\n"
        f"  10,12,14 Mon-Fri   — Credit regime intraday checks (2h intervals)\n"
        f"  08:00 Mon-Fri      — Full data refresh (all 12 phases, all universe symbols)\n"
        f"  16:10 Mon-Fri      — Daily P&L attribution (equity snapshot + benchmark)\n"
        f"  16:30 Mon-Fri      — EOD data refresh (close bar, options, news, macro)\n"
        f"  16:45 Mon-Fri      — Credit regime EOD re-validation\n"
        f"  17:00 Sun+Wed      — Memory compaction (trim oversized files)\n"
        f"  18:00 Sun          — Strategy lifecycle weekly (gap analysis, promote)\n"
        f"  19:00 Sun          — Community intel weekly (Reddit/GitHub/arXiv)\n"
        f"  20:00 Sun          — AutoResearchClaw deep research (research_queue)\n"
        f"  09:00 1st/mo       — Strategy lifecycle monthly (validate, retire)\n"
        f"\n"
        f"  Note: Trading execution handled by tmux trading loop\n"
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


def _print_cron() -> None:
    """Print equivalent cron entries."""
    print("# QuantStack scheduled jobs (add to crontab, TZ=America/New_York)")
    print(f"0   8  * * 1-5  cd {WORKDIR} && python scripts/acquire_historical_data.py")
    print(f"10 16  * * 1-5  cd {WORKDIR} && python scripts/scheduler.py --run-now daily_attribution")
    print(f"0  18  * * 0    cd {WORKDIR} && python scripts/scheduler.py --run-now strategy_lifecycle_weekly")
    print(f"0   9  1 * *    cd {WORKDIR} && python scripts/scheduler.py --run-now strategy_lifecycle_monthly")
    print("# Trading execution: tmux trading loop (prompts/trading_loop.md)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="QuantStack scheduler — deterministic jobs only")
    parser.add_argument("--dry-run", action="store_true", help="Print schedule without running")
    parser.add_argument("--cron", action="store_true", help="Print equivalent cron entries")
    parser.add_argument("--run-now", metavar="LABEL", help="Run a job immediately (data_refresh, autonomous_runner)")
    args = parser.parse_args()

    if args.cron:
        _print_cron()
        return

    if args.run_now:
        func_map = {
            "data_refresh": run_data_refresh,
            "eod_data_refresh": run_eod_data_refresh,
            "credit_regime_revalidation": run_credit_regime_revalidation,
            "daily_attribution": run_daily_attribution,
            "intraday_quote_refresh": run_intraday_quote_refresh,
            "av_intraday_5min": run_av_intraday_5min,
            "strategy_lifecycle_weekly": run_strategy_lifecycle_weekly,
            "strategy_lifecycle_monthly": run_strategy_lifecycle_monthly,
            "autoresclaw_weekly": run_autoresclaw_weekly,
            "community_intel_weekly": run_community_intel_weekly,
        }
        func = func_map.get(args.run_now)
        if func is None:
            print(f"Unknown job '{args.run_now}'. Valid: {list(func_map.keys())}")
            sys.exit(1)
        func(dry_run=args.dry_run)
        return

    if args.dry_run:
        print("\nScheduled jobs:\n")
        for job in JOBS:
            print(f"  {job['trigger']} → {job['label']}")
        print()
        return

    start_scheduler()


if __name__ == "__main__":
    main()
