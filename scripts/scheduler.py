#!/usr/bin/env python3
# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantStack Scheduler — deterministic jobs that run on a cron schedule.

The research loop and trading loop (tmux) handle all LLM-based work.
This scheduler runs ONLY the LLM-free Python jobs that need fixed timing:

  08:00 — Daily data refresh (Alpha Vantage cache update)
  09:35 — AutonomousRunner (deterministic signal → risk gate → broker)
  13:00 — AutonomousRunner (mid-day pass)

No Claude Code sessions. No /review or /trade skills. Those are the loops' job.

Usage:
    python scripts/scheduler.py [--dry-run]  # Print schedule without running
    python scripts/scheduler.py              # Start the scheduler daemon
    python scripts/scheduler.py --run-now data_refresh
    python scripts/scheduler.py --run-now autonomous_runner
    python scripts/scheduler.py --cron       # Print equivalent cron entries

Requirements:
    APScheduler: pip install quantpod[scheduler]
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("quantstack.scheduler")

WORKDIR = Path(os.getenv("QUANTPOD_WORKDIR", Path(__file__).parent.parent))
TIMEZONE = "US/Eastern"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def run_data_refresh(dry_run: bool = False) -> None:
    """Refresh Alpha Vantage cache — OHLCV, macro, news, insider, fundamentals."""
    label = "data_refresh"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    cmd = [
        sys.executable, "scripts/acquire_historical_data.py",
        "--phases", "ohlcv_daily", "macro", "news", "insider", "fundamentals",
    ]

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {' '.join(cmd)}")
        return

    try:
        result = subprocess.run(
            cmd, cwd=str(WORKDIR), timeout=1800, check=False,
        )
        if result.returncode != 0:
            logger.warning(f"'{label}' exited with code {result.returncode}")
        else:
            logger.info(f"'{label}' completed successfully")
    except subprocess.TimeoutExpired:
        logger.error(f"'{label}' timed out after 30 minutes")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_autonomous_loop(dry_run: bool = False) -> None:
    """AutonomousRunner — deterministic signal → risk gate → broker. No LLM."""
    label = "autonomous_runner"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    cmd = [sys.executable, "-m", "quantstack.autonomous.runner", "--paper-only"]

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {' '.join(cmd)}")
        return

    try:
        result = subprocess.run(
            cmd, cwd=str(WORKDIR), timeout=300, check=False,
        )
        if result.returncode != 0:
            logger.warning(f"'{label}' exited with code {result.returncode}")
        else:
            logger.info(f"'{label}' completed successfully")
    except subprocess.TimeoutExpired:
        logger.error(f"'{label}' timed out after 5 minutes")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

JOBS = [
    {"hour": 8, "minute": 0, "weekdays": "mon-fri", "func": run_data_refresh, "label": "data_refresh_08:00"},
    {"hour": 9, "minute": 35, "weekdays": "mon-fri", "func": run_autonomous_loop, "label": "autonomous_09:35"},
    {"hour": 13, "minute": 0, "weekdays": "mon-fri", "func": run_autonomous_loop, "label": "autonomous_13:00"},
]


def start_scheduler(dry_run: bool = False) -> None:
    """Start the APScheduler daemon."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print(
            "ERROR: APScheduler not installed.\n"
            "Install with: pip install 'apscheduler>=3.10.0'\n\n"
            "Equivalent cron entries:\n"
        )
        _print_cron()
        sys.exit(1)

    scheduler = BlockingScheduler(timezone=TIMEZONE)

    for job in JOBS:
        trigger = CronTrigger(
            hour=job["hour"], minute=job["minute"],
            day_of_week=job["weekdays"], timezone=TIMEZONE,
        )
        scheduler.add_job(
            job["func"], trigger=trigger, args=[dry_run],
            id=job["label"], name=job["label"],
            misfire_grace_time=300,
        )
        logger.info(
            f"Scheduled '{job['label']}' at {job['hour']:02d}:{job['minute']:02d} "
            f"({job['weekdays']}) {TIMEZONE}"
        )

    print(
        f"\nQuantStack Scheduler started (workdir={WORKDIR})\n"
        f"Scheduled {len(JOBS)} jobs. Press Ctrl+C to stop.\n\n"
        f"  08:00 — Data refresh (Alpha Vantage cache)\n"
        f"  09:35 — AutonomousRunner (deterministic execution)\n"
        f"  13:00 — AutonomousRunner (mid-day pass)\n"
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


def _print_cron() -> None:
    """Print equivalent cron entries."""
    print("# QuantStack scheduled jobs (add to crontab, TZ=America/New_York)")
    print(f"0  8 * * 1-5  cd {WORKDIR} && python scripts/acquire_historical_data.py --phases ohlcv_daily macro news insider fundamentals")
    print(f"35 9 * * 1-5  cd {WORKDIR} && python -m quantstack.autonomous.runner --paper-only")
    print(f"0 13 * * 1-5  cd {WORKDIR} && python -m quantstack.autonomous.runner --paper-only")


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
            "autonomous_runner": run_autonomous_loop,
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
            print(f"  {job['hour']:02d}:{job['minute']:02d} ({job['weekdays']}) → {job['label']}")
        print()
        return

    start_scheduler()


if __name__ == "__main__":
    main()
