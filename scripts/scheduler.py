#!/usr/bin/env python3
# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantPod Trading Session Scheduler — Enhancement 4.

Triggers discrete Claude Code sessions at key market times.
Claude Code runs, does its work, and exits.  This script just ensures
it runs when it should.

Supported schedules (all times US/Eastern):
  - 09:15 — Morning routine:  /review → /meta → /trade
  - 12:30 — Mid-day check:    /review
  - 15:45 — Pre-close check:  /review
  - 17:00 Friday — Weekly reflect: /reflect

Usage:
    python scripts/scheduler.py [--dry-run]  # Print what would run
    python scripts/scheduler.py              # Start the scheduler daemon

Requirements:
    APScheduler:  pip install apscheduler>=3.10.0
    Claude Code CLI must be on PATH:  claude --version

Environment variables:
    QUANTPOD_WORKDIR: Path to QuantPod repo (defaults to CWD)
    QUANTPOD_SYMBOL:  Default trading symbol for /trade sessions (default: SPY)
    CLAUDE_CODE_CMD:  Override Claude Code command (default: claude)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("quantpod.scheduler")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKDIR = Path(os.getenv("QUANTPOD_WORKDIR", Path(__file__).parent.parent))
SYMBOL = os.getenv("QUANTPOD_SYMBOL", "SPY")
CLAUDE_CMD = os.getenv("CLAUDE_CODE_CMD", "claude")
TIMEZONE = "US/Eastern"

# Schedule definition: (hour, minute, weekday_range, skill_prompt, label)
# weekday_range: None = Mon-Fri, "friday" = Friday only
SCHEDULE: List[dict] = [
    {
        "hour": 9,
        "minute": 15,
        "weekdays": "mon-fri",
        "prompt": (
            f"Morning routine for {SYMBOL}. Run /review first to check any open "
            f"positions, then run /meta for portfolio allocation, then /trade for "
            f"today's primary symbol {SYMBOL}."
        ),
        "label": "morning_routine",
    },
    {
        "hour": 12,
        "minute": 30,
        "weekdays": "mon-fri",
        "prompt": (
            f"Mid-day position check for all open positions. Run /review. "
            f"Focus on any positions that are near stop levels."
        ),
        "label": "midday_review",
    },
    {
        "hour": 15,
        "minute": 45,
        "weekdays": "mon-fri",
        "prompt": (
            f"Pre-close review — 15 minutes to close. Run /review on all open "
            f"positions. Consider if any positions should be closed EOD. "
            f"Context: 15 minutes remaining in the trading session."
        ),
        "label": "preclose_review",
    },
    {
        "hour": 17,
        "minute": 0,
        "weekdays": "fri",
        "prompt": (
            "Weekly reflection session. Run /reflect for full weekly review: "
            "assess outcomes, update strategy registry, update regime matrix, "
            "and improve skills based on what worked and what didn't this week."
        ),
        "label": "weekly_reflect",
    },
]


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def run_autonomous_loop(dry_run: bool = False) -> None:
    """Invoke the AutonomousRunner as a subprocess (no Claude Code session needed)."""
    label = "autonomous_runner"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    cmd = [sys.executable, "-m", "quant_pod.autonomous.runner", "--paper-only"]
    if dry_run:
        cmd.append("--dry-run")

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {' '.join(cmd)}")
        return

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKDIR),
            timeout=300,  # 5-minute max — if SignalEngine takes > 5 min, something is wrong
            check=False,
        )
        if result.returncode != 0:
            logger.warning(f"'{label}' exited with code {result.returncode}")
        else:
            logger.info(f"'{label}' completed successfully")
    except subprocess.TimeoutExpired:
        logger.error(f"'{label}' timed out after 5 minutes")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_alpha_discovery(dry_run: bool = False) -> None:
    """Invoke AlphaDiscoveryEngine overnight (no Claude Code session)."""
    label = "alpha_discovery"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    cmd = [sys.executable, "-m", "quant_pod.alpha_discovery.engine"]
    if dry_run:
        cmd.append("--dry-run")

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}: {' '.join(cmd)}")
        return

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKDIR),
            timeout=3600,  # 60-minute max for overnight discovery pass
            check=False,
        )
        if result.returncode != 0:
            logger.warning(f"'{label}' exited with code {result.returncode}")
        else:
            logger.info(f"'{label}' completed successfully")
    except subprocess.TimeoutExpired:
        logger.error(f"'{label}' timed out after 60 minutes")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")


def run_session(prompt: str, label: str, dry_run: bool = False) -> None:
    """Invoke Claude Code with the given session prompt."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering session: {label}")

    cmd = [CLAUDE_CMD, "--print", prompt]

    if dry_run:
        print(f"\n[DRY RUN] Would run at {timestamp}:")
        print(f"  Label:  {label}")
        print(f"  Prompt: {prompt[:100]}...")
        return

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKDIR),
            timeout=1800,  # 30-minute max per session
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                f"Session '{label}' exited with code {result.returncode}"
            )
        else:
            logger.info(f"Session '{label}' completed successfully")
    except subprocess.TimeoutExpired:
        logger.error(f"Session '{label}' timed out after 30 minutes")
    except FileNotFoundError:
        logger.error(
            f"Claude Code CLI not found at '{CLAUDE_CMD}'. "
            "Install Claude Code or set CLAUDE_CODE_CMD env var."
        )
    except Exception as exc:
        logger.error(f"Session '{label}' failed: {exc}")


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def start_scheduler(dry_run: bool = False) -> None:
    """Start the APScheduler-based trading session scheduler."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print(
            "ERROR: APScheduler not installed.\n"
            "Install with: pip install 'apscheduler>=3.10.0'\n"
            "\n"
            "Alternatively, add these cron entries (US/Eastern timezone):\n"
        )
        _print_cron_instructions()
        sys.exit(1)

    scheduler = BlockingScheduler(timezone=TIMEZONE)

    for job in SCHEDULE:
        weekdays = job["weekdays"]
        # APScheduler day_of_week: 'mon-fri' or 'fri'
        trigger = CronTrigger(
            hour=job["hour"],
            minute=job["minute"],
            day_of_week=weekdays,
            timezone=TIMEZONE,
        )
        # Capture loop variables
        _prompt = job["prompt"]
        _label = job["label"]
        _dry = dry_run

        scheduler.add_job(
            run_session,
            trigger=trigger,
            args=[_prompt, _label, _dry],
            id=_label,
            name=_label,
            misfire_grace_time=300,  # 5-minute grace window for missed triggers
        )
        logger.info(
            f"Scheduled '{_label}' at {job['hour']:02d}:{job['minute']:02d} "
            f"({weekdays}) {TIMEZONE}"
        )

    # Autonomous runner — runs after morning routine completes and at midday
    for hour, minute, label in [(9, 35, "autonomous_09:35"), (13, 0, "autonomous_13:00")]:
        scheduler.add_job(
            run_autonomous_loop,
            CronTrigger(hour=hour, minute=minute, day_of_week="mon-fri", timezone=TIMEZONE),
            args=[dry_run],
            id=label,
            name=label,
            misfire_grace_time=300,
        )
        logger.info(f"Scheduled '{label}' at {hour:02d}:{minute:02d} (mon-fri) {TIMEZONE}")

    # AlphaDiscovery — overnight, after market close + data is settled
    scheduler.add_job(
        run_alpha_discovery,
        CronTrigger(hour=22, minute=0, day_of_week="mon-fri", timezone=TIMEZONE),
        args=[dry_run],
        id="alpha_discovery",
        name="alpha_discovery",
        misfire_grace_time=600,
    )
    logger.info(f"Scheduled 'alpha_discovery' at 22:00 (mon-fri) {TIMEZONE}")

    print(
        f"\nQuantPod Scheduler started (workdir={WORKDIR}, symbol={SYMBOL})\n"
        f"Scheduled {len(SCHEDULE)} Claude sessions + 2 autonomous loops + 1 discovery.\n"
        f"Press Ctrl+C to stop.\n"
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")


def _print_cron_instructions() -> None:
    """Print equivalent cron entries if APScheduler is not available."""
    cron_lines = []
    for job in SCHEDULE:
        h, m = job["hour"], job["minute"]
        weekdays = job["weekdays"]

        # Convert APScheduler weekday notation to cron (0=Sun, 1=Mon... 5=Fri, 6=Sat)
        if weekdays == "mon-fri":
            cron_dow = "1-5"
        elif weekdays == "fri":
            cron_dow = "5"
        else:
            cron_dow = "*"

        label = job["label"]
        cron_lines.append(
            f"# {label}\n"
            f"{m} {h} * * {cron_dow}  TZ=America/New_York  "
            f"cd {WORKDIR} && {CLAUDE_CMD} --print \"{job['prompt'][:80]}...\"\n"
        )

    print("Cron entries (add to crontab with 'crontab -e'):")
    for line in cron_lines:
        print(line)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="QuantPod trading session scheduler"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schedule without starting the scheduler or running sessions",
    )
    parser.add_argument(
        "--cron",
        action="store_true",
        help="Print equivalent cron entries and exit",
    )
    parser.add_argument(
        "--run-now",
        metavar="LABEL",
        help="Run a specific session immediately (e.g., morning_routine)",
    )
    args = parser.parse_args()

    if args.cron:
        _print_cron_instructions()
        return

    if args.run_now:
        # Python-native jobs
        if args.run_now == "autonomous_runner":
            run_autonomous_loop(dry_run=args.dry_run)
            return
        if args.run_now == "alpha_discovery":
            run_alpha_discovery(dry_run=args.dry_run)
            return
        # Claude Code session jobs
        job = next((j for j in SCHEDULE if j["label"] == args.run_now), None)
        if job is None:
            valid = [j["label"] for j in SCHEDULE] + ["autonomous_runner", "alpha_discovery"]
            print(f"Unknown label '{args.run_now}'. Valid: {valid}")
            sys.exit(1)
        run_session(job["prompt"], job["label"], dry_run=args.dry_run)
        return

    if args.dry_run:
        print("\nDry run — scheduled sessions:\n")
        for job in SCHEDULE:
            print(
                f"  {job['hour']:02d}:{job['minute']:02d} ({job['weekdays']}) "
                f"→ {job['label']}"
            )
        print()
        return

    start_scheduler(dry_run=False)


if __name__ == "__main__":
    main()
