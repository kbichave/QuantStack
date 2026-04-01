#!/usr/bin/env python3
# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
AutoResearchClaw runner — translates research_queue tasks into AutoResearchClaw
invocations and writes results back to the strategies table / models/ directory.

Triggered by the scheduler Sunday 20:00 ET. Processes the top-N pending tasks
by priority. Requires Docker for sandboxed code execution.

Usage:
    python scripts/autoresclaw_runner.py           # process top 3 tasks
    python scripts/autoresclaw_runner.py --dry-run # print tasks without running
    python scripts/autoresclaw_runner.py --limit 5 # process top 5 tasks
    python scripts/autoresclaw_runner.py --task-id <uuid>  # run a specific task

Exit codes:
    0 — all processed tasks completed or were skipped
    1 — fatal setup error (Docker missing, DB unreachable)
    2 — one or more tasks failed (partial success is still exit 2)

Design invariants:
    - Never modifies risk_gate.py, kill_switch.py, or db.py.
    - Output artifacts go to reports/autoresclaw/YYYY-MM-DD/ and models/.
    - Tasks are marked 'running' before execution and 'done'/'failed' after.
    - If the process is killed mid-task, the task stays 'running' and will be
      retried on the next weekly run (idempotent via task_id).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path

from quantstack.db import open_db

logger = logging.getLogger("autoresclaw_runner")

WORKDIR = Path(os.getenv("QUANTSTACK_WORKDIR", Path(__file__).parent.parent))
OUTPUT_BASE = WORKDIR / "reports" / "autoresclaw"
MODELS_DIR = WORKDIR / "models"

# AutoResearchClaw CLI — installed separately via npm/pip.
# CLI: researchclaw run --topic "<prompt>" --auto-approve --agent claude-code
# Set AUTORESCLAW_CMD env var to override (e.g. for a custom install path).
AUTORESCLAW_CMD = os.getenv("AUTORESCLAW_CMD", "researchclaw")

# Timeout per task in seconds (AutoResearchClaw runs can take a long time).
TASK_TIMEOUT_SECONDS = int(os.getenv("AUTORESCLAW_TIMEOUT", "3600"))


# ---------------------------------------------------------------------------
# Task prompt builders — translate QuantStack events → ARC task prompts
# ---------------------------------------------------------------------------


def _build_prompt_ml_arch_search(ctx: dict) -> str:
    symbol = ctx.get("symbol", "unknown")
    psi = ctx.get("psi", 0.0)
    features = ctx.get("drifted_features", [])
    return (
        f"ML Architecture Search for {symbol}\n\n"
        f"Context: SignalEngine detected CRITICAL feature drift (PSI={psi:.3f}) "
        f"in features: {features}. The current XGBoost/LightGBM ensemble may no longer "
        f"be predictive for this symbol.\n\n"
        f"Task:\n"
        f"1. Hypothesis: which new features or model architecture would restore predictive power?\n"
        f"2. Generate training code using the existing feature pipeline in "
        f"src/quantstack/core/features/\n"
        f"3. Execute: train candidate model on historical data (avoid look-ahead bias)\n"
        f"4. Evaluate: OOS Sharpe, IC, max drawdown vs. current champion\n"
        f"5. Decision: PROCEED (register in models/) / REFINE (iterate) / PIVOT (different approach)\n\n"
        f"Output: if PROCEED, write model artifact to models/arc/{symbol}_{{date}}.pkl and "
        f"insert into ml_experiments table with status='arc_candidate'."
    )


def _build_prompt_rl_env_design(ctx: dict) -> str:
    env_type = ctx.get("env_type", "sizing")
    symbol = ctx.get("symbol", "SPY")
    return (
        f"RL Gym Environment Design: {env_type}\n\n"
        f"Context: Design and train a new RL environment for {env_type} decisions "
        f"on {symbol}. The existing environments in src/quantstack/finrl/environments.py "
        f"(ExecutionEnv, SizingEnv, AlphaSelectionEnv) are the reference.\n\n"
        f"Task:\n"
        f"1. Design a new gymnasium environment class for {env_type}\n"
        f"2. Generate training code using StableBaselines3 (PPO or SAC)\n"
        f"3. Execute: train for 100k steps on historical data\n"
        f"4. Evaluate: shadow performance on held-out period vs. rule-based baseline\n"
        f"5. Decision: PROCEED (register) / REFINE (more steps/reward shaping) / PIVOT\n\n"
        f"Output: if PROCEED, add environment class to src/quantstack/finrl/environments.py "
        f"and register via finrl_promote_model MCP tool."
    )


def _build_prompt_bug_fix(ctx: dict, task_id: str = "unknown") -> str:
    tool_name = ctx.get("tool_name", ctx.get("symbol", "unknown"))
    loop_name = ctx.get("loop_name", "trading_loop")
    consecutive = ctx.get("consecutive_errors", 0)
    last_error = ctx.get("last_error", "")
    stack_trace = ctx.get("stack_trace", "")

    # Fallback: trade-loss context (older format triggered by trade-reflector)
    pnl = ctx.get("realized_pnl_pct")
    trade_context = ""
    if pnl is not None:
        trade_context = (
            f"\nTrade context:\n"
            f"  P&L: {pnl:.1f}%  |  "
            f"  Regime at entry: {ctx.get('regime_at_entry', '?')}  |  "
            f"  Regime at exit: {ctx.get('regime_at_exit', '?')}\n"
        )

    output_dir = str(OUTPUT_BASE / datetime.now().strftime("%Y-%m-%d") / task_id)

    return (
        f"# Autonomous Bug Fix: {tool_name}\n\n"
        f"## Context\n"
        f"Tool `{tool_name}` in `{loop_name}` has failed {consecutive} consecutive times.\n"
        f"\nLast error:\n```\n{last_error}\n```\n"
        f"\nStack trace:\n```\n{stack_trace}\n```\n"
        f"{trade_context}\n"
        f"## Your task\n\n"
        f"1. **Locate** the failing code. Search `src/quantstack/` for `{tool_name}`.\n"
        f"2. **Reproduce** the failure — read the code and understand why the error "
        f"occurs given the stack trace.\n"
        f"3. **Fix it** — directly edit the source file(s) in `src/quantstack/`. "
        f"Write the smallest correct change. Do not refactor surrounding code.\n"
        f"4. **Validate** the fix:\n"
        f"   - Run `python3 -m py_compile <changed_file>` for each edited file.\n"
        f"   - Run `python3 -c 'import <module>'` to confirm it imports cleanly.\n"
        f"   - If tests exist for this module, run them.\n"
        f"5. **Write summary** to `{output_dir}/fix_summary.md` with:\n"
        f"   - Root cause (one sentence)\n"
        f"   - Files changed (list, one per line)\n"
        f"   - What the fix does\n"
        f"   - Validation results\n"
        f"   - Confidence: high / medium / low\n\n"
        f"## Hard constraints\n\n"
        f"- NEVER modify `risk_gate.py`, `kill_switch.py`, or `db.py`. "
        f"If the fix requires those files, write `## Requires Human Review` in the "
        f"summary with the reason, and make NO code changes.\n"
        f"- If confidence is low, write `## Confidence: low` — the auto-patcher will "
        f"revert the changes and wait for human review.\n"
        f"- Do not add new dependencies or modify `pyproject.toml`.\n"
    )


def _build_prompt_strategy_hypothesis(ctx: dict) -> str:
    domain = ctx.get("domain", "equity")
    gap = ctx.get("gap", "unspecified coverage gap")
    return (
        f"Strategy Hypothesis: {domain}\n\n"
        f"Context: Research loop identified a coverage gap: {gap}\n\n"
        f"Task:\n"
        f"1. Literature: what strategies exist for this gap? (use web search)\n"
        f"2. Hypothesis: propose a specific testable entry/exit rule\n"
        f"3. Backtest: implement and run walk-forward test (purged CV, min 21 folds)\n"
        f"4. Evaluate: Sharpe > 0.5, max DD < 15%, PBO > 0.55\n"
        f"5. Decision: PROCEED (insert to strategies as 'draft') / PIVOT\n\n"
        f"Output: if PROCEED, insert strategy into strategies table with "
        f"status='draft' and full parameter JSON."
    )


_PROMPT_BUILDERS = {
    "ml_arch_search": _build_prompt_ml_arch_search,
    "rl_env_design": _build_prompt_rl_env_design,
    "bug_fix": _build_prompt_bug_fix,
    "strategy_hypothesis": _build_prompt_strategy_hypothesis,
}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def check_docker() -> bool:
    """Return True if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def fetch_pending_tasks(conn, limit: int) -> list[dict]:
    """Fetch top-N pending tasks ordered by priority DESC, created_at ASC."""
    rows = conn.execute(
        """
        SELECT task_id, task_type, priority, topic, context_json, source, created_at
        FROM research_queue
        WHERE status = 'pending'
        ORDER BY priority DESC, created_at ASC
        LIMIT %s
        """,
        [limit],
    ).fetchall()
    return [
        {
            "task_id": r[0],
            "task_type": r[1],
            "priority": r[2],
            "topic": r[3] or "",
            "context": json.loads(r[4]) if isinstance(r[4], str) else (r[4] or {}),
            "source": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


def fetch_task_by_id(conn, task_id: str) -> dict | None:
    row = conn.execute(
        """
        SELECT task_id, task_type, priority, topic, context_json, source, created_at
        FROM research_queue WHERE task_id = %s
        """,
        [task_id],
    ).fetchone()
    if not row:
        return None
    return {
        "task_id": row[0],
        "task_type": row[1],
        "priority": row[2],
        "topic": row[3] or "",
        "context": json.loads(row[4]) if isinstance(row[4], str) else (row[4] or {}),
        "source": row[5],
        "created_at": row[6],
    }


def mark_task(conn, task_id: str, status: str, result_path: str = "", error: str = "") -> None:
    now = datetime.now()
    if status == "running":
        conn.execute(
            "UPDATE research_queue SET status=%s, started_at=%s WHERE task_id=%s",
            [status, now, task_id],
        )
    else:
        conn.execute(
            """
            UPDATE research_queue
            SET status=%s, completed_at=%s, result_path=%s, error_message=%s
            WHERE task_id=%s
            """,
            [status, now, result_path or None, error[:2000] if error else None, task_id],
        )


def run_task(task: dict, output_dir: Path, dry_run: bool = False) -> tuple[bool, str]:
    """
    Run a single AutoResearchClaw task.

    Returns (success, result_path_or_error).
    """
    task_id = task["task_id"]
    task_type = task["task_type"]
    ctx = task["context"]

    # Use topic from DB if set (written by newer inserters), else build from context.
    topic = task.get("topic") or ""
    if not topic:
        builder = _PROMPT_BUILDERS.get(task_type)
        if builder is None:
            return False, f"No prompt builder for task_type={task_type!r}"
        # Pass task_id to bug_fix builder so it knows where to write the summary.
        if task_type == "bug_fix":
            topic = builder(ctx, task_id=task_id)
        else:
            topic = builder(ctx)

    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / task_id
    result_dir.mkdir(parents=True, exist_ok=True)

    # Write topic to file as a record (not passed to CLI — ARC takes it inline).
    prompt_file = result_dir / "task_prompt.md"
    prompt_file.write_text(topic)

    # CLI: researchclaw run --topic "<topic>" --auto-approve --agent claude-code
    cmd = [
        AUTORESCLAW_CMD,
        "run",
        "--topic", topic,
        "--auto-approve",
        "--agent", "claude-code",
    ]

    logger.info(f"[ARC] Task {task_id} ({task_type}) → {result_dir}")
    logger.info(f"[ARC] Command: {AUTORESCLAW_CMD} run --topic '<prompt>' --auto-approve --agent claude-code")

    if dry_run:
        logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
        logger.info(f"[DRY RUN] Prompt:\n{prompt[:500]}...")
        return True, str(result_dir)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKDIR),
            timeout=TASK_TIMEOUT_SECONDS,
            capture_output=False,  # stream output to console
        )
        if result.returncode == 0:
            logger.info(f"[ARC] Task {task_id} completed successfully → {result_dir}")
            # For bug_fix tasks: validate and apply the edits ARC made directly to src/.
            if task_type == "bug_fix":
                _apply_bug_fix(task_id, result_dir)
            return True, str(result_dir)
        else:
            err = f"AutoResearchClaw exited with code {result.returncode}"
            logger.error(f"[ARC] Task {task_id} failed: {err}")
            return False, err
    except subprocess.TimeoutExpired:
        err = f"Task timed out after {TASK_TIMEOUT_SECONDS}s"
        logger.error(f"[ARC] Task {task_id}: {err}")
        return False, err
    except FileNotFoundError:
        err = (
            f"AutoResearchClaw CLI not found at {AUTORESCLAW_CMD!r}. "
            "Install it or set AUTORESCLAW_CMD env var."
        )
        logger.error(f"[ARC] {err}")
        return False, err


_PROTECTED_FILES = frozenset(["risk_gate.py", "kill_switch.py", "db.py"])


def _apply_bug_fix(task_id: str, result_dir: Path) -> None:
    """
    Validate and commit ARC's direct source edits after a bug_fix task.

    ARC edits files in-place (via Claude Code --auto-approve).  This function:
      1. Reads the fix_summary.md to check confidence and protected-file flags.
      2. Gets the list of changed files from git diff.
      3. Rejects the patch (git checkout) if:
         - Confidence is "low" in the summary
         - Any protected file was touched
         - Any changed .py file fails py_compile
         - Summary says "Requires Human Review"
      4. If valid: commits with a clear message and restarts affected loops.
    """
    summary_path = result_dir / "fix_summary.md"
    summary = summary_path.read_text() if summary_path.exists() else ""

    # Read confidence + human-review flags from summary
    low_confidence = "## confidence: low" in summary.lower()
    needs_review = "## requires human review" in summary.lower()

    if needs_review:
        logger.warning(
            f"[auto_patch] Task {task_id}: ARC flagged 'Requires Human Review' — "
            "reverting all changes and writing to session_handoffs.md"
        )
        _revert_and_note(task_id, summary, reason="human_review_required")
        return

    if low_confidence:
        logger.warning(
            f"[auto_patch] Task {task_id}: ARC reported low confidence — reverting"
        )
        _revert_and_note(task_id, summary, reason="low_confidence")
        return

    # Get changed files from git
    try:
        diff_result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(WORKDIR), capture_output=True, text=True, timeout=10,
        )
        changed_files = [f.strip() for f in diff_result.stdout.splitlines() if f.strip()]
    except Exception as exc:
        logger.error(f"[auto_patch] git diff failed: {exc}")
        return

    if not changed_files:
        logger.info(f"[auto_patch] Task {task_id}: no files changed — nothing to apply")
        return

    # Reject if protected files were touched
    protected_touched = [
        f for f in changed_files
        if any(f.endswith(p) for p in _PROTECTED_FILES)
    ]
    if protected_touched:
        logger.error(
            f"[auto_patch] Task {task_id}: ARC touched protected files {protected_touched} — reverting"
        )
        _revert_and_note(task_id, summary, reason=f"protected_files_touched: {protected_touched}")
        return

    # Syntax-check all changed Python files
    for rel_path in changed_files:
        if not rel_path.endswith(".py"):
            continue
        abs_path = WORKDIR / rel_path
        if not abs_path.exists():
            continue
        check = subprocess.run(
            ["python3", "-m", "py_compile", str(abs_path)],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            logger.error(
                f"[auto_patch] Task {task_id}: syntax error in {rel_path} — reverting\n"
                f"{check.stderr}"
            )
            _revert_and_note(task_id, summary, reason=f"syntax_error in {rel_path}: {check.stderr[:200]}")
            return

    # All checks passed — commit
    files_str = ", ".join(changed_files)
    commit_msg = f"fix(auto): {task_id} — {files_str[:120]}"
    commit_hash = ""
    try:
        subprocess.run(["git", "add", *changed_files], cwd=str(WORKDIR), check=True, timeout=10)
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=str(WORKDIR), check=True, timeout=10,
        )
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(WORKDIR), capture_output=True, text=True, timeout=5,
        )
        commit_hash = hash_result.stdout.strip()
        logger.info(f"[auto_patch] Task {task_id}: committed {commit_hash} — {files_str}")
    except subprocess.CalledProcessError as exc:
        logger.error(f"[auto_patch] git commit failed: {exc}")
        return

    # Mark bug as fixed in the bugs table
    _update_bug_status(task_id, "fixed", commit_hash=commit_hash, fix_summary=summary[:800])

    # Write note to session_handoffs.md
    handoffs = WORKDIR / ".claude" / "memory" / "session_handoffs.md"
    with open(handoffs, "a") as f:
        f.write(
            f"\n## Auto-fix applied — {datetime.now().strftime('%Y-%m-%d %H:%M')} "
            f"(task {task_id}, commit {commit_hash})\n"
            f"Files changed: {files_str}\n"
            f"Summary: {summary[:400]}\n"
        )

    # Restart affected loops via tmux so they pick up the new code
    _restart_loops_after_fix(changed_files)


def _update_bug_status(
    task_id: str,
    status: str,
    commit_hash: str = "",
    fix_summary: str = "",
) -> None:
    """
    Update bugs + research_queue rows after a fix attempt.

    Looks up bug_id via research_queue.context_json->>'bug_id'.
    Updates bugs.status, fixed_at, fix_commit, fix_summary.
    Marks research_queue task as 'done' (success) or 'failed' (reverted).
    """
    try:
        conn = open_db()
        # Fetch bug_id from context_json stored when the task was queued.
        row = conn.execute(
            "SELECT context_json FROM research_queue WHERE task_id = %s",
            [task_id],
        ).fetchone()
        if not row:
            logger.warning(f"[auto_patch] _update_bug_status: task {task_id} not found")
            conn.close()
            return

        ctx = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
        bug_id = ctx.get("bug_id")

        if bug_id:
            if status == "fixed":
                conn.execute(
                    """
                    UPDATE bugs
                    SET status = %s, fixed_at = NOW(), fix_commit = %s, fix_summary = %s
                    WHERE bug_id = %s
                    """,
                    [status, commit_hash or None, fix_summary or None, bug_id],
                )
            else:
                # reverted / wont_fix — clear in_progress back to open so it can retry
                conn.execute(
                    "UPDATE bugs SET status = 'open', arc_task_id = NULL WHERE bug_id = %s",
                    [bug_id],
                )

        # Mark the queue task terminal
        rq_status = "done" if status == "fixed" else "failed"
        conn.execute(
            "UPDATE research_queue SET status = %s, completed_at = NOW() WHERE task_id = %s",
            [rq_status, task_id],
        )
        conn.commit()
        conn.close()
        logger.info(f"[auto_patch] Bug {bug_id} → {status}, task {task_id} → {rq_status}")
    except Exception as exc:
        logger.error(f"[auto_patch] _update_bug_status failed: {exc}")


def _revert_and_note(task_id: str, summary: str, reason: str) -> None:
    """Revert ARC's changes and write a note to session_handoffs.md."""
    try:
        subprocess.run(
            ["git", "checkout", "--", "src/"],
            cwd=str(WORKDIR), timeout=10,
        )
    except Exception as exc:
        logger.error(f"[auto_patch] git checkout revert failed: {exc}")

    _update_bug_status(task_id, "reverted")

    handoffs = WORKDIR / ".claude" / "memory" / "session_handoffs.md"
    with open(handoffs, "a") as f:
        f.write(
            f"\n## Auto-fix REVERTED — {datetime.now().strftime('%Y-%m-%d %H:%M')} "
            f"(task {task_id})\n"
            f"Reason: {reason}\n"
            f"ARC summary: {summary[:400]}\n"
            f"Action required: manually review and apply fix, or re-investigate.\n"
        )


def _restart_loops_after_fix(changed_files: list[str]) -> None:
    """Interrupt running loop windows in tmux so they restart with new code."""
    # Determine which loops are likely affected
    targets = []
    src_files = " ".join(changed_files)
    # Heuristic: data/signal/execution changes affect trading; research/ml affect research
    if any(k in src_files for k in ("signal", "execution", "data", "coordination")):
        targets.append(("quantstack-loops:trading", "cat prompts/trading_loop.md | claude --model sonnet"))
    if any(k in src_files for k in ("research", "ml", "models", "features")):
        targets.append(("quantstack-loops:research", "cat prompts/research_loop.md | claude --model haiku"))
    # If unclear, restart both
    if not targets:
        targets = [
            ("quantstack-loops:trading", "cat prompts/trading_loop.md | claude --model sonnet"),
            ("quantstack-loops:research", "cat prompts/research_loop.md | claude --model haiku"),
        ]

    for tmux_target, restart_cmd in targets:
        try:
            # Send C-c to stop the current iteration, then restart
            subprocess.run(
                ["tmux", "send-keys", "-t", tmux_target, "C-c", ""],
                timeout=5,
            )
            time.sleep(1)
            subprocess.run(
                ["tmux", "send-keys", "-t", tmux_target, restart_cmd, "Enter"],
                timeout=5,
            )
            logger.info(f"[auto_patch] Restarted {tmux_target}")
        except Exception as exc:
            logger.warning(f"[auto_patch] Could not restart {tmux_target}: {exc}")


def process_tasks(
    tasks: list[dict],
    conn,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Process tasks sequentially. Returns (succeeded, failed)."""
    today = date.today().isoformat()
    output_dir = OUTPUT_BASE / today
    succeeded = 0
    failed = 0

    for task in tasks:
        task_id = task["task_id"]
        logger.info(
            f"[ARC] Processing task {task_id} "
            f"type={task['task_type']} priority={task['priority']} "
            f"source={task['source']}"
        )

        if not dry_run:
            mark_task(conn, task_id, "running")

        success, result = run_task(task, output_dir, dry_run=dry_run)

        if not dry_run:
            mark_task(
                conn, task_id,
                "done" if success else "failed",
                result_path=result if success else "",
                error=result if not success else "",
            )

        if success:
            succeeded += 1
        else:
            failed += 1

    return succeeded, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="AutoResearchClaw runner — processes research_queue tasks"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print tasks and prompts without executing")
    parser.add_argument("--limit", type=int, default=3,
                        help="Max tasks to process (default: 3)")
    parser.add_argument("--task-id", metavar="UUID",
                        help="Run a specific task by ID (ignores --limit)")
    args = parser.parse_args()

    # Check Docker (required for sandboxed execution)
    if not args.dry_run and not check_docker():
        logger.error(
            "Docker is not available. AutoResearchClaw requires Docker for "
            "sandboxed code execution. Install Docker and ensure it is running."
        )
        sys.exit(1)

    # Connect to DB
    try:
        conn = open_db()
    except Exception as exc:
        logger.error(f"Cannot connect to PostgreSQL: {exc}")
        sys.exit(1)

    # Fetch tasks
    if args.task_id:
        task = fetch_task_by_id(conn, args.task_id)
        if task is None:
            logger.error(f"Task {args.task_id!r} not found in research_queue")
            sys.exit(1)
        tasks = [task]
    else:
        tasks = fetch_pending_tasks(conn, limit=args.limit)

    if not tasks:
        logger.info("No pending tasks in research_queue. Nothing to do.")
        conn.close()
        return

    logger.info(f"Processing {len(tasks)} task(s){' [DRY RUN]' if args.dry_run else ''}...")
    succeeded, failed = process_tasks(tasks, conn, dry_run=args.dry_run)

    conn.close()

    logger.info(f"Done. succeeded={succeeded} failed={failed}")
    if failed > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
