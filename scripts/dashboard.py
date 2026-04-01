#!/usr/bin/env python3
# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0
"""
QuantStack health dashboard.

Usage:
    python scripts/dashboard.py           # print once and exit
    python scripts/dashboard.py --watch   # live refresh every 10s (Ctrl+C to quit)

Requires: rich (pip install rich), quantstack venv active.
All queries are read-only. Never blocks or modifies state.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# DB helpers — every section gracefully degrades if DB is unreachable
# ---------------------------------------------------------------------------

def _connect():
    import psycopg2
    url = os.environ.get("TRADER_PG_URL")
    if not url:
        raise RuntimeError("TRADER_PG_URL not set")
    return psycopg2.connect(url)


def _query(sql: str, params=()) -> list:
    """Run a read-only query. Returns rows or [] on any error."""
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _scalar(sql: str, params=(), default=None):
    rows = _query(sql, params)
    return rows[0][0] if rows else default


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_system(now: datetime) -> dict:
    """System panel data."""
    sentinel = Path(
        os.environ.get("KILL_SWITCH_SENTINEL", "~/.quantstack/KILL_SWITCH_ACTIVE")
    ).expanduser()
    kill_active_file = sentinel.exists()

    # DB kill switch
    ks_value = _scalar(
        "SELECT value FROM system_state WHERE key = 'kill_switch'", default="inactive"
    )
    kill_active_db = (ks_value or "").lower() == "active"
    kill_active = kill_active_file or kill_active_db

    # tmux session
    result = subprocess.run(
        ["tmux", "has-session", "-t", "quantstack-loops"],
        capture_output=True
    )
    tmux_running = result.returncode == 0

    # Paper mode — USE_REAL_TRADING=true with ALPACA_PAPER=true means Alpaca Paper
    use_real = os.environ.get("USE_REAL_TRADING", "false").lower() == "true"
    alpaca_paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    if not use_real:
        paper_mode = True
        trading_mode = "LOCAL PAPER"
    elif alpaca_paper:
        paper_mode = True
        trading_mode = "ALPACA PAPER"
    else:
        paper_mode = False
        trading_mode = "LIVE"

    # AV quota
    today_key = f"av_daily_calls_{now.strftime('%Y-%m-%d')}"
    av_calls = _scalar(
        "SELECT value FROM system_state WHERE key = %s", (today_key,), default="0"
    )
    av_limit = int(os.environ.get("AV_DAILY_CALL_LIMIT", "25000"))
    try:
        av_calls_int = int(av_calls or 0)
    except ValueError:
        av_calls_int = 0

    return {
        "kill_active": kill_active,
        "kill_source": "file+db" if (kill_active_file and kill_active_db)
                        else ("file" if kill_active_file else ("db" if kill_active_db else None)),
        "tmux_running": tmux_running,
        "paper_mode": paper_mode,
        "trading_mode": trading_mode,
        "av_calls": av_calls_int,
        "av_limit": av_limit,
    }


def _build_loops(now: datetime) -> list[dict]:
    """Loop health: trading (expect <5 min), research (expect <2 min)."""
    rows = _query("""
        SELECT DISTINCT ON (loop_name)
            loop_name, iteration, started_at, finished_at, status, errors
        FROM loop_heartbeats
        ORDER BY loop_name, started_at DESC
    """)

    STALE_MULTIPLES = {"trading_loop": 3, "research_loop": 3}
    EXPECTED_INTERVALS = {"trading_loop": 300, "research_loop": 120}

    loops = []
    seen = set()
    for loop_name, iteration, started_at, finished_at, status, errors in rows:
        seen.add(loop_name)
        last_ts = (finished_at or started_at)
        if last_ts is None:
            age_s = None
        else:
            # Handle both tz-aware and tz-naive timestamps
            if last_ts.tzinfo is None:
                last_ts_utc = last_ts.replace(tzinfo=timezone.utc)
            else:
                last_ts_utc = last_ts
            now_utc = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
            age_s = (now_utc - last_ts_utc).total_seconds()

        expected = EXPECTED_INTERVALS.get(loop_name, 300)
        stale_mult = STALE_MULTIPLES.get(loop_name, 3)

        if age_s is None:
            health = "UNKNOWN"
        elif age_s < expected * stale_mult:
            health = "HEALTHY"
        elif age_s < expected * 10:
            health = "STALE"
        else:
            health = "DEAD"

        age_str = _fmt_age(age_s)
        loops.append({
            "name": loop_name,
            "health": health,
            "iteration": iteration,
            "age_s": age_s,
            "age_str": age_str,
            "status": status,
            "errors": errors or 0,
        })

    # Add entries for loops we haven't seen yet
    for name in ["trading_loop", "research_loop"]:
        if name not in seen:
            loops.append({"name": name, "health": "NO DATA", "iteration": 0,
                          "age_s": None, "age_str": "never", "status": None, "errors": 0})

    return loops


def _build_portfolio() -> dict:
    """Open positions and today's P&L."""
    positions = _query("""
        SELECT symbol, quantity, avg_cost, current_price, unrealized_pnl,
               side, strategy_id
        FROM positions
        ORDER BY unrealized_pnl DESC NULLS LAST
    """)

    today_equity = _query("""
        SELECT total_equity, daily_pnl, daily_return_pct, cash, open_positions
        FROM daily_equity
        ORDER BY date DESC
        LIMIT 1
    """)

    today = today_equity[0] if today_equity else None

    # Fallback to cash_balance if daily_equity is empty
    if today is None:
        cash_row = _scalar("SELECT cash FROM cash_balance WHERE id = 1", default=None)
        if cash_row is not None:
            return {
                "positions": positions,
                "total_equity": float(cash_row),
                "daily_pnl": 0.0,
                "daily_return_pct": 0.0,
                "cash": float(cash_row),
                "open_count": len(positions),
            }

    return {
        "positions": positions,
        "total_equity": today[0] if today else None,
        "daily_pnl": today[1] if today else None,
        "daily_return_pct": today[2] if today else None,
        "cash": today[3] if today else None,
        "open_count": len(positions),
    }


def _build_strategies() -> dict:
    """Strategy pipeline counts."""
    rows = _query("SELECT status, COUNT(*) FROM strategies GROUP BY status ORDER BY status")
    counts = {r[0]: r[1] for r in rows}
    return counts


def _build_recent_fills() -> list:
    rows = _query("""
        SELECT symbol, side, fill_price, filled_quantity, filled_at
        FROM fills
        ORDER BY filled_at DESC
        LIMIT 6
    """)
    return rows


def _build_queues() -> dict:
    bugs_rows = _query("""
        SELECT status, COUNT(*) FROM bugs
        WHERE status IN ('open', 'in_progress', 'fixed')
        GROUP BY status
    """)
    rq_rows = _query("""
        SELECT status, COUNT(*) FROM research_queue
        WHERE status IN ('pending', 'running', 'done', 'failed')
        GROUP BY status
    """)
    return {
        "bugs": {r[0]: r[1] for r in bugs_rows},
        "research_queue": {r[0]: r[1] for r in rq_rows},
    }


def _build_resources(now: datetime) -> dict:
    """AV quota, memory file sizes, last community intel."""
    # Memory files with their compaction limits (mirrors start.sh LIMITS)
    COMPACTION_LIMITS = {
        "trade_journal.md": 150,
        "workshop_lessons.md": 100,
        "ml_experiment_log.md": 120,
        "ml_research_program.md": 80,
        "strategy_registry.md": None,  # no limit set — monitor only
    }
    memory_dir = Path(".claude/memory")
    mem_stats = {}
    for fname, limit in COMPACTION_LIMITS.items():
        fpath = memory_dir / fname
        if fpath.exists():
            line_count = len(fpath.read_text().splitlines())
            mem_stats[fname] = {"lines": line_count, "limit": limit}

    # Last community intel log entry
    ci_log = Path("data/logs/community_intel.log")
    last_ci = None
    if ci_log.exists():
        try:
            lines = [l for l in ci_log.read_text().splitlines() if l.strip()]
            if lines:
                last_ci = lines[-1][:80]  # last non-empty line, truncated
        except Exception:
            pass

    return {
        "memory": mem_stats,
        "last_community_intel": last_ci,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_age(seconds) -> str:
    if seconds is None:
        return "never"
    s = int(seconds)
    if s < 60:
        return f"{s}s ago"
    elif s < 3600:
        return f"{s // 60}m ago"
    else:
        return f"{s // 3600}h {(s % 3600) // 60}m ago"


def _fmt_pnl(val) -> str:
    if val is None:
        return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.2f}"


def _fmt_pct(val) -> str:
    if val is None:
        return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------

def _build_dashboard(now: datetime):
    """Build a rich Group renderable for the full dashboard."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    parts = []

    system = _build_system(now)
    loops = _build_loops(now)
    portfolio = _build_portfolio()
    strategies = _build_strategies()
    fills = _build_recent_fills()
    queues = _build_queues()
    resources = _build_resources(now)

    # ── Header ──────────────────────────────────────────────────────────────
    tm = system.get("trading_mode", "PAPER" if system["paper_mode"] else "LIVE")
    mode_str = f"[yellow]{tm}[/yellow]" if system["paper_mode"] else f"[bold red]{tm}[/bold red]"
    ks_str = "[bold red]ACTIVE[/bold red]" if system["kill_active"] else "[green]inactive[/green]"
    tmux_str = "[green]running[/green]" if system["tmux_running"] else "[red]stopped[/red]"
    header_text = (
        f"[bold]QUANTSTACK STATUS[/bold]  ·  {now.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  ·  Mode: {mode_str}"
        f"  ·  tmux: {tmux_str}"
        f"  ·  Kill switch: {ks_str}"
    )
    parts.append(Panel(Text.from_markup(header_text), style="bold blue"))

    # ── Loops ───────────────────────────────────────────────────────────────
    loop_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    loop_table.add_column("Loop", style="cyan", width=20)
    loop_table.add_column("Health", width=10)
    loop_table.add_column("Last beat", width=14)
    loop_table.add_column("Iteration", width=10)
    loop_table.add_column("Errors", width=8)

    HEALTH_STYLE = {"HEALTHY": "green", "STALE": "yellow", "DEAD": "bold red",
                    "UNKNOWN": "dim", "NO DATA": "dim"}
    for lp in loops:
        health_style = HEALTH_STYLE.get(lp["health"], "white")
        loop_table.add_row(
            lp["name"],
            f"[{health_style}]{lp['health']}[/{health_style}]",
            lp["age_str"],
            str(lp["iteration"]),
            str(lp["errors"]) if lp["errors"] else "0",
        )

    parts.append(Panel(loop_table, title="[bold]Loops[/bold]", border_style="blue"))

    # ── Portfolio ────────────────────────────────────────────────────────────
    eq_line = ""
    if portfolio["total_equity"] is not None:
        eq_line = (
            f"  Equity: [bold]${portfolio['total_equity']:,.2f}[/bold]"
            f"  |  Cash: ${portfolio['cash']:,.2f}"
            f"  |  Today's P&L: [bold]{_fmt_pnl(portfolio['daily_pnl'])}[/bold]"
            f" ({_fmt_pct(portfolio['daily_return_pct'])})"
            f"  |  Open positions: {portfolio['open_count']}"
        )
    else:
        eq_line = "  [dim]No equity snapshot yet[/dim]"

    pos_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    pos_table.add_column("Symbol", width=8)
    pos_table.add_column("Side", width=6)
    pos_table.add_column("Qty", width=8, justify="right")
    pos_table.add_column("Avg Cost", width=10, justify="right")
    pos_table.add_column("Current", width=10, justify="right")
    pos_table.add_column("Unreal P&L", width=12, justify="right")
    pos_table.add_column("Strategy", width=28)

    if portfolio["positions"]:
        for sym, qty, avg, cur, upnl, side, strat in portfolio["positions"]:
            pnl_str = _fmt_pnl(upnl)
            pnl_style = "green" if (upnl or 0) >= 0 else "red"
            pos_table.add_row(
                sym,
                (side or "").upper(),
                str(qty or ""),
                f"${avg:,.2f}" if avg else "n/a",
                f"${cur:,.2f}" if cur else "n/a",
                f"[{pnl_style}]{pnl_str}[/{pnl_style}]",
                (strat or "")[:28],
            )
    else:
        pos_table.add_row("[dim]No open positions[/dim]", "", "", "", "", "", "")

    from rich.columns import Columns
    parts.append(Panel(
        Columns([Text.from_markup(eq_line), pos_table]),
        title="[bold]Portfolio[/bold]", border_style="blue"
    ))

    # ── Strategy pipeline + Recent fills (side by side) ─────────────────────
    strat_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    strat_table.add_column("Status", style="cyan")
    strat_table.add_column("Count", justify="right")
    for status in ["live", "forward_testing", "backtested", "validated", "draft", "failed"]:
        count = strategies.get(status, 0)
        if count > 0 or status in ("live", "forward_testing"):
            style = "bold green" if status == "live" else (
                "green" if status == "forward_testing" else (
                "yellow" if status == "backtested" else (
                "dim" if status in ("draft", "failed") else "white")))
            strat_table.add_row(f"[{style}]{status}[/{style}]", str(count))

    fills_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    fills_table.add_column("Side", width=5)
    fills_table.add_column("Symbol", width=7)
    fills_table.add_column("Qty", width=6, justify="right")
    fills_table.add_column("Price", width=9, justify="right")
    fills_table.add_column("When", width=12)

    if fills:
        for sym, side, price, qty, filled_at in fills:
            side_style = "green" if (side or "").lower() == "buy" else "red"
            fills_table.add_row(
                f"[{side_style}]{(side or '').upper()}[/{side_style}]",
                sym,
                str(qty or ""),
                f"${price:,.2f}" if price else "n/a",
                _fmt_age((datetime.utcnow() - filled_at.replace(tzinfo=None)).total_seconds()
                         if filled_at else None),
            )
    else:
        fills_table.add_row("[dim]No fills yet[/dim]", "", "", "", "")

    parts.append(Columns([
        Panel(strat_table, title="[bold]Strategy Pipeline[/bold]", border_style="blue", width=28),
        Panel(fills_table, title="[bold]Recent Fills[/bold]", border_style="blue"),
    ]))

    # ── Queues ───────────────────────────────────────────────────────────────
    bugs = queues["bugs"]
    rq = queues["research_queue"]
    bugs_str = (
        f"[bold]{bugs.get('open', 0)}[/bold] open  "
        f"[yellow]{bugs.get('in_progress', 0)}[/yellow] in-progress  "
        f"[green]{bugs.get('fixed', 0)}[/green] fixed"
    )
    rq_str = (
        f"[bold]{rq.get('pending', 0)}[/bold] pending  "
        f"[yellow]{rq.get('running', 0)}[/yellow] running  "
        f"[green]{rq.get('done', 0)}[/green] done  "
        f"[red]{rq.get('failed', 0)}[/red] failed"
    )
    queue_text = f"  Bugs: {bugs_str}\n  Research queue: {rq_str}"
    parts.append(Panel(Text.from_markup(queue_text), title="[bold]Queues[/bold]", border_style="blue"))

    # ── Resources ────────────────────────────────────────────────────────────
    av_pct = int(system["av_calls"] / system["av_limit"] * 100) if system["av_limit"] else 0
    av_style = "red" if av_pct > 80 else ("yellow" if av_pct > 50 else "green")
    res_lines = [
        f"  AV quota today: [{av_style}]{system['av_calls']}/{system['av_limit']} ({av_pct}%)[/{av_style}]",
    ]

    mem = resources["memory"]
    if mem:
        mem_parts = []
        for fname, stat in mem.items():
            short = fname.replace(".md", "")
            limit = stat["limit"]
            lines = stat["lines"]
            if limit:
                pct = lines / limit * 100
                style = "red" if pct > 90 else ("yellow" if pct > 75 else "dim")
                mem_parts.append(f"[{style}]{short} {lines}/{limit}L[/{style}]")
            else:
                style = "yellow" if lines > 500 else "dim"
                mem_parts.append(f"[{style}]{short} {lines}L[/{style}]")
        res_lines.append("  Memory: " + "  ".join(mem_parts))

    if resources["last_community_intel"]:
        res_lines.append(f"  Last community intel: [dim]{resources['last_community_intel']}[/dim]")

    parts.append(Panel(
        Text.from_markup("\n".join(res_lines)),
        title="[bold]Resources[/bold]", border_style="blue"
    ))

    parts.append(Text.from_markup(
        "[dim]  ./stop.sh to stop  ·  ./start.sh to start  ·  "
        "./report.sh for monthly P&L  ·  --watch for live refresh[/dim]"
    ))

    return Group(*parts)


def _render(now: datetime):
    """Print dashboard once to stdout."""
    from rich.console import Console
    Console().print(_build_dashboard(now))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QuantStack health dashboard")
    parser.add_argument(
        "--watch", action="store_true",
        help="Refresh every 10 seconds (Ctrl+C to quit)"
    )
    parser.add_argument(
        "--interval", type=int, default=10,
        help="Refresh interval in seconds for --watch mode (default: 10)"
    )
    args = parser.parse_args()

    # Load .env if present
    env_file = Path(__file__).parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    if not args.watch:
        _render(datetime.utcnow())
        return

    from rich.live import Live
    from rich.console import Console

    console = Console()
    try:
        with Live(
            _build_dashboard(datetime.utcnow()),
            console=console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            while True:
                time.sleep(args.interval)
                live.update(_build_dashboard(datetime.utcnow()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
