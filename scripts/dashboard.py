#!/usr/bin/env python3
# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0
"""
QuantStack health dashboard.

Usage:
    python scripts/dashboard.py           # print once and exit
    python scripts/dashboard.py --watch   # live refresh every 10s (Ctrl+C to quit)
    ./status.sh                           # same (bash wrapper)

Requires: rich (pip install rich), quantstack venv active.
All queries are read-only. Never blocks or modifies state.
"""

from __future__ import annotations

import argparse
import json
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
    # Rewrite Docker-internal hostname for host-side access.
    # Docker Compose maps postgres container port 5432 → host port 5434
    # to avoid conflict with host-local PostgreSQL on 5432.
    url = url.replace("@postgres:5432", "@127.0.0.1:5434")
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

    # Docker containers
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, timeout=5,
        )
        containers = []
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        containers = []

    # Paper mode
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
        "containers": containers,
        "paper_mode": paper_mode,
        "trading_mode": trading_mode,
        "av_calls": av_calls_int,
        "av_limit": av_limit,
    }


def _build_heartbeats() -> list[dict]:
    """Read heartbeat files from inside Docker containers."""
    CONTAINER_MAP = {
        "trading": "quantstack-trading-graph",
        "research": "quantstack-research-graph",
        "supervisor": "quantstack-supervisor-graph",
    }
    THRESHOLDS = {"trading": 120, "research": 600, "supervisor": 360}

    results = []
    for graph, container in CONTAINER_MAP.items():
        threshold = THRESHOLDS[graph]
        try:
            result = subprocess.run(
                ["docker", "exec", container, "cat", f"/tmp/{graph}-heartbeat"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                ts = float(result.stdout.strip())
                age = time.time() - ts
                health = "OK" if age < threshold else "STALE"
                results.append({"name": graph, "health": health, "age": age,
                                "threshold": threshold})
                continue
        except Exception:
            pass

        # DB fallback
        rows = _query(
            "SELECT started_at FROM loop_heartbeats "
            "WHERE loop_name LIKE %s ORDER BY started_at DESC LIMIT 1",
            (graph + "%",)
        )
        if rows and rows[0][0]:
            ts = rows[0][0]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            health = "OK" if age < threshold else "STALE"
            results.append({"name": graph, "health": health, "age": age,
                            "threshold": threshold})
        else:
            results.append({"name": graph, "health": "NO DATA", "age": None,
                            "threshold": threshold})

    return results


def _build_loops(now: datetime) -> list[dict]:
    """Loop health from loop_heartbeats DB table."""
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

        loops.append({
            "name": loop_name, "health": health, "iteration": iteration,
            "age_s": age_s, "age_str": _fmt_age(age_s),
            "status": status, "errors": errors or 0,
        })

    for name in ["trading_loop", "research_loop"]:
        if name not in seen:
            loops.append({"name": name, "health": "NO DATA", "iteration": 0,
                          "age_s": None, "age_str": "never", "status": None, "errors": 0})

    return loops


def _build_portfolio() -> dict:
    """Open positions and today's P&L."""
    # positions table has no 'status' column — all rows are open (closed go to closed_trades)
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


def _build_strategies() -> tuple[dict, list]:
    """Strategy pipeline counts + individual strategies."""
    rows = _query("SELECT status, COUNT(*) FROM strategies GROUP BY status ORDER BY status")
    counts = {r[0]: r[1] for r in rows}

    details = _query("""
        SELECT name, status, instrument_type, time_horizon, symbol, created_at
        FROM strategies
        ORDER BY status, updated_at DESC
        LIMIT 20
    """)
    return counts, details


def _build_recent_fills() -> list:
    rows = _query("""
        SELECT symbol, side, fill_price, filled_quantity, filled_at
        FROM fills
        ORDER BY filled_at DESC
        LIMIT 6
    """)
    return rows


def _build_closed_trades() -> list[dict]:
    """Recent closed trades with realized P&L."""
    rows = _query("""
        SELECT symbol, side, quantity, entry_price, exit_price,
               realized_pnl, holding_days, strategy_id, exit_reason,
               closed_at
        FROM closed_trades
        ORDER BY closed_at DESC
        LIMIT 15
    """)
    return [
        {"symbol": r[0], "side": r[1], "qty": r[2], "entry": r[3], "exit": r[4],
         "pnl": r[5], "days": r[6] or 0, "strategy": r[7] or "",
         "reason": r[8] or "", "closed_at": r[9]}
        for r in rows
    ]


def _build_strategy_performance() -> list[dict]:
    """Per-strategy win rate and P&L from closed trades."""
    rows = _query("""
        SELECT
            COALESCE(strategy_id, 'unknown') AS strat,
            COUNT(*)                         AS trades,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(realized_pnl)                AS total_pnl,
            AVG(realized_pnl)                AS avg_pnl,
            MAX(realized_pnl)                AS best,
            MIN(realized_pnl)                AS worst,
            AVG(holding_days)                AS avg_days
        FROM closed_trades
        GROUP BY COALESCE(strategy_id, 'unknown')
        ORDER BY SUM(realized_pnl) DESC
    """)
    return [
        {"strategy": r[0], "trades": r[1], "wins": r[2],
         "win_rate": (r[2] / r[1] * 100) if r[1] else 0,
         "total_pnl": r[3], "avg_pnl": r[4],
         "best": r[5], "worst": r[6], "avg_days": r[7] or 0}
        for r in rows
    ]


def _build_research_details() -> list[dict]:
    """Research queue items with details."""
    rows = _query("""
        SELECT task_type, priority, topic, status, source,
               created_at, started_at, completed_at, error_message
        FROM research_queue
        ORDER BY
            CASE status
                WHEN 'running' THEN 0
                WHEN 'pending' THEN 1
                WHEN 'failed'  THEN 2
                WHEN 'done'    THEN 3
            END,
            priority DESC,
            created_at DESC
        LIMIT 20
    """)
    return [
        {"type": r[0], "priority": r[1], "topic": (r[2] or "")[:40],
         "status": r[3], "source": r[4] or "",
         "created_at": r[5], "started_at": r[6],
         "completed_at": r[7], "error": (r[8] or "")[:50]}
        for r in rows
    ]


def _build_signals() -> list:
    """Active signals from signal_state."""
    return _query("""
        SELECT symbol, action, confidence, position_size_pct,
               stop_loss, take_profit, generated_at, expires_at
        FROM signal_state
        ORDER BY confidence DESC NULLS LAST
        LIMIT 10
    """)


def _build_regime() -> dict | None:
    """Current market regime from loop_iteration_context."""
    val = _scalar("""
        SELECT context_json FROM loop_iteration_context
        WHERE context_key = 'current_regime'
        ORDER BY updated_at DESC LIMIT 1
    """)
    if val is None:
        return None
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return {"regime": val}
    return val if isinstance(val, dict) else {"regime": str(val)}


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
    COMPACTION_LIMITS = {
        "trade_journal.md": 150,
        "workshop_lessons.md": 100,
        "ml_experiment_log.md": 120,
        "ml_research_program.md": 80,
        "strategy_registry.md": None,
    }
    memory_dir = Path(".claude/memory")
    mem_stats = {}
    for fname, limit in COMPACTION_LIMITS.items():
        fpath = memory_dir / fname
        if fpath.exists():
            line_count = len(fpath.read_text().splitlines())
            mem_stats[fname] = {"lines": line_count, "limit": limit}

    ci_log = Path("data/logs/community_intel.log")
    last_ci = None
    if ci_log.exists():
        try:
            lines = [l for l in ci_log.read_text().splitlines() if l.strip()]
            if lines:
                last_ci = lines[-1][:80]
        except Exception:
            pass

    return {"memory": mem_stats, "last_community_intel": last_ci}


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
    from rich.columns import Columns
    from rich import box

    parts = []

    system = _build_system(now)
    heartbeats = _build_heartbeats()
    loops = _build_loops(now)
    portfolio = _build_portfolio()
    strat_counts, strat_details = _build_strategies()
    fills = _build_recent_fills()
    closed_trades = _build_closed_trades()
    strat_perf = _build_strategy_performance()
    research_items = _build_research_details()
    signals = _build_signals()
    regime = _build_regime()
    queues = _build_queues()
    resources = _build_resources(now)

    # ── Header ──────────────────────────────────────────────────────────────
    tm = system.get("trading_mode", "PAPER" if system["paper_mode"] else "LIVE")
    mode_str = f"[yellow]{tm}[/yellow]" if system["paper_mode"] else f"[bold red]{tm}[/bold red]"
    ks_str = "[bold red]ACTIVE[/bold red]" if system["kill_active"] else "[green]inactive[/green]"

    # Regime in header
    regime_str = "[dim]unknown[/dim]"
    if regime:
        regime_val = regime.get("regime", regime.get("label", str(regime)))
        if isinstance(regime_val, dict):
            regime_val = regime_val.get("label", str(regime_val))
        regime_str = str(regime_val)

    header_text = (
        f"[bold]QUANTSTACK[/bold]  {now.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  |  Mode: {mode_str}"
        f"  |  Regime: {regime_str}"
        f"  |  Kill: {ks_str}"
    )
    parts.append(Panel(Text.from_markup(header_text), style="bold blue"))

    # ── Containers + Heartbeats (side by side) ──────────────────────────────
    ct_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    ct_table.add_column("Container", style="cyan", width=28)
    ct_table.add_column("Status", width=30)

    for c in system["containers"]:
        name = c.get("Name", c.get("name", "?"))
        status = c.get("Status", c.get("status", "?"))
        style = "green" if "healthy" in status.lower() else (
            "yellow" if "starting" in status.lower() else "red")
        ct_table.add_row(name, f"[{style}]{status}[/{style}]")

    if not system["containers"]:
        ct_table.add_row("[dim]Docker not running[/dim]", "")

    hb_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    hb_table.add_column("Graph", style="cyan", width=14)
    hb_table.add_column("Health", width=10)
    hb_table.add_column("Age", width=12)

    HB_STYLE = {"OK": "green", "STALE": "yellow", "NO DATA": "dim"}
    for hb in heartbeats:
        style = HB_STYLE.get(hb["health"], "red")
        age_str = _fmt_age(hb["age"])
        hb_table.add_row(
            hb["name"],
            f"[{style}]{hb['health']}[/{style}]",
            age_str,
        )

    parts.append(Columns([
        Panel(ct_table, title="[bold]Containers[/bold]", border_style="blue"),
        Panel(hb_table, title="[bold]Heartbeats[/bold]", border_style="blue", width=42),
    ]))

    # ── Loops (DB heartbeats) ──────────────────────────────────────────────
    if loops and any(lp["health"] != "NO DATA" for lp in loops):
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
            f"  |  Today: [bold]{_fmt_pnl(portfolio['daily_pnl'])}[/bold]"
            f" ({_fmt_pct(portfolio['daily_return_pct'])})"
            f"  |  Open: {portfolio['open_count']}"
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
                sym, (side or "").upper(), str(qty or ""),
                f"${avg:,.2f}" if avg else "n/a",
                f"${cur:,.2f}" if cur else "n/a",
                f"[{pnl_style}]{pnl_str}[/{pnl_style}]",
                (strat or "")[:28],
            )
    else:
        pos_table.add_row("[dim]No open positions[/dim]", "", "", "", "", "", "")

    parts.append(Panel(
        Columns([Text.from_markup(eq_line), pos_table]),
        title="[bold]Portfolio[/bold]", border_style="blue"
    ))

    # ── Active Signals ─────────────────────────────────────────────────────
    if signals:
        sig_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        sig_table.add_column("Symbol", width=8)
        sig_table.add_column("Action", width=8)
        sig_table.add_column("Conf", width=6, justify="right")
        sig_table.add_column("Size%", width=6, justify="right")
        sig_table.add_column("Stop", width=10, justify="right")
        sig_table.add_column("Target", width=10, justify="right")
        sig_table.add_column("Generated", width=19)
        sig_table.add_column("Expires", width=19)

        for sym, action, conf, size_pct, stop, target, gen_at, exp_at in signals:
            action_style = "green" if action in ("buy", "long") else "red"
            sig_table.add_row(
                sym,
                f"[{action_style}]{action.upper()}[/{action_style}]",
                f"{conf:.0%}" if conf else "n/a",
                f"{size_pct:.1%}" if size_pct else "n/a",
                f"${stop:,.2f}" if stop else "n/a",
                f"${target:,.2f}" if target else "n/a",
                str(gen_at)[:19] if gen_at else "",
                str(exp_at)[:19] if exp_at else "",
            )
        parts.append(Panel(sig_table, title="[bold]Active Signals[/bold]", border_style="blue"))

    # ── Strategy Pipeline + Recent Fills (side by side) ────────────────────
    strat_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    strat_table.add_column("Status", style="cyan")
    strat_table.add_column("Count", justify="right")
    for status in ["live", "forward_testing", "backtested", "validated", "draft", "failed"]:
        count = strat_counts.get(status, 0)
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
                sym, str(qty or ""),
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

    # ── Strategy Details ───────────────────────────────────────────────────
    if strat_details:
        sd_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        sd_table.add_column("Name", width=30)
        sd_table.add_column("Status", width=16)
        sd_table.add_column("Type", width=8)
        sd_table.add_column("Horizon", width=8)
        sd_table.add_column("Symbol", width=8)
        sd_table.add_column("Created", width=12)

        STATUS_STYLE = {
            "live": "bold green", "forward_testing": "green",
            "backtested": "yellow", "validated": "white",
            "draft": "dim", "failed": "red",
        }
        for name, st, itype, horizon, sym, created in strat_details:
            style = STATUS_STYLE.get(st, "white")
            sd_table.add_row(
                (name or "")[:30],
                f"[{style}]{st}[/{style}]",
                (itype or "")[:8], (horizon or "")[:8], (sym or "")[:8],
                str(created)[:10] if created else "",
            )
        parts.append(Panel(sd_table, title="[bold]Strategies[/bold]", border_style="blue"))

    # ── Closed Trades (realized P&L) ──────────────────────────────────────
    if closed_trades:
        trade_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        trade_table.add_column("Symbol", width=7)
        trade_table.add_column("Side", width=5)
        trade_table.add_column("Qty", width=6, justify="right")
        trade_table.add_column("Entry", width=9, justify="right")
        trade_table.add_column("Exit", width=9, justify="right")
        trade_table.add_column("P&L", width=11, justify="right")
        trade_table.add_column("Days", width=5, justify="right")
        trade_table.add_column("Strategy", width=22)
        trade_table.add_column("Reason", width=14)
        trade_table.add_column("Closed", width=12)

        total_realized = sum(t["pnl"] or 0 for t in closed_trades)
        winners = sum(1 for t in closed_trades if (t["pnl"] or 0) > 0)
        for t in closed_trades:
            pnl_style = "green" if (t["pnl"] or 0) >= 0 else "red"
            trade_table.add_row(
                t["symbol"], (t["side"] or "").upper(), str(t["qty"] or ""),
                f"${t['entry']:,.2f}" if t["entry"] else "n/a",
                f"${t['exit']:,.2f}" if t["exit"] else "n/a",
                f"[{pnl_style}]{_fmt_pnl(t['pnl'])}[/{pnl_style}]",
                str(t["days"]),
                t["strategy"][:22], t["reason"][:14],
                _fmt_age((now - t["closed_at"].replace(tzinfo=None)).total_seconds()
                         if t["closed_at"] else None),
            )
        ct_summary = (
            f"  Last {len(closed_trades)} trades: "
            f"[bold]{_fmt_pnl(total_realized)}[/bold] realized  |  "
            f"{winners}W / {len(closed_trades) - winners}L"
        )
        parts.append(Panel(
            Columns([Text.from_markup(ct_summary), trade_table]),
            title="[bold]Closed Trades[/bold]", border_style="blue"
        ))

    # ── Strategy Performance ───────────────────────────────────────────────
    if strat_perf:
        sp_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        sp_table.add_column("Strategy", width=28)
        sp_table.add_column("Trades", width=7, justify="right")
        sp_table.add_column("Win%", width=6, justify="right")
        sp_table.add_column("Total P&L", width=12, justify="right")
        sp_table.add_column("Avg P&L", width=10, justify="right")
        sp_table.add_column("Best", width=10, justify="right")
        sp_table.add_column("Worst", width=10, justify="right")
        sp_table.add_column("Avg Days", width=9, justify="right")

        for sp in strat_perf:
            pnl_style = "green" if (sp["total_pnl"] or 0) >= 0 else "red"
            wr_style = "green" if sp["win_rate"] >= 50 else "red"
            sp_table.add_row(
                sp["strategy"][:28], str(sp["trades"]),
                f"[{wr_style}]{sp['win_rate']:.0f}%[/{wr_style}]",
                f"[{pnl_style}]{_fmt_pnl(sp['total_pnl'])}[/{pnl_style}]",
                _fmt_pnl(sp["avg_pnl"]),
                f"[green]{_fmt_pnl(sp['best'])}[/green]",
                f"[red]{_fmt_pnl(sp['worst'])}[/red]",
                f"{sp['avg_days']:.1f}",
            )
        parts.append(Panel(sp_table, title="[bold]Strategy Performance[/bold]",
                            border_style="blue"))

    # ── Research Queue ─────────────────────────────────────────────────────
    rq_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    rq_table.add_column("Type", width=18)
    rq_table.add_column("P", width=3, justify="right")
    rq_table.add_column("Topic", width=40)
    rq_table.add_column("Status", width=9)
    rq_table.add_column("Source", width=16)
    rq_table.add_column("Age", width=10)

    RQ_STATUS_STYLE = {"running": "bold yellow", "pending": "white",
                       "done": "green", "failed": "red"}
    if research_items:
        for ri in research_items:
            st_style = RQ_STATUS_STYLE.get(ri["status"], "dim")
            rq_table.add_row(
                ri["type"], str(ri["priority"]), ri["topic"],
                f"[{st_style}]{ri['status']}[/{st_style}]",
                ri["source"][:16],
                _fmt_age((now - ri["created_at"].replace(tzinfo=None)).total_seconds()
                         if ri["created_at"] else None),
            )
    else:
        rq_table.add_row("[dim]Research queue empty[/dim]", "", "", "", "", "")

    parts.append(Panel(rq_table, title="[bold]Research Queue[/bold]",
                        border_style="blue"))

    # ── Bugs + Queue Summary ───────────────────────────────────────────────
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
    queue_text = f"  Bugs: {bugs_str}\n  Research: {rq_str}"
    parts.append(Panel(Text.from_markup(queue_text), title="[bold]Queues[/bold]", border_style="blue"))

    # ── Resources ────────────────────────────────────────────────────────────
    av_pct = int(system["av_calls"] / system["av_limit"] * 100) if system["av_limit"] else 0
    av_style = "red" if av_pct > 80 else ("yellow" if av_pct > 50 else "green")
    res_lines = [
        f"  AV quota: [{av_style}]{system['av_calls']}/{system['av_limit']} ({av_pct}%)[/{av_style}]",
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
        res_lines.append(f"  Community intel: [dim]{resources['last_community_intel']}[/dim]")

    parts.append(Panel(
        Text.from_markup("\n".join(res_lines)),
        title="[bold]Resources[/bold]", border_style="blue"
    ))

    parts.append(Text.from_markup(
        "[dim]  ./stop.sh to stop  |  ./start.sh to start  |  --watch for live refresh[/dim]"
    ))

    return Group(*parts)


def _render(now: datetime):
    from rich.console import Console
    Console().print(_build_dashboard(now))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QuantStack health dashboard")
    parser.add_argument("--watch", action="store_true",
                        help="Refresh every 10 seconds (Ctrl+C to quit)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Refresh interval in seconds for --watch mode (default: 10)")
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
