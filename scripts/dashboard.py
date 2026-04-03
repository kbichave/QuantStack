#!/usr/bin/env python3
# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0
"""
QuantStack terminal dashboard — full system visibility in one screen.

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
        url = "postgresql://localhost/quantstack"
    url = url.replace("@postgres:5432", "@127.0.0.1:5434")
    url = url.replace("@host.docker.internal:", "@localhost:")
    return psycopg2.connect(url)


def _query(sql: str, params=()) -> list:
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
# Data builders
# ---------------------------------------------------------------------------

def _docker_containers() -> list[dict]:
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
        return containers
    except Exception:
        return []


def _build_header(now: datetime) -> dict:
    sentinel = Path(
        os.environ.get("KILL_SWITCH_SENTINEL", "~/.quantstack/KILL_SWITCH_ACTIVE")
    ).expanduser()
    kill_file = sentinel.exists()
    ks_val = _scalar(
        "SELECT value FROM system_state WHERE key = 'kill_switch'", default="inactive"
    )
    kill_db = (ks_val or "").lower() == "active"

    use_real = os.environ.get("USE_REAL_TRADING", "false").lower() == "true"
    alpaca_paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    if not use_real:
        mode = "PAPER"
    elif alpaca_paper:
        mode = "ALPACA PAPER"
    else:
        mode = "LIVE"

    today_key = f"av_daily_calls_{now.strftime('%Y-%m-%d')}"
    av_calls = _scalar("SELECT value FROM system_state WHERE key = %s", (today_key,), default="0")
    av_limit = int(os.environ.get("AV_DAILY_CALL_LIMIT", "25000"))
    try:
        av_used = int(av_calls or 0)
    except ValueError:
        av_used = 0

    # Regime from regime_states table
    regime_row = _query("""
        SELECT symbol, trend_regime, volatility_regime, confidence
        FROM regime_states WHERE symbol = 'SPY'
        ORDER BY timestamp DESC LIMIT 1
    """)
    if regime_row:
        r = regime_row[0]
        regime = f"{r[1]}/{r[2]}" if r[2] else str(r[1])
        regime_conf = r[3]
    else:
        regime_val = _scalar("""
            SELECT context_json FROM loop_iteration_context
            WHERE context_key = 'current_regime'
            ORDER BY updated_at DESC LIMIT 1
        """)
        regime = "unknown"
        regime_conf = None
        if regime_val is not None:
            if isinstance(regime_val, str):
                try:
                    regime_val = json.loads(regime_val)
                except json.JSONDecodeError:
                    regime = regime_val
            if isinstance(regime_val, dict):
                regime = regime_val.get("regime", regime_val.get("label", str(regime_val)))

    universe_count = _scalar(
        "SELECT COUNT(*) FROM universe WHERE is_active = TRUE", default=0
    )
    data_symbols = _scalar(
        "SELECT COUNT(DISTINCT symbol) FROM ohlcv WHERE timeframe = '1D'", default=0
    )

    return {
        "kill_active": kill_file or kill_db,
        "mode": mode,
        "av_used": av_used,
        "av_limit": av_limit,
        "regime": regime,
        "regime_conf": regime_conf,
        "universe": universe_count or 0,
        "data_symbols": data_symbols or 0,
    }


def _build_services() -> list[dict]:
    GRAPH_MAP = {
        "quantstack-trading-graph": "trading",
        "quantstack-research-graph": "research",
        "quantstack-supervisor-graph": "supervisor",
    }
    cycles = {}
    for r in _query("""
        SELECT DISTINCT ON (graph_name)
               graph_name, cycle_number, duration_seconds, status, created_at
        FROM graph_checkpoints ORDER BY graph_name, created_at DESC
    """):
        cycles[r[0]] = {"cycle": r[1], "dur": r[2], "status": r[3], "at": r[4]}

    heartbeats = {}
    for r in _query("""
        SELECT DISTINCT ON (loop_name)
               loop_name, started_at, status
        FROM loop_heartbeats ORDER BY loop_name, started_at DESC
    """):
        heartbeats[r[0]] = {"at": r[1], "status": r[2]}

    containers = _docker_containers()
    services = []
    for c in containers:
        name = c.get("Name", c.get("name", "?"))
        status = c.get("Status", c.get("status", "?"))
        healthy = "healthy" in status.lower() or "running" in status.lower()
        graph = GRAPH_MAP.get(name)
        cy = cycles.get(graph, {}) if graph else {}
        # Strip docker compose prefix hashes and quantstack- prefix
        display_name = name.replace("quantstack-", "")
        # Handle hash-prefixed names like "1c289f505728_supervisor-graph"
        if "_" in display_name and display_name.split("_")[0].isalnum() and len(display_name.split("_")[0]) == 12:
            display_name = display_name.split("_", 1)[1]
        svc = {
            "name": display_name,
            "up": healthy,
            "status": "UP" if healthy else ("START" if "starting" in status.lower() else "DOWN"),
        }
        if cy:
            svc["cycle"] = cy["cycle"]
            svc["dur"] = cy["dur"]
            svc["cy_status"] = cy["status"]
        services.append(svc)
    return services


def _build_portfolio() -> dict:
    positions = _query("""
        SELECT symbol, quantity, avg_cost, current_price, unrealized_pnl,
               side, strategy_id
        FROM positions ORDER BY unrealized_pnl DESC NULLS LAST
    """)
    equity_row = _query("""
        SELECT total_equity, daily_pnl, daily_return_pct, cash
        FROM daily_equity ORDER BY date DESC LIMIT 1
    """)
    eq = equity_row[0] if equity_row else None
    if eq is None:
        cash = _scalar("SELECT cash FROM cash_balance WHERE id = 1", default=None)
        return {"positions": positions, "equity": float(cash) if cash else None,
                "pnl": 0.0, "pnl_pct": 0.0, "cash": float(cash) if cash else None}
    return {
        "positions": positions,
        "equity": eq[0], "pnl": eq[1], "pnl_pct": eq[2], "cash": eq[3],
    }


def _build_closed_trades() -> list[dict]:
    rows = _query("""
        SELECT symbol, side, realized_pnl, holding_days, strategy_id,
               exit_reason, closed_at
        FROM closed_trades ORDER BY closed_at DESC NULLS LAST LIMIT 5
    """)
    return [{"symbol": r[0], "side": r[1], "pnl": r[2], "days": r[3],
             "strategy": (r[4] or "")[:20], "reason": r[5] or "",
             "at": r[6]} for r in rows]


def _build_strategies() -> dict:
    rows = _query("""
        SELECT name, status, symbol, instrument_type, time_horizon,
               (backtest_summary::jsonb ->> 'sharpe')::float
        FROM strategies ORDER BY
            CASE status WHEN 'live' THEN 0 WHEN 'forward_testing' THEN 1
                 WHEN 'backtested' THEN 2 WHEN 'validated' THEN 3 ELSE 4 END,
            updated_at DESC
        LIMIT 10
    """)
    # Status counts
    counts = {}
    for r in _query("SELECT status, COUNT(*) FROM strategies GROUP BY 1"):
        counts[r[0]] = r[1]
    # Type breakdown
    types = {}
    for r in _query("SELECT instrument_type, COUNT(*) FROM strategies GROUP BY 1 ORDER BY 2 DESC LIMIT 5"):
        types[r[0] or "unknown"] = r[1]

    return {
        "list": [{"name": r[0] or "", "status": r[1] or "", "symbol": r[2] or "",
                  "type": r[3] or "", "horizon": r[4] or "", "sharpe": r[5]}
                 for r in rows],
        "counts": counts,
        "types": types,
    }


def _build_research() -> dict:
    queue = _query("""
        SELECT task_type, status, topic, priority
        FROM research_queue
        WHERE status IN ('pending', 'running')
        ORDER BY
            CASE status WHEN 'running' THEN 0 ELSE 1 END,
            priority DESC LIMIT 5
    """)
    queue_total = _scalar("SELECT COUNT(*) FROM research_queue WHERE status = 'pending'", default=0)
    wip = _query("""
        SELECT symbol, domain, agent_id FROM research_wip
        ORDER BY heartbeat_at DESC LIMIT 4
    """)
    ml_count = _scalar("SELECT COUNT(*) FROM ml_experiments", default=0)
    ml_best = _scalar(
        "SELECT MAX(test_auc) FROM ml_experiments WHERE verdict != 'failed'",
        default=None,
    )
    alpha_active = _scalar(
        "SELECT COUNT(*) FROM alpha_research_program WHERE status = 'active'",
        default=0,
    )
    bugs_open = _scalar(
        "SELECT COUNT(*) FROM bugs WHERE status IN ('open', 'in_progress')",
        default=0,
    )
    return {
        "queue": [{"type": r[0], "status": r[1], "topic": r[2] or "", "priority": r[3]}
                  for r in queue],
        "queue_total": queue_total or 0,
        "wip": [{"symbol": r[0], "domain": r[1], "agent": r[2]} for r in wip],
        "ml_count": ml_count or 0,
        "ml_best_auc": ml_best,
        "alpha_active": alpha_active or 0,
        "bugs_open": bugs_open or 0,
    }


def _build_data_health() -> dict:
    """Data freshness and coverage summary."""
    fresh = _scalar("""
        SELECT COUNT(DISTINCT symbol) FROM ohlcv
        WHERE timeframe = '1D' AND timestamp >= CURRENT_DATE - INTERVAL '2 days'
    """, default=0)
    total = _scalar("SELECT COUNT(DISTINCT symbol) FROM ohlcv WHERE timeframe = '1D'", default=0)
    stale_symbols = _query("""
        WITH latest AS (
            SELECT symbol, MAX(timestamp) as latest
            FROM ohlcv WHERE timeframe = '1D' GROUP BY symbol
        )
        SELECT symbol, latest FROM latest
        WHERE latest < CURRENT_DATE - INTERVAL '3 days'
        ORDER BY latest LIMIT 3
    """)
    # Upcoming earnings for universe symbols
    earnings = _query("""
        SELECT e.symbol, e.report_date
        FROM earnings_calendar e
        JOIN universe u ON e.symbol = u.symbol AND u.is_active = TRUE
        WHERE e.report_date >= CURRENT_DATE AND e.report_date <= CURRENT_DATE + 14
        ORDER BY e.report_date LIMIT 5
    """)
    return {
        "fresh": fresh or 0,
        "total": total or 0,
        "stale": [{"symbol": r[0], "latest": r[1]} for r in stale_symbols],
        "upcoming_earnings": [{"symbol": r[0], "date": r[1]} for r in earnings],
    }


def _build_signals() -> list[dict]:
    rows = _query("""
        SELECT symbol, action, confidence, position_size_pct, generated_at
        FROM signal_state ORDER BY confidence DESC NULLS LAST LIMIT 5
    """)
    return [{"symbol": r[0], "action": r[1], "conf": r[2], "size": r[3], "at": r[4]}
            for r in rows]


def _build_reflections() -> list[dict]:
    rows = _query("""
        SELECT symbol, realized_pnl_pct, lesson, created_at
        FROM trade_reflections ORDER BY created_at DESC LIMIT 3
    """)
    return [{"symbol": r[0], "pnl_pct": r[1], "lesson": (r[2] or "")[:60],
             "at": r[3]} for r in rows]


def _build_decisions() -> list[dict]:
    """Recent decision events — what the system decided and why."""
    rows = _query("""
        SELECT event_type, symbol, action, agent_name, confidence,
               LEFT(output_summary, 80), created_at
        FROM decision_events ORDER BY created_at DESC LIMIT 5
    """)
    return [{"type": r[0] or "", "symbol": r[1] or "", "action": r[2] or "",
             "agent": r[3] or "", "conf": r[4],
             "summary": (r[5] or "")[:60], "at": r[6]} for r in rows]


_AGENT_TEAMS = {
    "quant_researcher": "research", "ml_scientist": "research",
    "strategy_rd": "research", "community_intel": "research",
    "equity_investment_researcher": "research", "equity_swing_researcher": "research",
    "options_researcher": "research", "execution_researcher": "research",
    "position_monitor": "trading", "entry_scanner": "trading",
    "exit_manager": "trading", "risk_assessor": "trading",
    "daily_planner": "trading", "fund_manager": "trading",
    "options_analyst": "trading", "trade_debater": "trading",
    "reflector": "trading", "execution_manager": "trading",
    "risk_analyst": "trading", "market_intel": "trading",
    "earnings_analyst": "trading", "executor": "trading",
    "trade_reflector": "trading",
    "health_monitor": "supervisor", "diagnostician": "supervisor",
    "self_healer": "supervisor", "strategy_promoter": "supervisor",
    "scheduler": "supervisor",
}


def _build_agent_events() -> dict[str, list[dict]]:
    now = datetime.utcnow()
    rows = _query("""
        SELECT graph_name, node_name, agent_name, event_type, content, created_at
        FROM agent_events ORDER BY id DESC LIMIT 60
    """)
    by_team: dict[str, list[dict]] = {}
    for graph, node, agent, etype, content, created_at in reversed(rows):
        age_s = None
        if created_at:
            ca = created_at.replace(tzinfo=None) if created_at.tzinfo else created_at
            age_s = (now - ca).total_seconds()
        team = graph or "unknown"
        agent_name = agent or node or "?"
        if team == "unknown":
            team = _AGENT_TEAMS.get(agent_name, "other")
        by_team.setdefault(team, []).append({
            "agent": agent_name, "type": etype or "",
            "content": (content or "")[:80], "age_s": age_s,
        })
    return by_team


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _age(seconds) -> str:
    if seconds is None:
        return "  -  "
    s = int(seconds)
    if s < 60:
        return f"{s:>3}s "
    if s < 3600:
        return f"{s // 60:>3}m "
    return f"{s // 3600}h{(s % 3600) // 60:02}m"


def _pnl(val) -> str:
    if val is None:
        return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.2f}"


def _pnl_style(val) -> str:
    if val is None:
        return "dim"
    return "green" if val >= 0 else "red"


def _sparkbar(used: int, total: int, width: int = 10) -> str:
    """Inline usage bar like [####------]."""
    if total == 0:
        return ""
    filled = min(int(used / total * width), width)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------

def _build_dashboard(now: datetime):
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.layout import Layout
    from rich import box

    parts = []
    hdr = _build_header(now)

    # ── Header bar ────────────────────────────────────────────────────
    mode = hdr["mode"]
    mode_s = f"[yellow]{mode}[/yellow]" if "PAPER" in mode else f"[bold red]{mode}[/bold red]"
    ks = "[bold red]KILL[/bold red]" if hdr["kill_active"] else "[green]ok[/green]"
    av_pct = int(hdr["av_used"] / hdr["av_limit"] * 100) if hdr["av_limit"] else 0
    av_c = "red" if av_pct > 80 else ("yellow" if av_pct > 50 else "dim")
    regime_s = hdr["regime"]
    if hdr.get("regime_conf") is not None:
        regime_s += f" ({hdr['regime_conf']:.0%})"
    parts.append(Text.from_markup(
        f"[bold blue]QUANTSTACK[/bold blue]  {now.strftime('%H:%M:%S')}"
        f"  |  {mode_s}"
        f"  |  Regime: [bold]{regime_s}[/bold]"
        f"  |  Kill: {ks}"
        f"  |  AV: [{av_c}]{hdr['av_used']}/{hdr['av_limit']}[/{av_c}]"
        f"  |  Universe: {hdr['universe']}"
        f"  |  Data: {hdr['data_symbols']} syms"
    ))

    # ── Services bar ──────────────────────────────────────────────────
    svcs = _build_services()
    svc_parts = []
    for s in svcs:
        h = "[green]UP[/green]" if s["up"] else "[red]DOWN[/red]"
        name = s["name"][:16]
        if "cycle" in s:
            cy_s = f"c#{s['cycle']}"
            dur = f"{s['dur']:.0f}s" if s.get("dur") else ""
            svc_parts.append(f"[cyan]{name}[/cyan] {h} {cy_s} {dur}")
        else:
            svc_parts.append(f"[cyan]{name}[/cyan] {h}")
    parts.append(Panel(
        Text.from_markup("  ".join(svc_parts)),
        title="[bold]Services[/bold]", border_style="blue", padding=(0, 1),
    ))

    # ── Row 1: Portfolio + Recent Trades ──────────────────────────────
    port = _build_portfolio()
    trades = _build_closed_trades()

    # Portfolio
    pl = []
    if port["equity"] is not None:
        ps = _pnl_style(port["pnl"])
        exposure = sum(abs((p[1] or 0) * (p[3] or p[2] or 0)) for p in port["positions"])
        cash_pct = (port["cash"] / port["equity"] * 100) if port["equity"] else 0
        pl.append(
            f"Equity [bold]${port['equity']:,.0f}[/bold]"
            f"  Cash ${port['cash']:,.0f} ({cash_pct:.0f}%)"
            f"  Today [{ps}]{_pnl(port['pnl'])}[/{ps}]"
        )
        if exposure > 0:
            exp_pct = exposure / port["equity"] * 100 if port["equity"] else 0
            pl.append(f"  Exposure: ${exposure:,.0f} ({exp_pct:.0f}%)")
    else:
        pl.append("[dim]No equity data[/dim]")
    if port["positions"]:
        for sym, qty, avg, cur, upnl, side, strat in port["positions"][:5]:
            ps = _pnl_style(upnl)
            strat_short = (strat or "")[:12]
            ret_pct = ((cur - avg) / avg * 100) if avg and cur else 0
            pl.append(
                f"[bold]{sym:6s}[/bold] {qty:>4} @ ${avg:>7.2f} -> ${cur:>7.2f}"
                f"  [{ps}]{_pnl(upnl)} ({ret_pct:+.1f}%)[/{ps}]"
                f"  [dim]{strat_short}[/dim]"
            )
    else:
        pl.append("[dim]No open positions[/dim]")
    port_panel = Panel(
        Text.from_markup("\n".join(pl)),
        title="[bold]Portfolio[/bold]", border_style="green",
    )

    # Recent closed trades
    tl = []
    if trades:
        for t in trades:
            ps = _pnl_style(t["pnl"])
            days = f"{t['days']}d" if t["days"] else "  "
            reason = t["reason"][:10] if t["reason"] else ""
            tl.append(
                f"[bold]{t['symbol']:6s}[/bold]"
                f" [{ps}]{_pnl(t['pnl']):>10s}[/{ps}]"
                f"  {days:>4}"
                f"  [dim]{t['strategy'][:16]}[/dim]"
                f"  {reason}"
            )
    else:
        tl.append("[dim]No closed trades yet[/dim]")
    trade_panel = Panel(
        Text.from_markup("\n".join(tl)),
        title="[bold]Recent Trades[/bold]", border_style="green",
    )
    parts.append(Columns([port_panel, trade_panel], equal=True))

    # ── Row 2: Strategies + Research ──────────────────────────────────
    strat_data = _build_strategies()
    research = _build_research()
    signals = _build_signals()

    STATUS_STYLE = {
        "live": "bold green", "forward_testing": "green",
        "backtested": "yellow", "validated": "white",
        "draft": "dim", "failed": "red", "retired": "dim",
    }

    # Strategies panel — summary line + list
    sl = []
    counts = strat_data["counts"]
    count_parts = []
    for st in ("live", "forward_testing", "draft", "retired"):
        n = counts.get(st, 0)
        if n > 0:
            c = STATUS_STYLE.get(st, "dim")
            label = st.replace("_", " ")[:8]
            count_parts.append(f"[{c}]{label}: {n}[/{c}]")
    if count_parts:
        sl.append("  ".join(count_parts))

    strats = strat_data["list"]
    if strats:
        for s in strats[:8]:
            st = STATUS_STYLE.get(s["status"], "dim")
            sharpe = f"SR {s['sharpe']:.2f}" if s["sharpe"] else ""
            horizon = s["horizon"][:5] if s["horizon"] else ""
            sl.append(
                f"[{st}]{s['status'][:10]:10s}[/{st}]"
                f" {s['symbol']:6s}"
                f" {s['name'][:20]:20s}"
                f" [dim]{s['type'][:6]:6s} {horizon:5s} {sharpe}[/dim]"
            )
    else:
        sl.append("[dim]No strategies[/dim]")

    # Signals inline at bottom
    if signals:
        sig_parts = []
        for s in signals[:4]:
            action = (s['action'] or '').upper()[:4]
            ac = "green" if action in ("BUY", "LONG") else "red"
            conf = f" {s['conf']:.0%}" if s['conf'] else ""
            sig_parts.append(f"[{ac}]{action}[/{ac}] {s['symbol']}{conf}")
        sl.append(f"[bold]Signals:[/bold] {'  '.join(sig_parts)}")

    strat_panel = Panel(
        Text.from_markup("\n".join(sl)),
        title="[bold]Strategies[/bold]", border_style="yellow",
    )

    # Research panel
    rl = []
    if research["wip"]:
        wip_str = ", ".join(f"{w['symbol']}/{w['domain']}" for w in research["wip"][:4])
        rl.append(f"[bold]WIP:[/bold] {wip_str}")
    for q in research["queue"][:3]:
        qs = "[green]RUN[/green]" if q["status"] == "running" else "[dim]PND[/dim]"
        rl.append(f"{qs} P{q['priority']} {q['type'][:16]}  [dim]{q['topic'][:25]}[/dim]")
    if research["queue_total"] > 3:
        rl.append(f"[dim]  +{research['queue_total'] - 3} more in queue[/dim]")
    if not research["queue"] and not research["wip"]:
        rl.append("[dim]Queue empty[/dim]")

    # ML + Alpha + Bugs summary line
    ml_str = f"ML: {research['ml_count']} exps"
    if research["ml_best_auc"]:
        ml_str += f" (best {research['ml_best_auc']:.3f})"
    alpha_str = f"Alpha: {research['alpha_active']}"
    bugs_c = "red" if research["bugs_open"] > 0 else "green"
    bugs_str = f"Bugs: [{bugs_c}]{research['bugs_open']}[/{bugs_c}]"
    rl.append(f"[dim]{ml_str} | {alpha_str} | {bugs_str}[/dim]")

    research_panel = Panel(
        Text.from_markup("\n".join(rl)),
        title="[bold]Research[/bold]", border_style="cyan",
    )
    parts.append(Columns([strat_panel, research_panel], equal=True))

    # ── Row 3: Data Health + Decisions / Lessons ──────────────────────
    data_health = _build_data_health()
    decisions = _build_decisions()
    reflections = _build_reflections()

    # Data health panel
    dh = []
    fresh = data_health["fresh"]
    total = data_health["total"]
    stale_count = total - fresh
    fc = "green" if stale_count == 0 else ("yellow" if stale_count < 5 else "red")
    dh.append(f"OHLCV: [{fc}]{fresh}/{total} fresh[/{fc}]  {_sparkbar(fresh, total)}")
    if data_health["stale"]:
        stale_syms = ", ".join(s["symbol"] for s in data_health["stale"])
        dh.append(f"[yellow]Stale: {stale_syms}[/yellow]")
    if data_health["upcoming_earnings"]:
        earn_parts = []
        for e in data_health["upcoming_earnings"][:4]:
            d = e["date"]
            ds = d.strftime("%m/%d") if hasattr(d, "strftime") else str(d)[:5]
            earn_parts.append(f"{e['symbol']} {ds}")
        dh.append(f"[bold]Earnings:[/bold] {', '.join(earn_parts)}")
    else:
        dh.append("[dim]No upcoming earnings in universe[/dim]")

    data_panel = Panel(
        Text.from_markup("\n".join(dh)),
        title="[bold]Data[/bold]", border_style="blue",
    )

    # Decision / Lessons panel
    dl = []
    if decisions:
        for d in decisions[:3]:
            sym = f"[bold]{d['symbol']:5s}[/bold]" if d["symbol"] else "[dim]     [/dim]"
            action = d["action"][:10] if d["action"] else d["type"][:10]
            ac = "green" if "buy" in action.lower() or "enter" in action.lower() else (
                "red" if "sell" in action.lower() or "exit" in action.lower() else "white")
            conf = f" {d['conf']:.0%}" if d["conf"] else ""
            dl.append(f"{sym} [{ac}]{action:10s}[/{ac}]{conf} [dim]{d['summary'][:35]}[/dim]")
    if reflections:
        for r in reflections[:2]:
            ps = _pnl_style(r["pnl_pct"])
            pct = f"{r['pnl_pct']:+.1f}%" if r["pnl_pct"] is not None else ""
            dl.append(
                f"[bold]{r['symbol']:5s}[/bold]"
                f" [{ps}]{pct:>6s}[/{ps}]"
                f" [dim]{r['lesson']}[/dim]"
            )
    if not dl:
        dl.append("[dim]No recent decisions or lessons[/dim]")

    decision_panel = Panel(
        Text.from_markup("\n".join(dl)),
        title="[bold]Decisions & Lessons[/bold]", border_style="magenta",
    )
    parts.append(Columns([data_panel, decision_panel], equal=True))

    # ── Row 4: Agent Activity ─────────────────────────────────────────
    all_events = _build_agent_events()
    TEAM_CFG = {
        "research":   {"style": "cyan",    "title": "Research Agents"},
        "trading":    {"style": "green",   "title": "Trading Agents"},
        "supervisor": {"style": "magenta", "title": "Supervisor Agents"},
    }
    TYPE_ICONS = {
        "agent_start": ">",
        "agent_response": "<",
        "tool_call": "*",
        "node_complete": "#",
    }

    def _chat(team: str, events: list[dict], max_lines: int = 8) -> Panel:
        cfg = TEAM_CFG.get(team, {"style": "yellow", "title": team.title()})
        lines = []
        for ev in events[-max_lines:]:
            icon = TYPE_ICONS.get(ev["type"], " ")
            age = _age(ev["age_s"])
            agent = ev["agent"][:14]
            content = ev["content"][:45]
            if ev["type"] == "tool_call" and "(" in content:
                tool_name = content.split("(")[0]
                content = f"[yellow]{tool_name}[/yellow][dim](...)[/dim]"
            elif ev["type"] == "agent_response":
                content = f"[green]{content}[/green]"
            elif ev["type"] == "agent_start":
                content = f"[{cfg['style']}]{content}[/{cfg['style']}]"
            lines.append(f"[dim]{age}[/dim]{icon} [bold]{agent:14s}[/bold] {content}")
        if not lines:
            lines = ["[dim]No activity[/dim]"]
        return Panel(
            Text.from_markup("\n".join(lines)),
            title=f"[bold]{cfg['title']}[/bold]", border_style=cfg["style"],
        )

    has_events = any(all_events.get(t) for t in ("research", "trading", "supervisor"))
    if has_events:
        agent_panels = []
        for team in ("research", "trading"):
            events = all_events.get(team, [])
            agent_panels.append(_chat(team, events))
        parts.append(Columns(agent_panels, equal=True))

        sup = all_events.get("supervisor", [])
        if sup:
            parts.append(_chat("supervisor", sup, max_lines=5))

    # ── Footer ────────────────────────────────────────────────────────
    parts.append(Text.from_markup(
        "[dim]./stop.sh  |  ./start.sh  |  --watch  |  http://localhost:8421[/dim]"
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
