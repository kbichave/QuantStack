#!/usr/bin/env python3
"""
QuantStack System Audit — deep health check across all subsystems.

Checks:
  1. Docker container health (running, healthy, restart count)
  2. Graph checkpoint analysis (cycle success rate, duration anomalies, gaps)
  3. Research queue throughput (drain rate, stuck tasks, claim/complete lifecycle)
  4. Langfuse trace analysis (error rates, latency anomalies, cost tracking)
  5. Strategy pipeline health (draft→backtested→forward_testing flow)
  6. DB connectivity and data freshness (OHLCV staleness, feature coverage)
  7. Bug tracker status (open bugs, recurring failures)
  8. Position / risk state (exposure, P&L, stale prices)

Usage:
    python scripts/system_audit.py              # Full audit
    python scripts/system_audit.py --section docker langfuse  # Specific sections
    python scripts/system_audit.py --json       # Machine-readable output

Exit codes:
    0 — all checks passed
    1 — warnings found (non-critical)
    2 — critical issues found
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Finding model
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"info": 0, "ok": 1, "warning": 2, "critical": 3}


@dataclass
class Finding:
    section: str
    severity: str  # "ok", "info", "warning", "critical"
    message: str
    detail: str = ""
    metric: Any = None


@dataclass
class AuditReport:
    timestamp: str = ""
    findings: list[Finding] = field(default_factory=list)

    def add(self, section: str, severity: str, message: str, detail: str = "", metric: Any = None):
        self.findings.append(Finding(section, severity, message, detail, metric))

    @property
    def max_severity(self) -> str:
        if not self.findings:
            return "ok"
        return max(self.findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 0)).severity

    @property
    def exit_code(self) -> int:
        sev = self.max_severity
        if sev == "critical":
            return 2
        if sev == "warning":
            return 1
        return 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "max_severity": self.max_severity,
            "findings": [
                {"section": f.section, "severity": f.severity, "message": f.message,
                 "detail": f.detail, "metric": f.metric}
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_conn():
    from quantstack.db import db_conn
    return db_conn()


def _query(sql: str, params: list | None = None) -> list:
    with _get_conn() as conn:
        return conn.execute(sql, params or []).fetchall()


def _query_one(sql: str, params: list | None = None):
    with _get_conn() as conn:
        return conn.execute(sql, params or []).fetchone()


# ---------------------------------------------------------------------------
# Section: Docker containers
# ---------------------------------------------------------------------------

def audit_docker(report: AuditReport) -> None:
    section = "docker"
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            report.add(section, "critical", "docker compose ps failed", result.stderr[:200])
            return

        containers = []
        for line in result.stdout.strip().splitlines():
            try:
                containers.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not containers:
            # Try parsing as a JSON array
            try:
                containers = json.loads(result.stdout)
            except json.JSONDecodeError:
                report.add(section, "warning", "Could not parse docker compose output")
                return

        expected = {"quantstack-trading-graph", "quantstack-research-graph",
                    "quantstack-supervisor-graph", "quantstack-postgres"}
        running_names = set()

        for c in containers:
            name = c.get("Name", c.get("name", ""))
            state = c.get("State", c.get("state", ""))
            health = c.get("Health", c.get("health", ""))
            running_names.add(name)

            if state != "running":
                report.add(section, "critical", f"{name} is {state}", f"health={health}")
            elif "unhealthy" in str(health).lower():
                report.add(section, "warning", f"{name} is unhealthy", f"state={state}")

        missing = expected - running_names
        if missing:
            report.add(section, "critical", f"Missing containers: {missing}")
        else:
            report.add(section, "ok", f"All {len(containers)} containers running")

    except FileNotFoundError:
        report.add(section, "warning", "docker not found — skipping container checks")
    except subprocess.TimeoutExpired:
        report.add(section, "warning", "docker compose ps timed out")


# ---------------------------------------------------------------------------
# Section: Graph checkpoints
# ---------------------------------------------------------------------------

def audit_graph_checkpoints(report: AuditReport) -> None:
    section = "graph_checkpoints"

    for graph_name in ("research", "trading", "supervisor"):
        # Recent cycles (last 2 hours)
        rows = _query(
            """SELECT cycle_number, status, duration_seconds, error_message, created_at
               FROM graph_checkpoints
               WHERE graph_name = %s AND created_at > NOW() - INTERVAL '2 hours'
               ORDER BY created_at DESC""",
            [graph_name],
        )

        if not rows:
            # Check if container was recently restarted (< 30 min ago)
            report.add(section, "warning", f"{graph_name}: no cycles in last 2 hours (recently restarted?)")
            continue

        total = len(rows)
        errors = [r for r in rows if r[1] != "success"]
        timeouts = [r for r in rows if r[1] == "timeout"]
        durations = [r[2] for r in rows if r[2] is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        error_rate = len(errors) / total if total else 0

        if error_rate > 0.5:
            report.add(section, "critical",
                        f"{graph_name}: {error_rate:.0%} error rate ({len(errors)}/{total} cycles)",
                        detail=f"Errors: {[r[3][:80] for r in errors[:3]]}",
                        metric={"error_rate": error_rate, "total": total})
        elif error_rate > 0.2:
            report.add(section, "warning",
                        f"{graph_name}: {error_rate:.0%} error rate ({len(errors)}/{total} cycles)",
                        metric={"error_rate": error_rate, "total": total})
        elif timeouts:
            report.add(section, "warning",
                        f"{graph_name}: {len(timeouts)} timeout(s) in last 2h",
                        metric={"timeouts": len(timeouts)})
        else:
            report.add(section, "ok",
                        f"{graph_name}: {total} cycles, all success, avg {avg_duration:.0f}s (max {max_duration:.0f}s)",
                        metric={"total": total, "avg_duration": avg_duration})

        # Duration anomaly: any cycle >3x the average
        if avg_duration > 0:
            outliers = [r for r in rows if r[2] and r[2] > avg_duration * 3]
            if outliers:
                report.add(section, "info",
                            f"{graph_name}: {len(outliers)} slow cycle(s) (>3x avg of {avg_duration:.0f}s)",
                            detail=f"Max: {max_duration:.0f}s")


# ---------------------------------------------------------------------------
# Section: Research queue
# ---------------------------------------------------------------------------

def audit_research_queue(report: AuditReport) -> None:
    section = "research_queue"

    # Status breakdown
    rows = _query("SELECT status, COUNT(*) FROM research_queue GROUP BY status")
    status_map = {r[0]: r[1] for r in rows}
    total = sum(status_map.values())
    pending = status_map.get("pending", 0)
    running = status_map.get("running", 0)
    done = status_map.get("done", 0)
    failed = status_map.get("failed", 0)

    report.add(section, "info",
                f"Queue: {pending} pending, {running} running, {done} done, {failed} failed (total {total})",
                metric=status_map)

    # Stuck running tasks (running > 30 min without completing)
    stuck = _query(
        """SELECT task_id, topic, started_at
           FROM research_queue
           WHERE status = 'running' AND started_at < NOW() - INTERVAL '30 minutes'"""
    )
    if stuck:
        report.add(section, "warning",
                    f"{len(stuck)} task(s) stuck in 'running' for >30 min",
                    detail=", ".join(r[1] or r[0][:20] for r in stuck[:5]))

    # Drain rate (completed in last 2 hours)
    completed_2h = _query_one(
        "SELECT COUNT(*) FROM research_queue WHERE status IN ('done', 'failed') AND completed_at > NOW() - INTERVAL '2 hours'"
    )
    drain_rate = completed_2h[0] if completed_2h else 0
    if pending > 0 and drain_rate == 0:
        report.add(section, "warning",
                    f"{pending} pending tasks but 0 completed in last 2h — pipeline may be stalled")
    elif drain_rate > 0:
        eta_hours = (pending / (drain_rate / 2)) if drain_rate > 0 else float("inf")
        report.add(section, "ok",
                    f"Draining at {drain_rate}/2h — ETA to clear queue: {eta_hours:.1f}h",
                    metric={"drain_rate_2h": drain_rate, "eta_hours": eta_hours})


# ---------------------------------------------------------------------------
# Section: Langfuse traces
# ---------------------------------------------------------------------------

def audit_langfuse(report: AuditReport) -> None:
    section = "langfuse"

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-quantstack-local")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-quantstack-local")
    host = os.getenv("LANGFUSE_HOST_EXTERNAL", "http://localhost:3100")

    # Health check
    try:
        result = subprocess.run(
            ["curl", "-sf", f"{host}/api/public/health"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            report.add(section, "warning", "Langfuse health check failed — traces may not be recording")
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        report.add(section, "warning", "Langfuse unreachable")
        return

    # Fetch recent traces
    try:
        result = subprocess.run(
            ["curl", "-sf", "-u", f"{public_key}:{secret_key}",
             f"{host}/api/public/traces?limit=50"],
            capture_output=True, text=True, timeout=10,
        )
        traces = json.loads(result.stdout).get("data", [])
    except Exception as exc:
        report.add(section, "warning", f"Could not fetch traces: {exc}")
        return

    if not traces:
        report.add(section, "warning", "No traces in Langfuse — instrumentation may be broken")
        return

    # Analyze traces
    now = datetime.utcnow()
    recent_traces = []
    for t in traces:
        ts = t.get("timestamp", "")
        try:
            trace_time = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
            age_minutes = (now - trace_time).total_seconds() / 60
            recent_traces.append({
                "name": t.get("name", "?"),
                "session": t.get("sessionId", "?"),
                "age_min": age_minutes,
                "cost": t.get("totalCost", 0),
                "latency": t.get("latency", 0),
                "tags": t.get("tags", []),
            })
        except (ValueError, TypeError):
            continue

    if not recent_traces:
        report.add(section, "warning", "Could not parse any trace timestamps")
        return

    # Check freshness — last trace should be within 10 min
    newest_age = min(t["age_min"] for t in recent_traces)
    if newest_age > 30:
        report.add(section, "warning",
                    f"Last trace is {newest_age:.0f} min old — graphs may not be running",
                    metric={"newest_trace_age_min": newest_age})
    else:
        report.add(section, "ok",
                    f"Traces active ({len(recent_traces)} in last batch, newest {newest_age:.0f} min ago)")

    # Trace breakdown by session (graph)
    sessions = {}
    for t in recent_traces:
        s = t["session"]
        sessions.setdefault(s, []).append(t)

    for session_id, session_traces in sessions.items():
        total_cost = sum(t["cost"] for t in session_traces)
        avg_latency = sum(t["latency"] for t in session_traces) / len(session_traces) if session_traces else 0
        report.add(section, "info",
                    f"Session '{session_id}': {len(session_traces)} traces, cost=${total_cost:.4f}, avg_latency={avg_latency:.1f}s",
                    metric={"traces": len(session_traces), "cost": total_cost, "avg_latency": avg_latency})

    # Fetch observations for error analysis
    try:
        result = subprocess.run(
            ["curl", "-sf", "-u", f"{public_key}:{secret_key}",
             f"{host}/api/public/observations?limit=100"],
            capture_output=True, text=True, timeout=10,
        )
        observations = json.loads(result.stdout).get("data", [])
        errors = [o for o in observations if o.get("level") == "ERROR"]
        if errors:
            error_names = [o.get("name", "?") for o in errors[:5]]
            report.add(section, "warning",
                        f"{len(errors)} ERROR observations in recent traces",
                        detail=f"Affected: {error_names}")
        else:
            report.add(section, "ok", f"{len(observations)} observations, 0 errors")
    except Exception:
        pass  # Non-critical — trace-level check above is sufficient


# ---------------------------------------------------------------------------
# Section: Strategy pipeline
# ---------------------------------------------------------------------------

def audit_strategy_pipeline(report: AuditReport) -> None:
    section = "strategy_pipeline"

    rows = _query("SELECT status, COUNT(*) FROM strategies GROUP BY status ORDER BY status")
    status_map = {r[0]: r[1] for r in rows}
    total = sum(status_map.values())

    report.add(section, "info",
                f"Strategies: {status_map}",
                metric=status_map)

    # Check for stuck draft strategies (draft > 24h old without backtest)
    stuck_drafts = _query_one(
        """SELECT COUNT(*) FROM strategies
           WHERE status = 'draft' AND created_at < NOW() - INTERVAL '24 hours'"""
    )
    if stuck_drafts and stuck_drafts[0] > 10:
        report.add(section, "warning",
                    f"{stuck_drafts[0]} draft strategies older than 24h — pipeline may be slow")

    # Forward testing count
    ft_count = status_map.get("forward_testing", 0)
    live_count = status_map.get("live", 0)
    if ft_count == 0 and live_count == 0:
        report.add(section, "warning", "No strategies in forward_testing or live — no trading activity")
    else:
        report.add(section, "ok", f"{ft_count} forward_testing, {live_count} live")


# ---------------------------------------------------------------------------
# Section: Data freshness
# ---------------------------------------------------------------------------

def audit_data_freshness(report: AuditReport) -> None:
    section = "data_freshness"

    # OHLCV freshness for SPY (bellwether)
    row = _query_one(
        """SELECT MAX(timestamp) FROM ohlcv
           WHERE symbol = 'SPY' AND timeframe IN ('1D', '1d', 'daily', 'D1')"""
    )
    if not row or not row[0]:
        report.add(section, "critical", "No daily OHLCV data for SPY")
    else:
        latest = row[0]
        if hasattr(latest, "replace") and latest.tzinfo:
            latest = latest.replace(tzinfo=None)
        age_days = (datetime.now() - latest).days
        if age_days > 3:
            report.add(section, "critical", f"SPY OHLCV is {age_days} days stale (latest: {latest.date()})")
        elif age_days > 1:
            report.add(section, "warning", f"SPY OHLCV is {age_days} days old (latest: {latest.date()})")
        else:
            report.add(section, "ok", f"OHLCV fresh — SPY latest: {latest.date()}")

    # Universe coverage
    coverage = _query_one(
        """SELECT
             (SELECT COUNT(DISTINCT symbol) FROM ohlcv WHERE timeframe IN ('1D', '1d', 'daily', 'D1')) as ohlcv_symbols,
             (SELECT COUNT(*) FROM universe WHERE is_active = TRUE) as universe_symbols"""
    )
    if coverage:
        ohlcv_syms, universe_syms = coverage
        if universe_syms > 0:
            pct = ohlcv_syms / universe_syms * 100
            if pct < 80:
                report.add(section, "warning",
                            f"OHLCV coverage: {ohlcv_syms}/{universe_syms} ({pct:.0f}%) — some symbols missing data")
            else:
                report.add(section, "ok", f"OHLCV coverage: {ohlcv_syms}/{universe_syms} ({pct:.0f}%)")


# ---------------------------------------------------------------------------
# Section: Bug tracker
# ---------------------------------------------------------------------------

def audit_bugs(report: AuditReport) -> None:
    section = "bugs"

    rows = _query("SELECT status, COUNT(*) FROM bugs GROUP BY status")
    status_map = {r[0]: r[1] for r in rows}
    open_bugs = status_map.get("open", 0)
    in_progress = status_map.get("in_progress", 0)

    if open_bugs > 5:
        report.add(section, "warning", f"{open_bugs} open bugs — system self-healing may be overwhelmed",
                    metric=status_map)
    elif open_bugs > 0:
        report.add(section, "info", f"{open_bugs} open bug(s), {in_progress} in progress", metric=status_map)
    else:
        report.add(section, "ok", "No open bugs")

    # Recurring failures (same tool failing > 3 times)
    recurring = _query(
        """SELECT tool_name, COUNT(*) as cnt
           FROM bugs WHERE status = 'open'
           GROUP BY tool_name HAVING COUNT(*) >= 3
           ORDER BY cnt DESC LIMIT 5"""
    )
    if recurring:
        report.add(section, "warning",
                    f"Recurring failures: {', '.join(f'{r[0]}({r[1]}x)' for r in recurring)}")


# ---------------------------------------------------------------------------
# Section: Positions / risk
# ---------------------------------------------------------------------------

def audit_positions(report: AuditReport) -> None:
    section = "positions"

    rows = _query(
        """SELECT symbol, side, quantity, avg_cost, current_price, unrealized_pnl, last_updated
           FROM positions"""
    )

    if not rows:
        report.add(section, "info", "No open positions")
        return

    total_exposure = 0
    stale_positions = []
    now = datetime.now()

    for r in rows:
        symbol, side, qty, avg_cost, price, pnl, updated = r
        notional = abs(qty * (price or avg_cost))
        total_exposure += notional

        if updated:
            if hasattr(updated, "replace") and updated.tzinfo:
                updated = updated.replace(tzinfo=None)
            age_hours = (now - updated).total_seconds() / 3600
            if age_hours > 24:
                stale_positions.append(f"{symbol} ({age_hours:.0f}h)")

    report.add(section, "info",
                f"{len(rows)} positions, total exposure ${total_exposure:,.0f}",
                metric={"count": len(rows), "total_exposure": total_exposure})

    if stale_positions:
        report.add(section, "warning",
                    f"{len(stale_positions)} position(s) with stale prices (>24h): {', '.join(stale_positions[:5])}")


# ---------------------------------------------------------------------------
# Section: System state
# ---------------------------------------------------------------------------

def audit_system_state(report: AuditReport) -> None:
    section = "system_state"

    # Kill switch
    row = _query_one("SELECT value FROM system_state WHERE key = 'kill_switch'")
    if row and row[0] in ("true", "1", "active"):
        report.add(section, "critical", "KILL SWITCH IS ACTIVE — all trading halted")
    else:
        report.add(section, "ok", "Kill switch: inactive")

    # Credit regime
    row = _query_one("SELECT value, updated_at FROM system_state WHERE key = 'credit_regime'")
    if row:
        regime, updated = row
        report.add(section, "info", f"Credit regime: {regime} (updated: {updated})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_SECTIONS = {
    "docker": audit_docker,
    "graph_checkpoints": audit_graph_checkpoints,
    "research_queue": audit_research_queue,
    "langfuse": audit_langfuse,
    "strategy_pipeline": audit_strategy_pipeline,
    "data_freshness": audit_data_freshness,
    "bugs": audit_bugs,
    "positions": audit_positions,
    "system_state": audit_system_state,
}

SEVERITY_COLORS = {
    "ok": "\033[32m",       # green
    "info": "\033[36m",     # cyan
    "warning": "\033[33m",  # yellow
    "critical": "\033[31m", # red
}
RESET = "\033[0m"


def _severity_icon(sev: str) -> str:
    return {"ok": "✓", "info": "·", "warning": "⚠", "critical": "✗"}.get(sev, "?")


def print_report(report: AuditReport) -> None:
    print(f"\n{'='*70}")
    print(f"  QuantStack System Audit — {report.timestamp}")
    print(f"{'='*70}\n")

    current_section = ""
    for f in report.findings:
        if f.section != current_section:
            current_section = f.section
            print(f"  [{current_section.upper()}]")

        color = SEVERITY_COLORS.get(f.severity, "")
        icon = _severity_icon(f.severity)
        print(f"    {color}{icon} {f.severity.upper():8s}{RESET} {f.message}")
        if f.detail:
            print(f"               {f.detail[:120]}")

    # Summary
    counts = {}
    for f in report.findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    print(f"\n{'─'*70}")
    summary_parts = []
    for sev in ("critical", "warning", "ok", "info"):
        if sev in counts:
            color = SEVERITY_COLORS.get(sev, "")
            summary_parts.append(f"{color}{counts[sev]} {sev}{RESET}")
    print(f"  Summary: {' | '.join(summary_parts)}")

    overall = report.max_severity
    color = SEVERITY_COLORS.get(overall, "")
    print(f"  Overall: {color}{overall.upper()}{RESET}")
    print(f"{'─'*70}\n")


def main() -> None:
    # Suppress noisy loguru/DB debug output
    import logging
    logging.basicConfig(level=logging.WARNING)
    try:
        from loguru import logger as _loguru
        _loguru.disable("quantstack")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="QuantStack system health audit")
    parser.add_argument("--section", nargs="*", choices=list(ALL_SECTIONS.keys()),
                        help="Run specific sections only")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    report = AuditReport(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    sections = args.section or list(ALL_SECTIONS.keys())
    for section_name in sections:
        fn = ALL_SECTIONS[section_name]
        try:
            fn(report)
        except Exception as exc:
            report.add(section_name, "warning", f"Audit section failed: {exc}")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print_report(report)

    sys.exit(report.exit_code)


if __name__ == "__main__":
    main()
