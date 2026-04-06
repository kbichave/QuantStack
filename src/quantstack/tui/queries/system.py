"""System-level queries: kill switch, AV usage, regime, graph checkpoints, heartbeats, agent events."""
from __future__ import annotations

import json
import socket
import subprocess
from dataclasses import dataclass
from datetime import date, datetime

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class RegimeState:
    symbol: str
    trend: str
    volatility: str
    confidence: float


@dataclass
class GraphCheckpoint:
    graph_name: str
    node_name: str
    cycle_number: int
    started_at: datetime
    duration_seconds: float | None


@dataclass
class Heartbeat:
    loop_name: str
    last_beat: datetime
    status: str


@dataclass
class AgentEvent:
    graph_name: str
    node_name: str
    agent_name: str
    event_type: str
    content: str
    created_at: datetime


@dataclass
class ServiceHealth:
    name: str
    status: str  # "running" | "down" | "unknown"
    port: int | None


KNOWN_SERVICES: list[tuple[str, int]] = [
    ("postgres", 5432),
    ("langfuse", 3100),
    ("ollama", 11434),
]


def _probe_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def fetch_docker_health() -> list[ServiceHealth]:
    """Check health of Docker Compose services.

    Strategy:
    1. docker compose ps --format json (local dev)
    2. TCP port probes (inside container / no Docker CLI)
    3. Return 'unknown' status (neither method available)
    """
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            services = []
            for line in result.stdout.strip().splitlines():
                try:
                    svc = json.loads(line)
                    name = svc.get("Service", svc.get("Name", "unknown"))
                    state = svc.get("State", "unknown")
                    status = "running" if state in ("running", "healthy") else "down"
                    services.append(ServiceHealth(name=name, status=status, port=None))
                except (json.JSONDecodeError, KeyError):
                    continue
            if services:
                return services
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Fallback: TCP port probes
    probed = []
    any_result = False
    for name, port in KNOWN_SERVICES:
        alive = _probe_port("localhost", port)
        if alive:
            any_result = True
        probed.append(ServiceHealth(name=name, status="running" if alive else "down", port=port))
    if any_result:
        return probed

    # Neither method worked
    return [ServiceHealth(name=name, status="unknown", port=port) for name, port in KNOWN_SERVICES]


def fetch_kill_switch(conn: PgConnection) -> bool:
    """Return True if kill switch is active (system halted), False otherwise."""
    try:
        conn.execute("SELECT value FROM system_state WHERE key = 'kill_switch'")
        row = conn.fetchone()
        return row is not None and row[0] == "active"
    except Exception:
        logger.warning("fetch_kill_switch failed", exc_info=True)
        return False


def fetch_av_calls(conn: PgConnection) -> int:
    """Return today's Alpha Vantage API call count."""
    try:
        key = f"av_calls_{date.today().isoformat()}"
        conn.execute("SELECT value FROM system_state WHERE key = %s", (key,))
        row = conn.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        logger.warning("fetch_av_calls failed", exc_info=True)
        return 0


def fetch_regime(conn: PgConnection) -> RegimeState | None:
    """Return the latest market regime assessment."""
    try:
        conn.execute(
            "SELECT symbol, trend_regime, volatility_regime, confidence "
            "FROM regime_states ORDER BY timestamp DESC LIMIT 1"
        )
        row = conn.fetchone()
        if not row:
            return None
        return RegimeState(symbol=row[0], trend=row[1] or "unknown", volatility=row[2] or "normal", confidence=float(row[3] or 0))
    except Exception:
        logger.warning("fetch_regime failed", exc_info=True)
        return None


def fetch_graph_checkpoints(conn: PgConnection) -> list[GraphCheckpoint]:
    """Return the latest checkpoint per graph."""
    try:
        conn.execute(
            "SELECT DISTINCT ON (graph_name) graph_name, status, cycle_number, "
            "created_at, duration_seconds FROM graph_checkpoints "
            "ORDER BY graph_name, created_at DESC"
        )
        return [
            GraphCheckpoint(
                graph_name=r[0], node_name=r[1] or "unknown", cycle_number=r[2] or 0,
                started_at=r[3], duration_seconds=r[4],
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_graph_checkpoints failed", exc_info=True)
        return []


def fetch_heartbeats(conn: PgConnection) -> list[Heartbeat]:
    """Return the latest heartbeat per loop."""
    try:
        conn.execute(
            "SELECT DISTINCT ON (loop_name) loop_name, "
            "COALESCE(finished_at, started_at) AS last_beat, status "
            "FROM loop_heartbeats ORDER BY loop_name, started_at DESC"
        )
        return [
            Heartbeat(loop_name=r[0], last_beat=r[1], status=r[2] or "unknown")
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_heartbeats failed", exc_info=True)
        return []


def fetch_agent_events(conn: PgConnection, limit: int = 60) -> list[AgentEvent]:
    """Return recent agent events, newest first."""
    try:
        conn.execute(
            "SELECT graph_name, node_name, agent_name, event_type, content, created_at "
            "FROM agent_events ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        return [
            AgentEvent(
                graph_name=r[0], node_name=r[1], agent_name=r[2],
                event_type=r[3], content=r[4], created_at=r[5],
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_agent_events failed", exc_info=True)
        return []
