"""Agent activity queries: graph state, cycle history, skills, calibration, prompt versions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class GraphActivity:
    graph_name: str
    current_node: str
    current_agent: str
    cycle_number: int
    cycle_started: datetime
    event_count: int


@dataclass
class CycleHistory:
    graph_name: str
    cycle_number: int
    duration_seconds: float
    primary_agent: str
    tool_count: int


@dataclass
class AgentSkill:
    agent_name: str
    accuracy: float | None
    win_rate: float | None
    avg_pnl: float | None
    information_coefficient: float | None
    trend: str


@dataclass
class CalibrationRecord:
    agent_name: str
    stated_confidence: float
    actual_win_rate: float
    is_overconfident: bool


@dataclass
class PromptVersion:
    agent_name: str
    version: int
    optimized_at: datetime
    active_candidates: int


def fetch_graph_activity(conn: PgConnection) -> list[GraphActivity]:
    """Return current state per graph using checkpoints + recent event counts."""
    try:
        conn.execute(
            "SELECT gc.graph_name, gc.status, "
            "gc.cycle_number, gc.created_at, "
            "COALESCE(ae.cnt, 0) "
            "FROM (SELECT DISTINCT ON (graph_name) * FROM graph_checkpoints "
            "  ORDER BY graph_name, created_at DESC) gc "
            "LEFT JOIN (SELECT graph_name, COUNT(*) AS cnt FROM agent_events "
            "  WHERE created_at >= NOW() - interval '1 hour' "
            "  GROUP BY graph_name) ae ON ae.graph_name = gc.graph_name"
        )
        return [
            GraphActivity(
                graph_name=r[0], current_node=r[1] or "unknown",
                current_agent=r[1] or "unknown",
                cycle_number=int(r[2] or 0), cycle_started=r[3], event_count=int(r[4]),
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_graph_activity failed", exc_info=True)
        return []


def fetch_cycle_history(conn: PgConnection, limit: int = 3) -> list[CycleHistory]:
    """Return last N completed cycles per graph."""
    try:
        conn.execute(
            "SELECT graph_name, cycle_number, "
            "COALESCE(duration_seconds, 0), "
            "status, 0 "
            "FROM graph_checkpoints WHERE duration_seconds IS NOT NULL "
            "ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        return [
            CycleHistory(
                graph_name=r[0], cycle_number=int(r[1] or 0),
                duration_seconds=float(r[2]), primary_agent=r[3] or "unknown",
                tool_count=int(r[4]),
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_cycle_history failed", exc_info=True)
        return []


def fetch_agent_skills(conn: PgConnection) -> list[AgentSkill]:
    """Return agent skill metrics. Computes accuracy and win rate from raw counts."""
    try:
        conn.execute(
            "SELECT agent_id, prediction_count, correct_predictions, "
            "signal_count, winning_signals, total_signal_pnl "
            "FROM agent_skills ORDER BY agent_id"
        )
        results = []
        for r in conn.fetchall():
            pred = int(r[1] or 0)
            correct = int(r[2] or 0)
            signals = int(r[3] or 0)
            wins = int(r[4] or 0)
            total_pnl = float(r[5] or 0)
            accuracy = correct / pred if pred > 0 else None
            win_rate = wins / signals if signals > 0 else None
            avg_pnl = total_pnl / signals if signals > 0 else None
            results.append(AgentSkill(
                agent_name=r[0],
                accuracy=accuracy,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                information_coefficient=None,
                trend="stable",
            ))
        return results
    except Exception:
        logger.warning("fetch_agent_skills failed", exc_info=True)
        return []


def fetch_calibration(conn: PgConnection) -> list[CalibrationRecord]:
    """Return calibration records with overconfidence flag. Aggregates from raw records."""
    try:
        conn.execute(
            "SELECT agent_name, AVG(stated_confidence) AS avg_conf, "
            "AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) AS actual_wr "
            "FROM calibration_records GROUP BY agent_name ORDER BY agent_name"
        )
        return [
            CalibrationRecord(
                agent_name=r[0],
                stated_confidence=float(r[1]),
                actual_win_rate=float(r[2]),
                is_overconfident=float(r[1]) > float(r[2]) + 0.1,
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_calibration failed", exc_info=True)
        return []


def fetch_prompt_versions(conn: PgConnection) -> list[PromptVersion]:
    """Return prompt version info per agent."""
    try:
        conn.execute(
            "SELECT node_name, version, created_at, "
            "(SELECT COUNT(*) FROM prompt_candidates pc "
            " WHERE pc.node_name = pv.node_name AND pc.status = 'active') "
            "FROM prompt_versions pv "
            "WHERE (node_name, version) IN ("
            "  SELECT node_name, MAX(version) FROM prompt_versions GROUP BY node_name"
            ") ORDER BY node_name"
        )
        return [
            PromptVersion(
                agent_name=r[0], version=int(r[1]),
                optimized_at=r[2], active_candidates=int(r[3] or 0),
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_prompt_versions failed", exc_info=True)
        return []
